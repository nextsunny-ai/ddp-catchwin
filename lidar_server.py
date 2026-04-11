"""
ULTRA DEPARTMENT STORE - DUAL LIDAR SERVER (A1M8 x 2)
2대 라이다 → 사람 위치 감지 → WebSocket → 게임 HTML

설치:
  pip install rplidar-roboticia numpy websockets

실행:
  python lidar_server.py

라이다 1대만 있을 때:
  LIDAR_B_PORT = None 으로 설정하면 1대만 사용
"""

import asyncio
import json
import math
import time
import numpy as np
import websockets
from rplidar import RPLidar

# ══════════════════════════════════════════
#  CONFIG - 현장에 맞게 수정
# ══════════════════════════════════════════
CONFIG = {
    # 라이다 포트 (Windows: "COM7" 등)
    # 2대 사용: A=왼쪽, B=오른쪽
    # 1대만 사용하려면 LIDAR_B_PORT = None
    "LIDAR_A_PORT": "COM7",       # 라이다 A - 포트 확인 후 수정!
    "LIDAR_B_PORT": None,         # 라이다 B (오른쪽) - None이면 1대만 사용
    "LIDAR_A_BAUD": 115200,       # A1M8: 115200 / A2M12: 256000
    "LIDAR_B_BAUD": 256000,       # A1M8: 115200 / A2M12: 256000

    # WebSocket 포트
    "WS_PORT": 8765,

    # ── 벽면 & 설치 정보 ──
    "WALL_WIDTH_M": 12.686,
    "WALL_HEIGHT_M": 4.8,

    # 라이다 A 감지 영역 (미터, 라이다 중심 기준)
    "A_X_MIN": -4.0,
    "A_X_MAX":  4.0,
    "A_Y_MIN":  0.3,
    "A_Y_MAX":  5.0,

    # 라이다 A 오프셋 (벽면 전체 좌표계에서의 위치)
    "A_X_OFFSET": -3.0,
    "A_Y_OFFSET": 0.0,

    # 라이다 B 감지 영역
    "B_X_MIN": -4.0,
    "B_X_MAX":  4.0,
    "B_Y_MIN":  0.3,
    "B_Y_MAX":  5.0,

    # 라이다 B 오프셋
    "B_X_OFFSET": 3.0,
    "B_Y_OFFSET": 0.0,

    # ── 감지 파라미터 (A1M8 최적화) ──
    "CLUSTER_DIST": 0.7,  # 클러스터 거리 - A1M8 저해상도 대응 (기존 0.4)
    "MIN_POINTS": 1,      # 최소 포인트 - 먼 거리에서 포인트 1개도 감지 (기존 3)
    "MAX_PLAYERS": 6,

    # ── 스무딩 & 추적 ──
    "SMOOTHING": 0.4,       # 위치 스무딩 (0=즉시, 1=매우느림) - 떨림 방지
    "PLAYER_TIMEOUT": 2.0,  # 이 시간(초) 동안 감지 안 되면 플레이어 제거
    "MERGE_DIST": 1.0,      # 두 라이다의 같은 사람 병합 거리(m)
}

# ══════════════════════════════════════════
#  LIDAR 데이터 파싱
# ══════════════════════════════════════════
def parse_scan(scan, area_cfg, offset_x=0, offset_y=0):
    points = []
    for quality, angle, distance in scan:
        if quality == 0 or distance == 0:
            continue
        dist_m = distance / 1000.0
        angle_rad = math.radians(angle)
        x = dist_m * math.sin(angle_rad)
        y = dist_m * math.cos(angle_rad)

        if (area_cfg["x_min"] <= x <= area_cfg["x_max"] and
                area_cfg["y_min"] <= y <= area_cfg["y_max"]):
            points.append((x + offset_x, y + offset_y))
    return points


# ══════════════════════════════════════════
#  클러스터링 (BFS 방식 - 연결된 포인트 모두 수집)
# ══════════════════════════════════════════
def cluster_points(points, cluster_dist, min_points):
    if not points:
        return []

    clusters = []
    used = [False] * len(points)

    for i in range(len(points)):
        if used[i]:
            continue
        stack = [i]
        cluster = []
        while stack:
            idx = stack.pop()
            if used[idx]:
                continue
            used[idx] = True
            cluster.append(points[idx])
            for j in range(len(points)):
                if not used[j]:
                    if math.hypot(points[idx][0]-points[j][0],
                                  points[idx][1]-points[j][1]) < cluster_dist:
                        stack.append(j)

        if len(cluster) >= min_points:
            cx = sum(c[0] for c in cluster) / len(cluster)
            cy = sum(c[1] for c in cluster) / len(cluster)
            clusters.append((cx, cy, len(cluster)))

    clusters.sort(key=lambda c: -c[2])
    return [(c[0], c[1]) for c in clusters[:CONFIG["MAX_PLAYERS"]]]


# ══════════════════════════════════════════
#  두 라이다 클러스터 병합
# ══════════════════════════════════════════
def merge_clusters(clusters_a, clusters_b, merge_dist):
    if not clusters_b:
        return clusters_a
    if not clusters_a:
        return clusters_b

    merged = list(clusters_a)
    for bx, by in clusters_b:
        matched = False
        for i, (ax, ay) in enumerate(merged):
            if math.hypot(ax - bx, ay - by) < merge_dist:
                merged[i] = ((ax + bx) / 2, (ay + by) / 2)
                matched = True
                break
        if not matched:
            merged.append((bx, by))
    return merged[:CONFIG["MAX_PLAYERS"]]


# ══════════════════════════════════════════
#  플레이어 추적 & 스무딩
# ══════════════════════════════════════════
tracked_players = {}
next_player_id = 0

def update_tracking(clusters):
    global next_player_id
    now = time.time()
    sm = CONFIG["SMOOTHING"]
    match_dist = CONFIG["CLUSTER_DIST"] * 2

    used_clusters = [False] * len(clusters)
    used_players = set()

    # 거리 기반 매칭 (가장 가까운 것 우선)
    pairs = []
    for pid, pdata in tracked_players.items():
        for ci, (cx, cy) in enumerate(clusters):
            d = math.hypot(pdata["x"] - cx, pdata["y"] - cy)
            pairs.append((d, pid, ci))
    pairs.sort()

    for d, pid, ci in pairs:
        if pid in used_players or used_clusters[ci]:
            continue
        if d < match_dist:
            cx, cy = clusters[ci]
            tracked_players[pid]["x"] = tracked_players[pid]["x"] * sm + cx * (1 - sm)
            tracked_players[pid]["y"] = tracked_players[pid]["y"] * sm + cy * (1 - sm)
            tracked_players[pid]["last_seen"] = now
            used_players.add(pid)
            used_clusters[ci] = True

    # 새 플레이어
    for ci, (cx, cy) in enumerate(clusters):
        if not used_clusters[ci]:
            pid = f"lidar_{next_player_id}"
            next_player_id += 1
            tracked_players[pid] = {"x": cx, "y": cy, "last_seen": now}

    # 타임아웃 제거
    timeout = CONFIG["PLAYER_TIMEOUT"]
    expired = [pid for pid, p in tracked_players.items() if now - p["last_seen"] > timeout]
    for pid in expired:
        del tracked_players[pid]

    return [
        {"id": pid, "x": round(p["x"], 3), "y": round(p["y"], 3)}
        for pid, p in tracked_players.items()
    ]


# ══════════════════════════════════════════
#  배경 차감 (고정 물체 무시)
# ══════════════════════════════════════════
background_points = []
bg_calibrated = False
BG_TOLERANCE = 0.3  # 배경 포인트에서 이 거리(m) 이내면 무시

def filter_background(points):
    if not bg_calibrated or not background_points:
        return points
    filtered = []
    for px, py in points:
        is_bg = False
        for bx, by in background_points:
            if math.hypot(px - bx, py - by) < BG_TOLERANCE:
                is_bg = True
                break
        if not is_bg:
            filtered.append((px, py))
    return filtered

async def calibrate_background():
    global background_points, bg_calibrated
    print("[BG] 배경 스캔 시작 - 5초간 사람은 영역 밖으로!")
    all_points = []
    area_a = {
        "x_min": CONFIG["A_X_MIN"], "x_max": CONFIG["A_X_MAX"],
        "y_min": CONFIG["A_Y_MIN"], "y_max": CONFIG["A_Y_MAX"],
    }
    for _ in range(50):
        pts = parse_scan(list(latest_scan_a), area_a,
                         CONFIG["A_X_OFFSET"], CONFIG["A_Y_OFFSET"]) if latest_scan_a else []
        all_points.extend(pts)
        await asyncio.sleep(0.1)
    grid = {}
    for x, y in all_points:
        key = (round(x, 1), round(y, 1))
        if key not in grid:
            grid[key] = (x, y)
    background_points = list(grid.values())
    bg_calibrated = True
    print(f"[BG] 배경 스캔 완료 - {len(background_points)}개 고정 포인트 등록")

# ══════════════════════════════════════════
#  WebSocket
# ══════════════════════════════════════════
connected_clients = set()
lidar_reset_requested = False

async def ws_handler(websocket):
    global lidar_reset_requested
    connected_clients.add(websocket)
    print(f"[WS] 클라이언트 연결 ({len(connected_clients)}명)")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                cmd = data.get("command")
                if cmd == "calibrate":
                    asyncio.create_task(calibrate_background())
                elif cmd == "reset":
                    lidar_reset_requested = True
                    print("[CMD] 라이다 리셋 요청됨")
            except:
                pass
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] 클라이언트 연결 해제")

async def broadcast(data):
    if connected_clients:
        msg = json.dumps(data)
        await asyncio.gather(
            *[ws.send(msg) for ws in connected_clients],
            return_exceptions=True
        )


# ══════════════════════════════════════════
#  LIDAR 읽기 (각 라이다별 별도 태스크)
# ══════════════════════════════════════════
latest_scan_a = []
latest_scan_b = []

async def lidar_read_loop(port, baud, label, scan_storage):
    global lidar_reset_requested
    while True:
        print(f"[{label}] 연결 시도: {port} (baud: {baud})")
        try:
            lidar = RPLidar(port, baudrate=baud)
            lidar.clean_input()
            info = lidar.get_info()
            print(f"[{label}] 연결 성공: {info}")
            lidar.clean_input()
            health = lidar.get_health()
            print(f"[{label}] 상태: {health}")

            for scan in lidar.iter_scans():
                if lidar_reset_requested:
                    lidar_reset_requested = False
                    print(f"[{label}] 리셋 요청 — 재연결 중...")
                    break
                scan_storage.clear()
                scan_storage.extend(scan)
                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"[{label}] 오류: {e}")
        finally:
            try:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
            except:
                pass
            print(f"[{label}] 연결 종료 — 5초 후 재연결 시도")
            await asyncio.sleep(5)


async def processing_loop():
    area_a = {
        "x_min": CONFIG["A_X_MIN"], "x_max": CONFIG["A_X_MAX"],
        "y_min": CONFIG["A_Y_MIN"], "y_max": CONFIG["A_Y_MAX"],
    }
    area_b = {
        "x_min": CONFIG["B_X_MIN"], "x_max": CONFIG["B_X_MAX"],
        "y_min": CONFIG["B_Y_MIN"], "y_max": CONFIG["B_Y_MAX"],
    }

    frame_count = 0
    while True:
        points_a = parse_scan(list(latest_scan_a), area_a,
                              CONFIG["A_X_OFFSET"], CONFIG["A_Y_OFFSET"]) if latest_scan_a else []
        points_b = parse_scan(list(latest_scan_b), area_b,
                              CONFIG["B_X_OFFSET"], CONFIG["B_Y_OFFSET"]) if latest_scan_b else []

        points_a = filter_background(points_a)
        points_b = filter_background(points_b)

        clusters_a = cluster_points(points_a, CONFIG["CLUSTER_DIST"], CONFIG["MIN_POINTS"])
        clusters_b = cluster_points(points_b, CONFIG["CLUSTER_DIST"], CONFIG["MIN_POINTS"])
        merged = merge_clusters(clusters_a, clusters_b, CONFIG["MERGE_DIST"])
        player_list = update_tracking(merged)

        await broadcast({"players": player_list})

        frame_count += 1
        if frame_count % 20 == 0 and (points_a or points_b):
            coords = [(round(p["x"],2), round(p["y"],2)) for p in player_list]
            print(f"[SCAN] A:{len(points_a)}pts -> {len(merged)}cl -> {len(player_list)}p | pos={coords}")

        await asyncio.sleep(0.05)  # ~20fps


# ══════════════════════════════════════════
#  메인
# ══════════════════════════════════════════
async def main():
    print("=" * 55)
    print("  ULTRA DEPARTMENT STORE - DUAL LIDAR SERVER")
    print("=" * 55)
    print(f"  벽면: {CONFIG['WALL_WIDTH_M']}m x {CONFIG['WALL_HEIGHT_M']}m")
    print(f"  WebSocket: ws://localhost:{CONFIG['WS_PORT']}")
    print(f"  라이다 A: {CONFIG['LIDAR_A_PORT']}")
    print(f"  라이다 B: {CONFIG['LIDAR_B_PORT'] or '미사용 (1대 모드)'}")
    print(f"  감지: CLUSTER={CONFIG['CLUSTER_DIST']}m MIN_PTS={CONFIG['MIN_POINTS']}")
    print(f"  스무딩: {CONFIG['SMOOTHING']} / 타임아웃: {CONFIG['PLAYER_TIMEOUT']}s")
    print("=" * 55)

    ws_server = await websockets.serve(ws_handler, "0.0.0.0", CONFIG["WS_PORT"])
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"[WS] 서버 시작 - ws://0.0.0.0:{CONFIG['WS_PORT']}")
    print(f"[WS] 다른 PC에서 접속: ws://{local_ip}:{CONFIG['WS_PORT']}")

    tasks = [processing_loop()]

    if CONFIG["LIDAR_A_PORT"]:
        tasks.append(lidar_read_loop(
            CONFIG["LIDAR_A_PORT"], CONFIG["LIDAR_A_BAUD"],
            "LIDAR-A", latest_scan_a
        ))

    if CONFIG["LIDAR_B_PORT"]:
        tasks.append(lidar_read_loop(
            CONFIG["LIDAR_B_PORT"], CONFIG["LIDAR_B_BAUD"],
            "LIDAR-B", latest_scan_b
        ))

    if not CONFIG["LIDAR_A_PORT"] and not CONFIG["LIDAR_B_PORT"]:
        print("[DEMO] 라이다 미연결 - 데모모드 (게임에서 마우스 사용)")

    await asyncio.gather(ws_server.wait_closed(), *tasks)

if __name__ == "__main__":
    asyncio.run(main())
