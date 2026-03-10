import cv2
import numpy as np
import pyautogui
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

 # değerler
WINDOW_NAME = "mouse movement cam control"

# Farneback parametreleri
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=30,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# hassasiyet
MOUSE_SENSITIVITY = 2.5
# titreme yumuşatma
SMOOTHING = 0.7
# min piksel hareketi
MOVEMENT_THRESHOLD = 1.5

# sallama
SHAKE_WINDOW = 2
# min yön değişimi
SHAKE_REVERSAL_COUNT = 1
# sallama sayılması için gereken min hareket sayısı
SHAKE_MIN_DX = 2.0
# tıklama sonrası bekleme sayısı - çift tıklama olmaması için
CLICK_COOLDOWN = 0.8

@dataclass
class TrackingState:
    """Takip durumu"""
    prev_gray: Optional[np.ndarray] = None
    smoothed_dx: float = 0.0
    smoothed_dy: float = 0.0
    frame_count: int = 0
    fps_time: float = field(default_factory=time.time)
    fps: float = 0.0
    # sallama için son karedeki N değeri
    dx_history: deque = field(default_factory=lambda: deque(maxlen=SHAKE_WINDOW))
    last_click_time: float = 0.0
    click_flash: int = 0

# Farneback Optik Akış
def farneback_flow(
    prev_gray: np.ndarray,                                  # önceki kare
    curr_gray: np.ndarray,                                  # geçerli kare
) -> tuple[np.ndarray, np.ndarray]:
    """ Farneback optik akışı hesaplar """
    flow = cv2.calcOpticalFlowFarneback(                    # her piksel için akış takibi
        prev_gray, curr_gray, None, **FARNEBACK_PARAMS
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # akış büyüklüğü haritası
    return flow, magnitude

# Hareket Hesaplama
def compute_flow_motion(
    flow: np.ndarray,
    magnitude: np.ndarray,
    threshold_percentile: float = 75,                       # gürültüyü düşürmek için büyüklük eşiği yüzdesi
) -> tuple[float, float]:
    """ Yoğun optik akış haritasından baskın hareketi çıkarır. """
    thresh = np.percentile(magnitude, threshold_percentile)
    mask = magnitude > thresh
    if not np.any(mask):
        return 0.0, 0.0
    dx = float(np.mean(flow[mask, 0]))
    dy = float(np.mean(flow[mask, 1]))
    return dx, dy

def smooth_motion(
    state: TrackingState,                # geçerli takip durumu
    raw_dx: float,                       # yatay hareket
    raw_dy: float,                       # dikey hareket
) -> tuple[float, float]:
    """ Üstel hareketli ortalama ile fare titremesini azaltır. """
    state.smoothed_dx = SMOOTHING * state.smoothed_dx + (1 - SMOOTHING) * raw_dx
    state.smoothed_dy = SMOOTHING * state.smoothed_dy + (1 - SMOOTHING) * raw_dy
    return state.smoothed_dx, state.smoothed_dy

# Sallama Tespiti
def detect_shake(state: TrackingState, raw_dx: float) -> bool:
    """ Son SHAKE_WINDOW karedeki dx geçmişinde hızlı sağ-sol sallama olup olmadığını tespit eder. """
    state.dx_history.append(raw_dx)

    # yalnızca güçlü hareketleri filtreler
    strong = [v for v in state.dx_history if abs(v) >= SHAKE_MIN_DX]

    if len(strong) < SHAKE_REVERSAL_COUNT + 1:
        return False

    # Ardışık işaret değişimlerini say
    reversals = sum(
        1 for i in range(1, len(strong))
        if strong[i] * strong[i - 1] < 0   # zıt işaretler
    )

    if reversals >= SHAKE_REVERSAL_COUNT:
        now = time.time()
        if now - state.last_click_time >= CLICK_COOLDOWN:
            state.last_click_time = now
            state.dx_history.clear()
            return True

    return False

def move_mouse(dx: float, dy: float) -> None:
    """ Fare Kontrolü """
    if abs(dx) < MOVEMENT_THRESHOLD and abs(dy) < MOVEMENT_THRESHOLD:
        return
    screen_w, screen_h = pyautogui.size()
    cur_x, cur_y = pyautogui.position()
    new_x = int(np.clip(cur_x + dx * MOUSE_SENSITIVITY, 0, screen_w - 1))
    new_y = int(np.clip(cur_y + dy * MOUSE_SENSITIVITY, 0, screen_h - 1))
    pyautogui.moveTo(new_x, new_y, _pause=False)

def left_click() -> None:
    """Sol fare tıklaması yapar."""
    pyautogui.click(_pause=False)

def draw_farneback(
    frame: np.ndarray,
    flow: np.ndarray,
    step: int = 16,
) -> np.ndarray:
    """Farneback akışını ızgara vektörleriyle görselleştirir."""
    vis = frame.copy()
    h, w = frame.shape[:2]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    fx = flow[ys, xs, 0]
    fy = flow[ys, xs, 1]
    for x, y, dx, dy in zip(xs.ravel(), ys.ravel(), fx.ravel(), fy.ravel()):
        ex, ey = int(x + dx * 3), int(y + dy * 3)
        cv2.arrowedLine(vis, (int(x), int(y)), (ex, ey), (255, 100, 0), 1, tipLength=0.5)
    return vis


def draw_flow_hsv(flow: np.ndarray) -> np.ndarray:
    """Farneback akışını HSV renk haritasıyla görselleştirir."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_hud(
    frame: np.ndarray,
    dx: float,
    dy: float,
    fps: float,
    click_flash: int,
) -> np.ndarray:
    """Ekranda bilgi paneli çizer."""
    vis = frame.copy()
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (320, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

    click_label = "*** SOL TIK! ***" if click_flash > 0 else "Sallama = Sol Tik"
    click_color = (0, 255, 0) if click_flash > 0 else (150, 150, 150)

    lines = [
        ("Mod: Farneback Dense Optik Akis", (255, 100, 0)),
        (f"FPS: {fps:.1f}",                 (200, 200, 200)),
        (f"Hareket: dx={dx:+.1f}  dy={dy:+.1f}", (200, 200, 200)),
        (click_label,                        click_color),
        ("q / ESC  ->  Cikis",              (150, 150, 150)),
    ]
    for i, (text, col) in enumerate(lines):
        cv2.putText(vis, text, (10, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
    return vis

def update_fps(state: TrackingState) -> None:
    """ FPS sayacını günceller."""
    state.frame_count += 1
    elapsed = time.time() - state.fps_time
    if elapsed >= 1.0:
        state.fps = state.frame_count / elapsed
        state.frame_count = 0
        state.fps_time = time.time()

def run(camera_index: int = 0) -> None:
    """ Uygulamayı başlatır ve ana döngüyü yürütür. """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_index} açılamadı.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pyautogui.FAILSAFE = False

    state = TrackingState()

    print("=" * 55)
    print("  Farneback Dense Optik Akış  –  Başlatılıyor…")
    print("=" * 55)
    print("  Hızlı yatay sallama  →  Sol Tık")
    print("  q / ESC              →  Çıkış")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("HATA! Kare okunamadı.")
            break

        frame = cv2.flip(frame, 1)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        update_fps(state)

        # ilk kare başlatma
        if state.prev_gray is None:
            state.prev_gray = curr_gray.copy()
            continue

        # Farneback akışı
        flow, magnitude = farneback_flow(state.prev_gray, curr_gray)
        raw_dx, raw_dy = compute_flow_motion(flow, magnitude)

        # Sallama tespiti - Sol tık
        if detect_shake(state, raw_dx):
            left_click()
            state.click_flash = 8
            print("TIKLAMA! Sol tık!")
        elif state.click_flash > 0:
            state.click_flash -= 1

        # Görselleştirme
        display_frame = draw_farneback(frame, flow)

        # HSV haritasını sağ alt köşeye ekler.
        h, w = frame.shape[:2]
        th, tw = h // 4, w // 4
        hsv_overlay = draw_flow_hsv(flow)
        display_frame[h - th:h, w - tw:w] = cv2.resize(hsv_overlay, (tw, th))

        # tıklama anında çerçeve rengini yeşile boyar.
        if state.click_flash > 0:
            cv2.rectangle(display_frame, (0, 0),
                          (display_frame.shape[1] - 1, display_frame.shape[0] - 1),
                          (0, 255, 0), 4)

        # fare hareketi
        sdx, sdy = smooth_motion(state, raw_dx, raw_dy)
        move_mouse(sdx, sdy)

        # HUD
        display_frame = draw_hud(display_frame, sdx, sdy, state.fps, state.click_flash)

        # sonraki kareye hazırlan
        state.prev_gray = curr_gray.copy()

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Uygulama kapatıldı.")


if __name__ == "__main__":
    run(camera_index=0)
