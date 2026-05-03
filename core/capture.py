"""
TradingView 창 자동 감지 및 캡쳐 모듈
- GPU 가속 앱(Electron) 호환: 화면 직접 캡쳐 방식
"""
import io
import time
import base64
import pyautogui
from PIL import Image

import win32gui
import win32con


def find_tradingview_window():
    """TradingView 데스크탑 앱 창을 찾아 핸들 반환"""
    results = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and "tradingview" in title.lower():
                results.append((hwnd, title))
        return True

    win32gui.EnumWindows(callback, None)

    if not results:
        return None, None

    # 가장 큰 창 선택 (메인 차트일 가능성 높음)
    best = None
    best_area = 0
    for hwnd, title in results:
        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        area = w * h
        if area > best_area and w > 200 and h > 200:
            best = (hwnd, title)
            best_area = area

    return best


def capture_tradingview():
    """TradingView 창을 자동으로 찾아 캡쳐.
    반환: (image, window_title) 또는 (None, error_msg)
    """
    result = find_tradingview_window()
    if result is None or result[0] is None:
        return None, "TradingView 앱을 찾을 수 없습니다. TradingView 데스크탑 앱이 실행 중인지 확인하세요."

    hwnd, title = result

    try:
        # TradingView 창을 최전면으로 가져오기
        # 최소화 상태면 복원
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.5)

        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.5)  # 창이 완전히 렌더링되도록 대기

        # 창 영역 가져오기
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y

        # 화면에서 직접 해당 영역 캡쳐 (GPU 가속 앱도 정상 캡쳐)
        img = pyautogui.screenshot(region=(x, y, w, h))

        return img, title

    except Exception as e:
        return None, f"캡쳐 실패: {str(e)}"


def image_to_base64(img, max_size=1920):
    """PIL Image를 base64 문자열로 변환 (리사이즈 포함)"""
    # 너무 크면 리사이즈
    if img.width > max_size or img.height > max_size:
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
