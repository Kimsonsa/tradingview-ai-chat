"""
TradingView 창 자동 감지 및 캡쳐 모듈
"""
import io
import ctypes
from ctypes import wintypes
from PIL import Image

import win32gui
import win32ui
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
        area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        if area > best_area:
            best = (hwnd, title)
            best_area = area

    return best


def capture_window(hwnd):
    """특정 윈도우 핸들의 화면을 캡쳐하여 PIL Image로 반환"""
    try:
        # 창을 포그라운드로 가져오기
        win32gui.SetForegroundWindow(hwnd)
        import time
        time.sleep(0.3)

        # 창 크기 가져오기
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y

        # 화면 캡쳐 (PrintWindow 사용)
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        # PrintWindow로 캡쳐 (DWM 합성 포함)
        ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)

        # 비트맵 → PIL Image
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        img = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr, "raw", "BGRX", 0, 1
        )

        # 정리
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        return img

    except Exception as e:
        # PrintWindow 실패 시 pyautogui fallback
        import pyautogui
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        img = pyautogui.screenshot(region=(x, y, x2 - x, y2 - y))
        return img


def capture_tradingview():
    """TradingView 창을 자동으로 찾아 캡쳐. 반환: (image, window_title) 또는 (None, error_msg)"""
    result = find_tradingview_window()
    if result is None or result[0] is None:
        return None, "TradingView 앱을 찾을 수 없습니다. TradingView 데스크탑 앱이 실행 중인지 확인하세요."

    hwnd, title = result
    img = capture_window(hwnd)
    if img is None:
        return None, "화면 캡쳐에 실패했습니다."

    return img, title


def image_to_base64(img, max_size=1920):
    """PIL Image를 base64 문자열로 변환 (리사이즈 포함)"""
    import base64

    # 너무 크면 리사이즈
    if img.width > max_size or img.height > max_size:
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
