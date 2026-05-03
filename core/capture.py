"""
TradingView 창 자동 감지 및 캡쳐 모듈
- GPU 가속 앱(Electron) 호환: 화면 직접 캡쳐 방식
"""
import io
import re
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


INTERVAL_PARSE_MAP = {
    '1': '1분', '3': '3분', '5': '5분', '15': '15분', '30': '30분',
    '45': '45분', '60': '1시간', '120': '2시간', '180': '3시간',
    '240': '4시간', '1D': '1일', 'D': '1일', '1W': '1주', 'W': '1주',
    '1M': '1개월', 'M': '1개월',
}

BINANCE_INTERVAL_MAP = {
    '1분': '1m', '3분': '3m', '5분': '5m', '15분': '15m', '30분': '30m',
    '45분': '45m', '1시간': '1h', '2시간': '2h', '3시간': '3h',
    '4시간': '4h', '1일': '1d', '1주': '1w', '1개월': '1M',
}


def parse_window_title(title):
    """TradingView 창 타이틀에서 종목과 타임프레임 추출
    예: 'BTCUSDT.P, 60, Binance — TradingView' → ('BTCUSDT', '1시간')
    예: 'BTCUSDT.P ▲ 78,661.4 +0.01 × 세 탭' → ('BTCUSDT', None)
    """
    symbol = None
    interval_label = None

    # 종목 추출: XXXUSDT 패턴
    sym_match = re.search(r'([A-Z]{2,10}USDT)(?:\.P)?', title.upper())
    if sym_match:
        symbol = sym_match.group(1)

    # 타임프레임 추출: 쉼표 구분 형식 (예: 'BTCUSDT.P, 60, Binance')
    parts = [p.strip() for p in title.split(',')]
    for part in parts:
        clean = part.strip()
        if clean in INTERVAL_PARSE_MAP:
            interval_label = INTERVAL_PARSE_MAP[clean]
            break

    # 타임프레임: 타이틀 내 · 구분 형식 (예: '1시간 · Binance')
    if not interval_label:
        tf_match = re.search(r'(\d+[mhDWM]|\d+분|\d+시간|\d+일)', title)
        if tf_match:
            tf = tf_match.group(1)
            # 1h, 4h 등
            h_match = re.match(r'(\d+)h', tf)
            m_match = re.match(r'(\d+)m', tf)
            if h_match:
                interval_label = f"{h_match.group(1)}시간"
            elif m_match:
                interval_label = f"{m_match.group(1)}분"
            elif tf in INTERVAL_PARSE_MAP:
                interval_label = INTERVAL_PARSE_MAP[tf]
            else:
                interval_label = tf

    return symbol, interval_label


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
