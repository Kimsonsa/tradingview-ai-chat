"""TradingView 창 자동 감지 및 캡쳐 모듈
- GPU 가속 앱(Electron) 호환: 화면 직접 캡쳐 방식
- SetForegroundWindow 제한 우회
- 창 타이틀에서 종목/타임프레임 자동 감지
"""
import io
import re
import time
import base64
import pyautogui
from PIL import Image

import win32gui
import win32con


def _is_tradingview_window(title):
    """TradingView 앱 창인지 판별"""
    t = title.lower()
    # TradingView 앱 타이틀 패턴: 'ETHUSDT.P ▲ 2,326.88 +0.47%' 형태
    if re.search(r'[A-Z]{2,10}USDT', title, re.IGNORECASE):
        return True
    if 'tradingview' in t:
        return True
    if 'binance' in t and ('chart' in t or 'perpetual' in t):
        return True
    return False


def find_tradingview_window():
    """TradingView 데스크탑 앱 창을 찾아 핸들 반환"""
    results = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and _is_tradingview_window(title):
                results.append((hwnd, title))
        return True

    win32gui.EnumWindows(callback, None)

    if not results:
        return None, None

    # 'tradingview-ai-chat' 같은 터미널/에디터 창 제외
    exclude = ['antigravity', 'powershell', 'vscode', 'visual studio', 'extension:']
    filtered = [(h, t) for h, t in results
                if not any(ex in t.lower() for ex in exclude)]

    if not filtered:
        filtered = results

    # 가장 큰 창 선택
    best = None
    best_area = 0
    for hwnd, title in filtered:
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
    """TradingView 창 타이틀에서 종목과 타임프레임 추출"""
    symbol = None
    interval_label = None

    # 종목 추출: XXXUSDT 패턴
    sym_match = re.search(r'([A-Z]{2,10}USDT)(?:\.P)?', title.upper())
    if sym_match:
        symbol = sym_match.group(1)

    # 타임프레임 추출: 쉼표 구분 형식
    parts = [p.strip() for p in title.split(',')]
    for part in parts:
        clean = part.strip()
        if clean in INTERVAL_PARSE_MAP:
            interval_label = INTERVAL_PARSE_MAP[clean]
            break

    # 타임프레임: · 구분 형식
    if not interval_label:
        tf_match = re.search(r'(\d+[mhDWM]|\d+분|\d+시간|\d+일)', title)
        if tf_match:
            tf = tf_match.group(1)
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


def detect_chart_info():
    """TradingView 창에서 현재 종목/타임프레임 정보만 가져오기 (캡쳐 없이)"""
    result = find_tradingview_window()
    if result is None or result[0] is None:
        return None, None, None

    hwnd, title = result
    symbol, interval = parse_window_title(title)
    return symbol, interval, title


def _bring_to_front(hwnd):
    """창을 최전면으로 (SetForegroundWindow 제한 우회)"""
    try:
        # 최소화 상태면 복원
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.3)

        # Alt키 트릭으로 포그라운드 잠금 해제
        import ctypes
        ctypes.windll.user32.keybd_event(0x12, 0, 0, 0)  # Alt down
        ctypes.windll.user32.keybd_event(0x12, 0, 2, 0)  # Alt up
        time.sleep(0.05)

        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.4)
        return True
    except Exception:
        # 실패해도 계속 진행 (side-by-side 배치면 캡쳐 가능)
        time.sleep(0.2)
        return False


def capture_tradingview():
    """TradingView 창을 자동으로 찾아 캡쳐.
    반환: (image, window_title) 또는 (None, error_msg)
    """
    result = find_tradingview_window()
    if result is None or result[0] is None:
        return None, "TradingView 앱을 찾을 수 없습니다. TradingView 데스크탑 앱이 실행 중인지 확인하세요."

    hwnd, title = result

    try:
        # 창을 최전면으로 시도 (실패해도 계속)
        _bring_to_front(hwnd)

        # 창 영역 가져오기
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y

        if w < 100 or h < 100:
            return None, "TradingView 창이 너무 작습니다."

        # 화면에서 직접 해당 영역 캡쳐 (멀티 모니터 지원)
        from PIL import ImageGrab
        img = ImageGrab.grab(bbox=(x, y, x2, y2), all_screens=True)

        return img, title

    except Exception as e:
        return None, f"캡쳐 실패: {str(e)}"


def image_to_base64(img, max_size=1600):
    """PIL Image를 base64 문자열로 변환 (JPEG 압축)"""
    # 리사이즈
    if img.width > max_size or img.height > max_size:
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # RGB로 변환
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85, optimize=True)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64, len(buffer.getvalue())
