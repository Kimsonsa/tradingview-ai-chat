"""TradingView 창 자동 감지 및 캡쳐 모듈 (Windows / Linux 크로스플랫폼)

Windows: win32gui + PIL.ImageGrab
Linux:   xdotool + scrot (subprocess)
"""
import io
import os
import re
import sys
import time
import base64
import subprocess
from PIL import Image

# ─── OS 판별 ───
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

# ─── Windows 전용 초기화 ───
if IS_WINDOWS:
    import ctypes
    import win32gui
    import win32con

    # 멀티모니터 DPI 스케일링 대응 — 모든 모니터에서 정확한 좌표 보장
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-Monitor DPI Aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # System DPI Aware (fallback)
        except Exception:
            pass


# ═══════════════════════════════════════════════
# 공통 상수 / 유틸
# ═══════════════════════════════════════════════

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


def _is_tradingview_window(title):
    """TradingView 앱 창인지 판별"""
    t = title.lower()
    # TradingView 앱 타이틀 패턴: 'ETHUSDT.P ▲ 2,326.88 +0.47%' 형태
    # 숫자 포함 심볼(예: 1000PEPEUSDT)도 인식하도록 [A-Z0-9] 허용
    if re.search(r'[A-Z0-9]{2,12}USDT', title, re.IGNORECASE):
        return True
    if 'tradingview' in t:
        return True
    if 'binance' in t and ('chart' in t or 'perpetual' in t):
        return True
    return False


def parse_window_title(title):
    """TradingView 창 타이틀에서 종목과 타임프레임 추출"""
    symbol = None
    interval_label = None

    # 종목 추출: XXXUSDT 패턴 (1000PEPE 등 숫자 포함 심볼도 허용)
    sym_match = re.search(r'([A-Z0-9]{2,12}USDT)(?:\.P)?', title.upper())
    if sym_match:
        symbol = sym_match.group(1)

    # 타임프레임 추출: 다양한 구분자로 분리 (·, —, -, |, ,)
    parts = re.split(r'[·\—\-\|,]', title)
    for part in parts:
        clean = part.strip()
        if clean in INTERVAL_PARSE_MAP:
            interval_label = INTERVAL_PARSE_MAP[clean]
            break

    # 폴백: 패턴 매칭 (15m, 4h, 1D 등)
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


# ═══════════════════════════════════════════════
# Windows 구현
# ═══════════════════════════════════════════════

def _win_find_tradingview():
    """[Windows] TradingView 데스크탑 앱 창을 찾아 핸들 반환"""
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

    # 터미널/에디터 창 제외
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


def _win_bring_to_front(hwnd):
    """[Windows] 창을 최전면으로 (SetForegroundWindow 제한 우회)"""
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.3)

        import ctypes
        ctypes.windll.user32.keybd_event(0x12, 0, 0, 0)   # Alt down
        ctypes.windll.user32.keybd_event(0x12, 0, 2, 0)   # Alt up
        time.sleep(0.05)

        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.4)
        return True
    except Exception:
        time.sleep(0.2)
        return False


def _win_capture():
    """[Windows] TradingView 창 캡쳐"""
    result = _win_find_tradingview()
    if result is None or result[0] is None:
        return None, "TradingView 앱을 찾을 수 없습니다. TradingView 데스크탑 앱이 실행 중인지 확인하세요."

    hwnd, title = result

    try:
        _win_bring_to_front(hwnd)

        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y

        if w < 100 or h < 100:
            return None, "TradingView 창이 너무 작습니다."

        from PIL import ImageGrab
        img = ImageGrab.grab(bbox=(x, y, x2, y2), all_screens=True)

        return img, title

    except Exception as e:
        return None, f"캡쳐 실패: {str(e)}"


def _win_detect_chart_info():
    """[Windows] TradingView 창에서 종목/타임프레임 정보만 가져오기"""
    result = _win_find_tradingview()
    if result is None or result[0] is None:
        return None, None, None

    hwnd, title = result
    symbol, interval = parse_window_title(title)
    return symbol, interval, title


# ═══════════════════════════════════════════════
# Linux 구현
# ═══════════════════════════════════════════════

def _linux_find_tradingview():
    """[Linux] xdotool로 TradingView 창 찾기 → (window_id, title)"""
    try:
        # 모든 창 목록 가져오기
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", ""],
            capture_output=True, text=True, timeout=5
        )
        window_ids = result.stdout.strip().split('\n')
        window_ids = [wid for wid in window_ids if wid.strip()]
    except FileNotFoundError:
        return None, None, "xdotool이 설치되어 있지 않습니다. 'sudo apt install xdotool'로 설치하세요."
    except Exception as e:
        return None, None, f"창 검색 실패: {e}"

    candidates = []
    exclude = ['antigravity', 'code', 'vscode', 'terminal', 'extension:']

    for wid in window_ids:
        try:
            name_result = subprocess.run(
                ["xdotool", "getwindowname", wid],
                capture_output=True, text=True, timeout=2
            )
            title = name_result.stdout.strip()
            if title and _is_tradingview_window(title):
                if not any(ex in title.lower() for ex in exclude):
                    candidates.append((wid, title))
        except Exception:
            continue

    if not candidates:
        return None, None, "TradingView 앱을 찾을 수 없습니다. TradingView 데스크탑 앱이 실행 중인지 확인하세요."

    # 가장 큰 창 선택
    best = None
    best_area = 0
    for wid, title in candidates:
        try:
            geo_result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", wid],
                capture_output=True, text=True, timeout=2
            )
            geo = {}
            for line in geo_result.stdout.strip().split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    geo[k] = int(v)
            w = geo.get('WIDTH', 0)
            h = geo.get('HEIGHT', 0)
            area = w * h
            if area > best_area and w > 200 and h > 200:
                best = (wid, title, geo)
                best_area = area
        except Exception:
            continue

    if best is None:
        return candidates[0][0], candidates[0][1], None

    return best[0], best[1], best[2] if len(best) > 2 else None


def _linux_bring_to_front(wid):
    """[Linux] xdotool로 창을 최전면으로"""
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", wid],
                       timeout=3, capture_output=True)
        time.sleep(0.4)
        return True
    except Exception:
        time.sleep(0.2)
        return False


def _linux_capture():
    """[Linux] TradingView 창 캡쳐"""
    wid, title, geo_or_err = _linux_find_tradingview()
    if wid is None:
        return None, geo_or_err or "TradingView 앱을 찾을 수 없습니다."

    try:
        # 창을 최전면으로
        _linux_bring_to_front(wid)

        # 창 위치/크기 가져오기 (최신)
        geo_result = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", wid],
            capture_output=True, text=True, timeout=2
        )
        geo = {}
        for line in geo_result.stdout.strip().split('\n'):
            if '=' in line:
                k, v = line.split('=', 1)
                geo[k] = int(v)

        x = geo.get('X', 0)
        y = geo.get('Y', 0)
        w = geo.get('WIDTH', 0)
        h = geo.get('HEIGHT', 0)

        if w < 100 or h < 100:
            return None, "TradingView 창이 너무 작습니다."

        # scrot으로 특정 영역 캡쳐 (가장 안정적)
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "tradeai_capture.png")
        try:
            # 방법 1: scrot -a (영역 지정)
            subprocess.run(
                ["scrot", "-a", f"{x},{y},{w},{h}", "-o", tmp_path],
                timeout=5, capture_output=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # 방법 2: import (ImageMagick) - 창 ID로 직접 캡쳐
                subprocess.run(
                    ["import", "-window", wid, tmp_path],
                    timeout=5, capture_output=True, check=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                # 방법 3: PIL ImageGrab (X11 환경)
                try:
                    from PIL import ImageGrab
                    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                    return img, title
                except Exception:
                    return None, "캡쳐 도구를 찾을 수 없습니다. 'sudo apt install scrot' 또는 'sudo apt install imagemagick'으로 설치하세요."

        if os.path.exists(tmp_path):
            img = Image.open(tmp_path)
            img.load()  # 파일 핸들 해제
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return img, title
        else:
            return None, "캡쳐 파일이 생성되지 않았습니다."

    except Exception as e:
        return None, f"캡쳐 실패: {str(e)}"


def _linux_detect_chart_info():
    """[Linux] TradingView 창에서 종목/타임프레임 정보만 가져오기"""
    wid, title, _ = _linux_find_tradingview()
    if wid is None:
        return None, None, None

    symbol, interval = parse_window_title(title)
    return symbol, interval, title


# ═══════════════════════════════════════════════
# 공통 API (OS 자동 분기)
# ═══════════════════════════════════════════════

def find_tradingview_window():
    """TradingView 데스크탑 앱 창 찾기 (크로스플랫폼)"""
    if IS_WINDOWS:
        return _win_find_tradingview()
    elif IS_LINUX:
        wid, title, _ = _linux_find_tradingview()
        return (wid, title) if wid else (None, None)
    else:
        return None, None


def detect_chart_info():
    """TradingView 창에서 현재 종목/타임프레임 정보만 가져오기 (캡쳐 없이)"""
    if IS_WINDOWS:
        return _win_detect_chart_info()
    elif IS_LINUX:
        return _linux_detect_chart_info()
    else:
        return None, None, None


def capture_tradingview():
    """TradingView 창을 자동으로 찾아 캡쳐 (크로스플랫폼)
    반환: (image, window_title) 또는 (None, error_msg)
    """
    if IS_WINDOWS:
        return _win_capture()
    elif IS_LINUX:
        return _linux_capture()
    else:
        return None, f"지원하지 않는 OS: {sys.platform}"
