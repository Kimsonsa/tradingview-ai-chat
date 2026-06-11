"""
RSI 파동 분석 렌더링 — SVG 파동 맵 / 가격 레벨 & 진입 지도 / TF 상세 카드

분석 로직(core/rsi_wave.py)에서 분리한 표현 계층.
의존 방향: rsi_render → rsi_wave (단방향. 역방향 참조 금지 — 순환 임포트)
"""
from core.rsi_wave import (
    WAVE_TIMEFRAMES, TF_LABELS_SHORT,
    REGIME_LABELS, REGIME_COLORS, SIGNAL_LABELS,
    get_regime_rsi_params,
    build_level_map, build_entry_scenarios, assess_entry,
)


# ═══════════════════════════════════════════════
# SVG 시각화 생성
# ═══════════════════════════════════════════════

def _get_arrow_color(direction):
    """화살표 방향에 따른 색상: 상승=초록, 하락=빨강"""
    if direction == "up":
        return "#22C55E"
    else:
        return "#EF4444"


def _svg_arrow(cx, cy, direction, color, adx=None):
    """SVG 화살표 요소 생성 — 순수 위/아래 방향"""
    if adx is not None and adx >= 40:
        length = 32
        width = 3.5
        head_w = 8
    elif adx is not None and adx >= 25:
        length = 26
        width = 3
        head_w = 7
    else:
        length = 20
        width = 2.5
        head_w = 6

    half = length / 2

    if direction == "up":
        x1, y1 = cx, cy + half
        x2, y2 = cx, cy - half
        p1 = f"{cx},{cy - half}"
        p2 = f"{cx - head_w},{cy - half + 10}"
        p3 = f"{cx + head_w},{cy - half + 10}"
        line_y2 = cy - half + 10
    elif direction == "down":
        x1, y1 = cx, cy - half
        x2, y2 = cx, cy + half
        p1 = f"{cx},{cy + half}"
        p2 = f"{cx - head_w},{cy + half - 10}"
        p3 = f"{cx + head_w},{cy + half - 10}"
        line_y2 = cy + half - 10
    else:
        x1, y1 = cx, cy - half
        x2, y2 = cx, cy + half
        p1 = f"{cx},{cy + half}"
        p2 = f"{cx - head_w},{cy + half - 10}"
        p3 = f"{cx + head_w},{cy + half - 10}"
        line_y2 = cy + half - 10

    return f"""
        <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{cx:.1f}" y2="{line_y2:.1f}"
              stroke="{color}" stroke-width="{width}" stroke-linecap="round"
              filter="url(#glow)"/>
        <polygon points="{p1} {p2} {p3}" fill="{color}" filter="url(#glow)"/>
    """


def generate_wave_svg(results):
    """분석 결과를 SVG 파동 위치 맵으로 변환 (v2 — 레짐 표시 추가)

    Returns:
        str: HTML 문자열 (div + inline SVG)
    """
    # ── 레이아웃 상수 (PAD_B 확장: 레짐 라벨 공간) ──
    W, H = 720, 490
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 25, 45, 120
    PLOT_W = W - PAD_L - PAD_R
    PLOT_H = H - PAD_T - PAD_B

    def rsi_to_y(rsi):
        return PAD_T + PLOT_H - (rsi / 100 * PLOT_H)

    # X 위치 계산
    n = len(WAVE_TIMEFRAMES)
    x_margin = 50
    x_start = PAD_L + x_margin
    x_end = W - PAD_R - x_margin
    x_spacing = (x_end - x_start) / (n - 1) if n > 1 else 0
    x_positions = [x_start + i * x_spacing for i in range(n)]

    # Y 기준선
    y_100 = rsi_to_y(100)
    y_80 = rsi_to_y(80)
    y_50 = rsi_to_y(50)
    y_20 = rsi_to_y(20)
    y_0 = rsi_to_y(0)

    svg_parts = []

    # ── HTML 래퍼 시작 ──
    svg_parts.append(f"""<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>body{{margin:0;padding:0;background:transparent;font-family:'Inter',sans-serif;}}</style>
</head><body>
<div style="width:100%;max-width:720px;margin:0 auto;border-radius:16px;overflow:hidden;
            box-shadow:0 4px 24px rgba(0,0,0,0.25);">
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:auto;display:block;">
  <defs>
    <linearGradient id="bg_grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1a1a2e"/>
      <stop offset="100%" stop-color="#16213e"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- 배경 -->
  <rect width="{W}" height="{H}" fill="url(#bg_grad)" rx="16"/>

  <!-- 과매수 구간 배경 -->
  <rect x="{PAD_L}" y="{y_100:.0f}" width="{PLOT_W}" height="{y_80 - y_100:.0f}"
        fill="rgba(239,68,68,0.07)" rx="4"/>

  <!-- 과매도 구간 배경 -->
  <rect x="{PAD_L}" y="{y_20:.0f}" width="{PLOT_W}" height="{y_0 - y_20:.0f}"
        fill="rgba(34,197,94,0.07)" rx="4"/>
""")

    # ── 수평 기준선 ──
    grid_lines = [
        (100, y_100, "#444466", "4,6", "100"),
        (80,  y_80,  "#EF4444", "6,4", "80"),
        (50,  y_50,  "#555577", "4,6", "50"),
        (20,  y_20,  "#22C55E", "6,4", "20"),
        (0,   y_0,   "#444466", "4,6", "0"),
    ]
    for rsi_val, y, color, dash, label in grid_lines:
        opacity = "0.6" if rsi_val in (80, 20) else "0.3"
        svg_parts.append(
            f'  <line x1="{PAD_L}" y1="{y:.0f}" x2="{W - PAD_R}" y2="{y:.0f}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="{dash}" opacity="{opacity}"/>'
        )
        font_color = color if rsi_val in (80, 20) else "#888899"
        font_weight = "600" if rsi_val in (80, 20) else "400"
        svg_parts.append(
            f'  <text x="{PAD_L - 8}" y="{y + 4:.0f}" fill="{font_color}" '
            f'font-size="11" font-family="Inter,sans-serif" text-anchor="end" '
            f'font-weight="{font_weight}">{label}</text>'
        )

    # ── 과매수/과매도 라벨 ──
    svg_parts.append(
        f'  <text x="{W - PAD_R - 4}" y="{y_80 - 6:.0f}" fill="#EF4444" '
        f'font-size="10" font-family="Inter,sans-serif" text-anchor="end" opacity="0.7">과매수</text>'
    )
    svg_parts.append(
        f'  <text x="{W - PAD_R - 4}" y="{y_20 + 14:.0f}" fill="#22C55E" '
        f'font-size="10" font-family="Inter,sans-serif" text-anchor="end" opacity="0.7">과매도</text>'
    )

    # ── 타이틀 ──
    svg_parts.append(
        f'  <text x="{W / 2}" y="28" fill="#E0E0F0" font-size="15" '
        f'font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">'
        f'🌊 RSI 파동 위치 맵 v2</text>'
    )

    # ── 데이터 포인트 수집 ──
    points = []
    for i, tf in enumerate(WAVE_TIMEFRAMES):
        r = results.get(tf)
        if not r or r.get("error"):
            continue
        rsi = r["rsi"]
        x = x_positions[i]
        y = rsi_to_y(rsi)
        arrow = r.get("arrow_dir", "right")
        color = _get_arrow_color(arrow)
        adx = r.get("adx")
        points.append((x, y, rsi, tf, arrow, color, adx))

    # ── 연결선 (RSI 프로파일) ──
    if len(points) >= 2:
        polyline_pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in points)
        svg_parts.append(
            f'  <polyline points="{polyline_pts}" fill="none" '
            f'stroke="rgba(255,255,255,0.12)" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # ── 레짐별 RSI 목표 라인 (각 타임프레임별 작은 대시) ──
    for i, tf in enumerate(WAVE_TIMEFRAMES):
        r = results.get(tf)
        if not r or r.get("error"):
            continue
        regime = r.get("regime", "MIXED")
        params = r.get("regime_params") or get_regime_rsi_params(regime)
        target_rsi = params.get("expected_target", 50)
        bounce_cap = params.get("bounce_cap")

        x = x_positions[i]
        target_y = rsi_to_y(target_rsi)
        rc = REGIME_COLORS.get(regime, "#94A3B8")

        # 작은 수평 틱 (목표 RSI)
        tick_w = 12
        svg_parts.append(
            f'  <line x1="{x - tick_w:.1f}" y1="{target_y:.0f}" x2="{x + tick_w:.1f}" y2="{target_y:.0f}" '
            f'stroke="{rc}" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.5"/>'
        )

        # 반등 한계 표시 (하락 추세)
        if bounce_cap:
            cap_y = rsi_to_y(bounce_cap[1])
            svg_parts.append(
                f'  <line x1="{x - tick_w:.1f}" y1="{cap_y:.0f}" x2="{x + tick_w:.1f}" y2="{cap_y:.0f}" '
                f'stroke="#FCA5A5" stroke-width="1" stroke-dasharray="2,2" opacity="0.4"/>'
            )

    # ── 각 타임프레임 화살표 + 라벨 ──
    for x, y, rsi, tf, arrow, color, adx in points:
        svg_parts.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" opacity="0.2" '
            f'filter="url(#glow)"/>'
        )
        svg_parts.append(_svg_arrow(x, y, arrow, color, adx))
        label_y = y - 22 if rsi >= 50 else y + 28
        svg_parts.append(
            f'  <text x="{x:.1f}" y="{label_y:.0f}" fill="{color}" '
            f'font-size="12" font-family="Inter,sans-serif" text-anchor="middle" '
            f'font-weight="600">{rsi:.1f}</text>'
        )

    # ── X축 타임프레임 라벨 ──
    for i, tf in enumerate(WAVE_TIMEFRAMES):
        x = x_positions[i]
        short = TF_LABELS_SHORT.get(tf, tf)
        r = results.get(tf)

        # 아이콘 (사이클 상태)
        if r and not r.get("error"):
            rsi = r["rsi"]
            if rsi >= 80:
                icon = "🔴"
            elif rsi <= 20:
                icon = "🟢"
            elif rsi >= 65 or rsi <= 35:
                icon = "🟠"
            else:
                icon = "🟡"
        else:
            icon = "⚪"

        # TF 라벨
        svg_parts.append(
            f'  <text x="{x:.1f}" y="{H - PAD_B + 22:.0f}" fill="#BBBBCC" '
            f'font-size="12" font-family="Inter,sans-serif" text-anchor="middle" '
            f'font-weight="500">{short}</text>'
        )

        # 포지션 라벨
        if r and not r.get("error"):
            pos = r.get("position", "")
            conf = r.get("confidence", "")
            if pos == "롱":
                pos_color = "#22C55E"
                pos_label = f"▲{pos}:{conf}"
            elif pos == "숏":
                pos_color = "#EF4444"
                pos_label = f"▼{pos}:{conf}"
            else:
                pos_color = "#94A3B8"
                pos_label = f"●{pos}"
            svg_parts.append(
                f'  <text x="{x:.1f}" y="{H - PAD_B + 40:.0f}" fill="{pos_color}" '
                f'font-size="10" font-family="Inter,sans-serif" text-anchor="middle" '
                f'font-weight="600">{pos_label}</text>'
            )

            # 레짐 라벨 (NEW)
            regime = r.get("regime", "")
            if regime:
                rc = REGIME_COLORS.get(regime, "#94A3B8")
                rs = REGIME_LABELS.get(regime, regime)
                svg_parts.append(
                    f'  <text x="{x:.1f}" y="{H - PAD_B + 56:.0f}" fill="{rc}" '
                    f'font-size="9" font-family="Inter,sans-serif" text-anchor="middle" '
                    f'font-weight="500">{rs}</text>'
                )

            # 신호 유형 라벨 (NEW — 핵심 신호만 표시)
            signal = r.get("signal_type", "")
            if signal in ("STRONG_LONG_REVERSAL", "BEARISH_CONTINUATION", "COUNTER_TREND_SCALP", "SCALP_LONG_ONLY",
                          "BEARISH_EXPANSION", "BULLISH_EXPANSION"):
                sig_labels = {
                    "STRONG_LONG_REVERSAL": "강롱전환",
                    "BEARISH_CONTINUATION": "하락지속",
                    "COUNTER_TREND_SCALP": "역추세",
                    "SCALP_LONG_ONLY": "스캘핑만",
                    "BEARISH_EXPANSION": "💥하방확장",
                    "BULLISH_EXPANSION": "💥상방확장",
                }
                sig_colors = {
                    "STRONG_LONG_REVERSAL": "#22C55E",
                    "BEARISH_CONTINUATION": "#EF4444",
                    "COUNTER_TREND_SCALP": "#F59E0B",
                    "SCALP_LONG_ONLY": "#F59E0B",
                    "BEARISH_EXPANSION": "#FF0000",
                    "BULLISH_EXPANSION": "#00FF00",
                }
                sig_text = sig_labels.get(signal, "")
                sig_color = sig_colors.get(signal, "#94A3B8")
                svg_parts.append(
                    f'  <text x="{x:.1f}" y="{H - PAD_B + 70:.0f}" fill="{sig_color}" '
                    f'font-size="8" font-family="Inter,sans-serif" text-anchor="middle" '
                    f'font-weight="600" opacity="0.8">{sig_text}</text>'
                )

    # ── 범례 ──
    legend_y = H - 12
    legends = [
        ("#22C55E", "▲ 상승 중 (직전 과매도 후)"),
        ("#EF4444", "▼ 하락 중 (직전 과매수 후)"),
    ]
    legend_start = W / 2 - len(legends) * 80 / 2
    for j, (lc, lt) in enumerate(legends):
        lx = legend_start + j * 160
        svg_parts.append(
            f'  <circle cx="{lx:.0f}" cy="{legend_y - 3}" r="4" fill="{lc}"/>'
        )
        svg_parts.append(
            f'  <text x="{lx + 8:.0f}" y="{legend_y:.0f}" fill="#888899" '
            f'font-size="9" font-family="Inter,sans-serif">{lt}</text>'
        )

    # ── 닫기 ──
    svg_parts.append("</svg>\n</div>\n</body></html>")

    return "\n".join(svg_parts)


def generate_price_ladder_svg(results, position=None):
    """가격 레벨 & 진입 지도 — 고정 행 간격 사다리 (HTML 프래그먼트 반환).

    가격 비례 배치는 레벨이 현재가 근처에 밀집하면 라벨이 겹쳐 가독성이
    무너진다 → 레벨당 한 행(고정 34px)으로 표시하고, 비례감은 각 행의
    '현재가 대비 거리(%)'로 보완한다. 겹침이 구조적으로 발생하지 않는다.

    position: 사용자가 가장 마지막에 입력한 포지션 dict 하나만.
        (이전 입력·청산 안 누른 기록은 무시 — 마지막 입력만 유효한 값으로 취급)
        진입가 행(호박색)에 가격·수량·현재 PnL을 표시한다.
    """
    lm = build_level_map(results)
    if not lm:
        return ""
    es = build_entry_scenarios(results)
    ea = assess_entry(results)

    price = lm["ref_price"]
    above = lm["above"][:5]
    below = lm["below"][:5]
    if not (above or below):
        return ""

    direction = ea["direction"] if ea else "중립"

    # 진입가 → (R, grade), 목표가
    entry_info = {}
    target_price = None
    if es:
        for s in es["scenarios"]:
            entry_info[s["entry"]] = (s["R"], s["grade"])
        if es["scenarios"]:
            target_price = es["scenarios"][0]["target"]

    # 행 구성: 모든 행(레벨/현재가/내 포지션)을 가격 내림차순으로 정렬
    items = [(c["price"], "level", c) for c in above + below]
    items.append((price, "current", None))
    if position and position.get("entry") and position.get("direction") in ("롱", "숏"):
        items.append((float(position["entry"]), "position", position))
    items.sort(key=lambda x: -x[0])
    rows = [(kind, payload) for _, kind, payload in items]

    W = 720
    ROW_H = 34
    TOP = 54
    FOOT = 28
    H = TOP + len(rows) * ROW_H + FOOT
    LINE_X1, LINE_X2 = 178, 532

    grade_color = {"양호": "#22C55E", "보통": "#F59E0B", "부적합": "#EF4444", "산출불가": "#94A3B8"}
    p = []
    p.append(f'''<div style="width:100%;max-width:720px;margin:10px auto 0;border-radius:16px;
                overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.25);">
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block;">
  <defs>
    <linearGradient id="bg_grad2" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1a1a2e"/><stop offset="100%" stop-color="#16213e"/>
    </linearGradient>
    <filter id="glow2"><feGaussianBlur stdDeviation="1.6" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="{W}" height="{H}" fill="url(#bg_grad2)" rx="16"/>''')

    cur_idx = next(i for i, (k, _) in enumerate(rows) if k == "current")
    cur_y = TOP + cur_idx * ROW_H + ROW_H / 2

    # ── 방향별 존 음영 (수익 zone / 위험 zone) ──
    zone_top, zone_bot = TOP, H - FOOT
    if direction in ("숏", "롱"):
        up_col = "rgba(239,68,68,0.06)" if direction == "숏" else "rgba(34,197,94,0.06)"
        dn_col = "rgba(34,197,94,0.06)" if direction == "숏" else "rgba(239,68,68,0.06)"
        p.append(f'<rect x="{LINE_X1}" y="{zone_top}" width="{LINE_X2 - LINE_X1}" height="{cur_y - zone_top:.0f}" fill="{up_col}"/>')
        p.append(f'<rect x="{LINE_X1}" y="{cur_y:.0f}" width="{LINE_X2 - LINE_X1}" height="{zone_bot - cur_y:.0f}" fill="{dn_col}"/>')

    # ── 타이틀 + 방향 배지 ──
    p.append(f'<text x="20" y="30" fill="#E0E0F0" font-size="15" font-family="Inter,sans-serif" font-weight="600">📐 가격 레벨 &amp; 진입 지도</text>')
    if direction != "중립":
        dcol = "#EF4444" if direction == "숏" else "#22C55E"
        p.append(f'<rect x="{W-110}" y="14" width="92" height="22" rx="11" fill="{dcol}" opacity="0.18"/>')
        p.append(f'<text x="{W-64}" y="29" fill="{dcol}" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">{direction} 우위</text>')

    # ── 행 렌더링 (고정 간격 — 겹침 없음) ──
    for i, (kind, c) in enumerate(rows):
        ry = TOP + i * ROW_H + ROW_H / 2

        if kind == "current":
            # 현재가 행 — 흰 라인 + 파란 배지
            p.append(f'<line x1="{LINE_X1-6}" y1="{ry:.1f}" x2="{LINE_X2+6}" y2="{ry:.1f}" stroke="#FFFFFF" stroke-width="2" filter="url(#glow2)"/>')
            p.append(f'<rect x="{LINE_X1-110}" y="{ry-11:.1f}" width="100" height="22" rx="6" fill="#2962FF"/>')
            p.append(f'<text x="{LINE_X1-60:.0f}" y="{ry+4:.1f}" fill="#fff" font-size="11.5" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">현재 {price:,.1f}</text>')
            if price in entry_info:
                R, grade = entry_info[price]
                bc = grade_color.get(grade, "#94A3B8")
                rtxt = f"{R}R" if R is not None else "-"
                bx = (LINE_X1 + LINE_X2) / 2 - 50
                p.append(f'<rect x="{bx:.0f}" y="{ry-10:.1f}" width="100" height="18" rx="9" fill="#16213e" stroke="{bc}" stroke-width="1"/>')
                p.append(f'<text x="{bx+50:.0f}" y="{ry+3:.1f}" fill="{bc}" font-size="10.5" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">추격 {rtxt}·{grade}</text>')
            continue

        if kind == "position":
            # 내 포지션 행 (호박색) — 마지막 입력 1건: 진입가·수량·PnL
            entry = float(c["entry"])
            is_long = c.get("direction") == "롱"
            pcol = "#FFC53D"
            pnl = ((price - entry) if is_long else (entry - price)) / entry * 100
            dist = (entry - price) / price * 100
            p.append(f'<text x="{LINE_X1-10}" y="{ry+1:.1f}" fill="{pcol}" font-size="13" font-family="Inter,sans-serif" text-anchor="end" font-weight="700">{entry:,.1f}</text>')
            p.append(f'<text x="{LINE_X1-10}" y="{ry+13:.1f}" fill="#777799" font-size="9" font-family="Inter,sans-serif" text-anchor="end">{dist:+.2f}%</text>')
            p.append(f'<line x1="{LINE_X1}" y1="{ry:.1f}" x2="{LINE_X2}" y2="{ry:.1f}" stroke="{pcol}" stroke-width="1.8" stroke-dasharray="7,3" opacity="0.85"/>')
            qty_txt = f" · {c['qty']:,.6g}개" if c.get("qty") else ""
            pnl_col = "#22C55E" if pnl >= 0 else "#EF4444"
            badge = f"내 {c.get('direction')}{qty_txt} · PnL {pnl:+.2f}%"
            bw = max(120, 14 + len(badge) * 7)
            bx = (LINE_X1 + LINE_X2) / 2 - bw / 2
            p.append(f'<rect x="{bx:.0f}" y="{ry-10:.1f}" width="{bw}" height="19" rx="9" fill="#16213e" stroke="{pcol}" stroke-width="1.2"/>')
            p.append(f'<text x="{bx+bw/2:.0f}" y="{ry+3.5:.1f}" fill="{pcol}" font-size="10.5" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">내 {c.get("direction")}{qty_txt} · <tspan fill="{pnl_col}">PnL {pnl:+.2f}%</tspan></text>')
            p.append(f'<text x="{LINE_X2+8}" y="{ry+4:.1f}" fill="{pcol}" font-size="9.5" font-family="Inter,sans-serif" opacity="0.8">📍 내 포지션</text>')
            continue

        lp = c["price"]
        is_res = lp > price
        col = "#EF4444" if is_res else "#22C55E"
        conf = c["n"] >= 3
        dist = (lp - price) / price * 100

        # 좌측: 가격 + 컨플루언스 별 + 거리%
        star = "⭐" if conf else ""
        p.append(f'<text x="{LINE_X1-10}" y="{ry+1:.1f}" fill="{col}" font-size="13" font-family="Inter,sans-serif" text-anchor="end" font-weight="{"700" if conf else "500"}">{lp:,.1f}{star}</text>')
        p.append(f'<text x="{LINE_X1-10}" y="{ry+13:.1f}" fill="#777799" font-size="9" font-family="Inter,sans-serif" text-anchor="end">{dist:+.2f}%</text>')

        # 라인 (컨플루언스=실선 굵게 / 단일=점선)
        sw = "1.8" if conf else "1"
        dash = "1,0" if conf else "5,5"
        p.append(f'<line x1="{LINE_X1}" y1="{ry:.1f}" x2="{LINE_X2}" y2="{ry:.1f}" stroke="{col}" stroke-width="{sw}" stroke-dasharray="{dash}" opacity="0.6"/>')

        # 우측: 출처 태그
        tags = ", ".join(c["labels"][:3])
        p.append(f'<text x="{LINE_X2+8}" y="{ry+4:.1f}" fill="#8888AA" font-size="9.5" font-family="Inter,sans-serif">{tags}</text>')

        # 중앙 배지: 진입 R / 1차 목표 (배경 채워 라인 위에 떠 보이게)
        if lp in entry_info:
            R, grade = entry_info[lp]
            bc = grade_color.get(grade, "#94A3B8")
            rtxt = f"{R}R" if R is not None else "-"
            bx = (LINE_X1 + LINE_X2) / 2 - 44
            p.append(f'<rect x="{bx:.0f}" y="{ry-10:.1f}" width="88" height="19" rx="9" fill="#16213e" stroke="{bc}" stroke-width="1"/>')
            p.append(f'<text x="{bx+44:.0f}" y="{ry+3.5:.1f}" fill="{bc}" font-size="10.5" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">진입 {rtxt}·{grade}</text>')
        elif target_price is not None and abs(lp - target_price) < 0.01:
            bx = (LINE_X1 + LINE_X2) / 2 - 40
            p.append(f'<rect x="{bx:.0f}" y="{ry-10:.1f}" width="80" height="19" rx="9" fill="#16213e" stroke="#22C55E" stroke-width="1"/>')
            p.append(f'<text x="{bx+40:.0f}" y="{ry+3.5:.1f}" fill="#22C55E" font-size="10.5" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">🎯 1차 목표</text>')

    # ── 범례 ──
    p.append(f'<text x="20" y="{H-9}" fill="#777799" font-size="9" font-family="Inter,sans-serif">🔴저항  🟢지지·목표  ⭐컨플루언스(실선)  ·  좌측 %=현재가 대비 거리  ·  R=목표거리÷손절거리</text>')

    p.append("</svg></div>")
    return "\n".join(p)


# ═══════════════════════════════════════════════
# 텍스트 카드 생성
# ═══════════════════════════════════════════════

def generate_tf_cards(results):
    """타임프레임별 상세 카드 마크다운 생성 (v2 — 레짐/다이버전스/점수 표시)"""
    cards = []

    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if not r or r.get("error"):
            err = r.get("error", "데이터 없음") if r else "데이터 없음"
            cards.append(f"**⏱ {tf}** — ⚠️ {err}\n")
            continue

        # 기본 데이터
        price = r["price"]
        rsi = r["rsi"]
        adx = r.get("adx")
        adx_str = f"{adx}" if adx is not None else "N/A"
        regime = r.get("regime", "MIXED")
        regime_label = REGIME_LABELS.get(regime, regime)

        # EMA
        ema_str = f"20={r['ema20']:.1f} | 50={r['ema50']:.1f} | 200={r['ema200']:.1f}"

        # VWAP
        if r.get("vwap"):
            vwap_pos = "위" if price > r["vwap"] else "아래"
            vwap_str = f"{r['vwap']:.1f} ({vwap_pos})"
        else:
            vwap_str = "N/A"

        # 볼밴
        bb_str = "N/A"
        if r.get("bb_bw") is not None:
            bw = r["bb_bw"]
            if bw < 3:
                bb_status = "🔴스퀴즈"
            elif bw < 5:
                bb_status = "🟡수축"
            elif bw > 10:
                bb_status = "🟢확장"
            else:
                bb_status = ""
            bb_str = f"{bw}% {bb_status}"

        # MACD
        macd_dir = "🟢" if r["macd_hist"] > 0 else "🔴"

        # 다이버전스 상태 (v2)
        div_str = ""
        div_v2 = r.get("div_v2")
        div_status = r.get("div_status")
        if div_v2:
            status_label = {"CONFIRMED": "✅확정", "UNCONFIRMED": "⏳미확정"}.get(div_status, "")
            div_str = f"\n> {div_v2['label']} — {status_label}"
        failed_div = r.get("failed_div")
        if failed_div:
            div_str += f"\n> ⚠️ **다이버전스 실패** — {failed_div['detail']}"

        # RSI 회복 강도
        rsi_recovery = r.get("rsi_recovery") or {}
        recovery_str = ""
        if rsi_recovery.get("strength") and rsi_recovery["strength"] != "NEUTRAL":
            strength_labels = {
                "VERY_WEAK": "🔴매우약함", "WEAK": "🟠약함",
                "NORMAL": "🟡정상", "STRONG": "🟢강함"
            }
            recovery_str = f" | RSI회복: {strength_labels.get(rsi_recovery['strength'], rsi_recovery['strength'])}"

        # 베어 플래그
        bear_flag_str = ""
        if r.get("bear_flag"):
            bear_flag_str = f"\n> ⚠️ **베어 플래그** — {r['bear_flag']['detail']}"

        # 스퀴즈 확장
        squeeze_str = ""
        if r.get("squeeze_expansion"):
            sq = r["squeeze_expansion"]
            sq_icon = "🔴💥" if sq["type"] == "BEARISH_EXPANSION" else "🟢💥"
            squeeze_str = f"\n> {sq_icon} **스퀴즈 확장** — {sq['detail']}"

        # 거래량 패턴
        vol_pat = r.get("vol_pattern") or {}
        vol_pat_str = ""
        if vol_pat.get("pattern") == "ABSORPTION":
            vol_pat_str = f" | 📊거래량흡수"
        elif vol_pat.get("pattern") == "CONTINUATION":
            vol_pat_str = f" | 📊하락지속형"

        # CVD 추세 (시장가 매수/매도 우위)
        cvd_trend_str = ""
        if r.get("cvd") is not None and r.get("cvd_ema") is not None:
            cvd_trend_str = " | CVD " + ("매수우위↑" if r["cvd"] > r["cvd_ema"] else "매도우위↓")

        # 거래량/OBV/CVD 종합 다이버전스 (NEW)
        synth_div_str = ""
        synth_div = r.get("synth_div")
        if synth_div and synth_div.get("overall_bias") != "NEUTRAL":
            synth_div_str = f"\n> 📊 **종합 다이버전스**: {synth_div['summary']}"
        cvd_div = r.get("cvd_div")
        if cvd_div:
            synth_div_str += f"\n> {cvd_div['label']} — {cvd_div['detail']}"
        vol_div = r.get("vol_div")
        if vol_div and vol_div.get("bias") != "NEUTRAL":
            synth_div_str += f"\n> {vol_div['label']} — {vol_div['detail']}"
        obv_div = r.get("obv_div")
        if obv_div:
            synth_div_str += f"\n> {obv_div['label']} — {obv_div['detail']}"

        # OI 변화 + 펀딩 (NEW)
        oi_str = ""
        oi_an = r.get("oi_analysis")
        if oi_an:
            oi_str += f"\n> 🔗 **OI {oi_an['label']}** ({oi_an['oi_change_pct']:+.1f}%) — {oi_an['detail']}"
        fund = r.get("funding_analysis")
        if fund and fund.get("squeeze_risk"):
            oi_str += f"\n> 💸 **{fund['label']}** — {fund['detail']}"

        # 롱/숏 점수
        long_s = r.get("long_score", 0)
        short_s = r.get("short_score", 0)
        signal = r.get("signal_type", "")
        signal_label = SIGNAL_LABELS.get(signal, signal)

        # 목표가
        targets = r.get("targets") or {}
        target_str = ""
        long_tgts = targets.get("long", [])
        short_tgts = targets.get("short", [])
        rsi_tgt = targets.get("rsi_target", "")
        if long_tgts:
            target_str += " | 롱목표: " + ", ".join(f"{t[0]}={t[1]}({t[2]})" for t in long_tgts[:3])
        if short_tgts:
            target_str += " | 숏목표: " + ", ".join(f"{t[0]}={t[1]}({t[2]})" for t in short_tgts[:2])

        # HTF 필터
        htf_str = ""
        htf = r.get("htf_filter", "")
        if htf == "HTF_BEARISH":
            htf_str = " | ⚠️상위프레임 하락 → 롱 감점"
        elif htf == "HTF_BULLISH":
            htf_str = " | ✅상위프레임 상승 → 롱 가점"

        card = f"""**⏱ {tf}** — {r['cycle_pos']} | 레짐: {regime_label}
| 항목 | 값 |
|------|-----|
| 현재가 | {price:,.1f} USDT |
| RSI(14) | **{rsi:.1f}** (이전: {r['prev_rsi']:.1f}){recovery_str} |
| EMA | {ema_str} → {r['ema_trend']} |
| ADX | {adx_str} ({r['market_type']}) |
| VWAP | {vwap_str} |
| 볼밴폭 | {bb_str} |
| MACD Hist | {r['macd_hist']:.2f} {macd_dir} |
| 거래량 | 5봉평균 대비 {r['vol_ratio']}%{vol_pat_str}{cvd_trend_str} |

📍 **{r['cycle_desc']}** — {r['rsi_strategy_valid']}
🎯 **{signal_label}** (롱:{long_s} / 숏:{short_s}){htf_str}
📐 {rsi_tgt}{target_str}{div_str}{bear_flag_str}{squeeze_str}{synth_div_str}{oi_str}

---
"""
        cards.append(card)

    return "\n".join(cards)
