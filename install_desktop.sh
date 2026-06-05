#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════
# TradeAI 바탕화면 아이콘 설치 스크립트 (리눅스 전용)
#
# 사용법 (실제 리눅스 PC에서, 저장소 폴더 안에서 한 번만 실행):
#   bash install_desktop.sh
#
# 하는 일:
#   1) 실행용 런처(TradeAI.sh) 생성 — Streamlit 앱을 띄우고 브라우저를 엶
#   2) 바탕화면 + 앱 메뉴에 'TradeAI Assistant' 아이콘 등록
#   3) 더블클릭으로 바로 실행 가능하게 신뢰(trusted) 표시
# ════════════════════════════════════════════════════════════════
set -e

# ── 저장소 위치(이 스크립트가 있는 폴더)를 자동 인식 ──
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ICON_PATH="$REPO_DIR/pwa/icon-512.png"
LAUNCHER="$REPO_DIR/TradeAI.sh"

echo "▸ 저장소 위치: $REPO_DIR"

# ── 파이썬 / streamlit 확인 ──
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ python3 가 설치되어 있지 않습니다. 먼저 설치하세요: sudo apt install python3 python3-pip"
    exit 1
fi
if ! python3 -c "import streamlit" >/dev/null 2>&1 && [ ! -d "$REPO_DIR/venv" ]; then
    echo "⚠️  streamlit 미설치 — 의존성을 먼저 설치하세요:"
    echo "      cd \"$REPO_DIR\" && pip3 install -r requirements.txt"
    echo "   (지금 계속 진행해도 되지만, 아이콘 클릭 시 동작하려면 위 설치가 필요합니다.)"
fi

# ── 1) 런처 스크립트 생성 ──
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# TradeAI 실행 런처 (install_desktop.sh 가 자동 생성)
DIR="$REPO_DIR"
cd "\$DIR" || exit 1

# venv 가 있으면 활성화
[ -d "\$DIR/venv" ] && source "\$DIR/venv/bin/activate"

# 이미 8502 포트에서 떠 있지 않으면 Streamlit 백그라운드 기동(터미널 닫혀도 유지)
if ! curl -s -o /dev/null http://localhost:8502 2>/dev/null; then
    setsid streamlit run assistant.py --server.port 8502 --server.headless true \\
        > "\$DIR/.tradeai_run.log" 2>&1 &
fi

# 서버가 응답할 때까지 최대 30초 대기
for i in \$(seq 1 30); do
    curl -s -o /dev/null http://localhost:8502 2>/dev/null && break
    sleep 1
done

# 기본 브라우저로 열기
xdg-open http://localhost:8502 >/dev/null 2>&1 || true
EOF
chmod +x "$LAUNCHER"
echo "✓ 런처 생성: $LAUNCHER"

# ── 2) .desktop 항목 생성 ──
DESKTOP_ENTRY="[Desktop Entry]
Version=1.0
Type=Application
Name=TradeAI Assistant
Name[ko]=TradeAI 어시스턴트
Comment=TradingView AI 차트 분석 어시스턴트
Exec=/bin/bash \"$LAUNCHER\"
Icon=$ICON_PATH
Terminal=false
Categories=Office;Finance;
StartupNotify=true"

APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"
echo "$DESKTOP_ENTRY" > "$APPS_DIR/tradeai.desktop"
chmod +x "$APPS_DIR/tradeai.desktop"
echo "✓ 앱 메뉴 등록: $APPS_DIR/tradeai.desktop"

# ── 3) 바탕화면에 복사 ──
# 한국어/영어 바탕화면 폴더 모두 탐색
DESK_DIR=""
for d in "$HOME/Desktop" "$HOME/바탕화면" "$(xdg-user-dir DESKTOP 2>/dev/null)"; do
    if [ -n "$d" ] && [ -d "$d" ]; then DESK_DIR="$d"; break; fi
done

if [ -n "$DESK_DIR" ]; then
    cp "$APPS_DIR/tradeai.desktop" "$DESK_DIR/TradeAI.desktop"
    chmod +x "$DESK_DIR/TradeAI.desktop"
    # GNOME: 더블클릭 허용을 위한 신뢰 표시
    gio set "$DESK_DIR/TradeAI.desktop" metadata::trusted true 2>/dev/null || true
    echo "✓ 바탕화면 아이콘 생성: $DESK_DIR/TradeAI.desktop"
else
    echo "⚠️  바탕화면 폴더를 찾지 못했습니다. 앱 메뉴(런처)에서 'TradeAI'를 검색해 실행하세요."
fi

# 앱 메뉴 DB 갱신
command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$APPS_DIR" 2>/dev/null || true

echo ""
echo "════════════════════════════════════════════"
echo "✅ 설치 완료!"
echo "   바탕화면의 'TradeAI' 아이콘을 더블클릭하면 실행됩니다."
echo "   (첫 클릭 시 '실행 허용'을 물으면 허용을 선택하세요.)"
echo "════════════════════════════════════════════"
