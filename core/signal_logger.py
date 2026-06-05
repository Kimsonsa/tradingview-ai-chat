"""
RSI 파동 신호 로깅 + 결과 평가 모듈 (3단계)

목적: 모델이 내놓은 신호(점수/포지션/레짐/다이버전스/CVD/OI/펀딩)를 진입가와 함께
기록하고, TF별 호라이즌이 지나면 실제 가격 결과로 적중 여부를 평가한다.
→ 손으로 정한 가중치를 '근거 있게' 조정하기 위한 데이터 토대.

저장: Supabase(PostgreSQL) 사용 가능하면 우선, 아니면 로컬 JSON 폴백 (단일 스토어).
"""
import json
import os
import uuid
from datetime import datetime, timedelta

from core.market_data import fetch_klines, INTERVAL_MAP

# 로컬 폴백 경로
SIGNALS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals")
SIGNAL_FILE = os.path.join(SIGNALS_DIR, "rsi_signals.json")

# TF별 1봉 길이(분) — 디둡 윈도우
TF_MINUTES = {
    "1분": 1, "5분": 5, "15분": 15, "1시간": 60,
    "4시간": 240, "1일": 1440, "1주": 10080,
}

# TF별 평가 호라이즌(분) — 신호 발생 후 결과를 판정하는 시간 (대략 6~24봉)
EVAL_HORIZON_MIN = {
    "1분": 30, "5분": 120, "15분": 360, "1시간": 1440,
    "4시간": 4320, "1일": 10080, "1주": 43200,
}

# 방향성 실현수익 판정 데드존(%)
_WIN_THRESHOLD = 0.1


# ═══════════════════════════════════════════════
# DB 연결 (streamlit secrets — 없으면 로컬 폴백)
# ═══════════════════════════════════════════════

def _get_conn():
    try:
        import streamlit as st
        import psycopg2
        cfg = st.secrets["supabase"]
        return psycopg2.connect(
            host=cfg["host"], port=cfg["port"], dbname=cfg["dbname"],
            user=cfg["user"], password=cfg["password"], connect_timeout=5,
        )
    except Exception:
        return None


_DB_READY = False


def _init_table():
    conn = _get_conn()
    if conn is None:
        return False
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS rsi_wave_signals (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                symbol TEXT,
                timeframe TEXT,
                price DOUBLE PRECISION,
                position TEXT,
                confidence TEXT,
                signal_type TEXT,
                regime TEXT,
                rsi DOUBLE PRECISION,
                long_score INT,
                short_score INT,
                cvd_bias TEXT,
                oi_quadrant TEXT,
                funding_pct DOUBLE PRECISION,
                divergences TEXT,
                horizon_min INT,
                evaluated BOOLEAN DEFAULT FALSE,
                evaluated_at TEXT,
                exit_price DOUBLE PRECISION,
                mfe_pct DOUBLE PRECISION,
                mae_pct DOUBLE PRECISION,
                return_pct DOUBLE PRECISION,
                outcome TEXT
            )
        """)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


def _ensure_db():
    global _DB_READY
    if not _DB_READY:
        _DB_READY = _init_table()
    return _DB_READY


# 컬럼 순서 (DB insert/select 공용)
_COLS = [
    "id", "created_at", "symbol", "timeframe", "price", "position", "confidence",
    "signal_type", "regime", "rsi", "long_score", "short_score", "cvd_bias",
    "oi_quadrant", "funding_pct", "divergences", "horizon_min", "evaluated",
    "evaluated_at", "exit_price", "mfe_pct", "mae_pct", "return_pct", "outcome",
]


# ═══════════════════════════════════════════════
# 로컬 폴백 I/O
# ═══════════════════════════════════════════════

def _local_load():
    if not os.path.exists(SIGNAL_FILE):
        return []
    try:
        with open(SIGNAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _local_save(rows):
    os.makedirs(SIGNALS_DIR, exist_ok=True)
    try:
        with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ═══════════════════════════════════════════════
# 행 빌드 + 디둡
# ═══════════════════════════════════════════════

def _active_divs(r):
    """TF 결과에서 활성 다이버전스/패턴 타입을 문자열로"""
    out = []
    dv = r.get("div_v2")
    if dv:
        out.append(dv.get("type", ""))
    if r.get("cvd_div"):
        out.append(r["cvd_div"].get("type", ""))
    if r.get("obv_div"):
        out.append(r["obv_div"].get("type", ""))
    if r.get("failed_div"):
        out.append("FAILED_DIV")
    sq = r.get("squeeze_expansion")
    if sq:
        out.append(sq.get("type", ""))
    return ",".join(x for x in out if x)


def _build_rows(symbol, results):
    """analyze_rsi_wave 결과 → 신호 행 리스트 (TF당 1행)"""
    now_iso = datetime.now().isoformat()
    rows = []
    for tf, r in results.items():
        if not r or r.get("error"):
            continue

        cvd_bias = None
        if r.get("cvd") is not None and r.get("cvd_ema") is not None:
            cvd_bias = "BUY" if r["cvd"] > r["cvd_ema"] else "SELL"

        oi_an = r.get("oi_analysis") or {}
        fund = r.get("funding_analysis") or {}

        rows.append({
            "id": f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{symbol}_{tf}_{uuid.uuid4().hex[:6]}",
            "created_at": now_iso,
            "symbol": symbol,
            "timeframe": tf,
            "price": float(r.get("price", 0)),
            "position": r.get("position", ""),
            "confidence": r.get("confidence", ""),
            "signal_type": r.get("signal_type", ""),
            "regime": r.get("regime", ""),
            "rsi": float(r.get("rsi", 0)),
            "long_score": int(r.get("long_score", 0)),
            "short_score": int(r.get("short_score", 0)),
            "cvd_bias": cvd_bias,
            "oi_quadrant": oi_an.get("quadrant"),
            "funding_pct": fund.get("funding_pct"),
            "divergences": _active_divs(r),
            "horizon_min": EVAL_HORIZON_MIN.get(tf, 1440),
            "evaluated": False,
            "evaluated_at": None,
            "exit_price": None,
            "mfe_pct": None,
            "mae_pct": None,
            "return_pct": None,
            "outcome": None,
        })
    return rows


def _latest_per_tf(symbol):
    """심볼의 TF별 최근 기록 시각 {tf: datetime} — 디둡용"""
    out = {}
    if _ensure_db():
        conn = _get_conn()
        if conn:
            try:
                c = conn.cursor()
                c.execute(
                    "SELECT timeframe, MAX(created_at) FROM rsi_wave_signals "
                    "WHERE symbol = %s GROUP BY timeframe", (symbol,)
                )
                for tf, last in c.fetchall():
                    try:
                        out[tf] = datetime.fromisoformat(last)
                    except Exception:
                        pass
                return out
            except Exception:
                pass
            finally:
                conn.close()
    # 로컬
    for row in _local_load():
        if row.get("symbol") != symbol:
            continue
        tf = row.get("timeframe")
        try:
            ts = datetime.fromisoformat(row["created_at"])
        except Exception:
            continue
        if tf not in out or ts > out[tf]:
            out[tf] = ts
    return out


def _dedup(symbol, rows):
    """같은 TF를 1봉 길이 안에 중복 기록하지 않도록 필터"""
    latest = _latest_per_tf(symbol)
    now = datetime.now()
    kept = []
    for row in rows:
        tf = row["timeframe"]
        last = latest.get(tf)
        window = TF_MINUTES.get(tf, 60)
        if last and (now - last) < timedelta(minutes=window):
            continue  # 너무 최근 → 스킵
        kept.append(row)
    return kept


# ═══════════════════════════════════════════════
# 공용 API — 로깅
# ═══════════════════════════════════════════════

def log_rsi_wave_signals(symbol, results):
    """RSI 파동 분석 결과를 신호로 저장. 반환: 저장된 행 수"""
    rows = _build_rows(symbol, results)
    rows = _dedup(symbol, rows)
    if not rows:
        return 0

    if _ensure_db():
        conn = _get_conn()
        if conn:
            try:
                c = conn.cursor()
                placeholders = ",".join(["%s"] * len(_COLS))
                col_str = ",".join(_COLS)
                c.executemany(
                    f"INSERT INTO rsi_wave_signals ({col_str}) VALUES ({placeholders}) "
                    f"ON CONFLICT (id) DO NOTHING",
                    [tuple(row[k] for k in _COLS) for row in rows],
                )
                conn.commit()
                return len(rows)
            except Exception:
                conn.rollback()
            finally:
                conn.close()

    # 로컬 폴백
    existing = _local_load()
    existing.extend(rows)
    _local_save(existing)
    return len(rows)


# ═══════════════════════════════════════════════
# 공용 API — 평가 (결과 백필)
# ═══════════════════════════════════════════════

def _evaluate_window(position, entry, window_candles):
    """구간 캔들로 방향성 MFE/MAE/실현수익 계산

    Returns: (mfe_pct, mae_pct, return_pct, outcome) — 평가 불가 시 None
    """
    if not window_candles or entry <= 0:
        return None

    highs = [c["high"] for c in window_candles]
    lows = [c["low"] for c in window_candles]
    end_close = window_candles[-1]["close"]
    hi = max(highs)
    lo = min(lows)

    if position == "롱":
        mfe = (hi - entry) / entry * 100
        mae = (lo - entry) / entry * 100
        ret = (end_close - entry) / entry * 100
    elif position == "숏":
        mfe = (entry - lo) / entry * 100
        mae = (entry - hi) / entry * 100
        ret = (entry - end_close) / entry * 100
    else:  # 중립 — 방향 없음
        ret = (end_close - entry) / entry * 100
        return (None, None, round(ret, 3), "SKIP")

    if ret > _WIN_THRESHOLD:
        outcome = "WIN"
    elif ret < -_WIN_THRESHOLD:
        outcome = "LOSS"
    else:
        outcome = "NEUTRAL"

    return (round(mfe, 3), round(mae, 3), round(ret, 3), outcome)


def _pending_signals(symbol=None):
    """평가 대기(미평가) 신호 로드"""
    if _ensure_db():
        conn = _get_conn()
        if conn:
            try:
                c = conn.cursor()
                q = ("SELECT " + ",".join(_COLS) +
                     " FROM rsi_wave_signals WHERE evaluated = FALSE")
                params = ()
                if symbol:
                    q += " AND symbol = %s"
                    params = (symbol,)
                c.execute(q, params)
                return [dict(zip(_COLS, row)) for row in c.fetchall()], "db"
            except Exception:
                pass
            finally:
                conn.close()
    rows = [r for r in _local_load() if not r.get("evaluated")]
    if symbol:
        rows = [r for r in rows if r.get("symbol") == symbol]
    return rows, "local"


def _write_eval(updates, mode):
    """평가 결과 일괄 반영. updates: [(id, fields_dict), ...]"""
    if not updates:
        return
    if mode == "db":
        conn = _get_conn()
        if conn:
            try:
                c = conn.cursor()
                for sid, f in updates:
                    c.execute(
                        "UPDATE rsi_wave_signals SET evaluated=TRUE, evaluated_at=%s, "
                        "exit_price=%s, mfe_pct=%s, mae_pct=%s, return_pct=%s, outcome=%s "
                        "WHERE id=%s",
                        (f["evaluated_at"], f["exit_price"], f["mfe_pct"], f["mae_pct"],
                         f["return_pct"], f["outcome"], sid),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
            finally:
                conn.close()
        return
    # 로컬
    by_id = {sid: f for sid, f in updates}
    rows = _local_load()
    for row in rows:
        if row["id"] in by_id:
            row.update(by_id[row["id"]])
            row["evaluated"] = True
    _local_save(rows)


def evaluate_pending_signals(symbol=None, max_groups=14):
    """호라이즌이 지난 미평가 신호를 실제 가격으로 평가. 반환: 평가된 행 수"""
    pending, mode = _pending_signals(symbol)
    now = datetime.now()

    # 성숙한 것만 (created_at + horizon <= now)
    mature = []
    for row in pending:
        try:
            created = datetime.fromisoformat(row["created_at"])
        except Exception:
            continue
        horizon = int(row.get("horizon_min") or 1440)
        if created + timedelta(minutes=horizon) <= now:
            mature.append((created, horizon, row))

    if not mature:
        return 0

    # (symbol, tf)별 그룹 → klines 1회로 그룹 전체 평가
    groups = {}
    for created, horizon, row in mature:
        groups.setdefault((row["symbol"], row["timeframe"]), []).append((created, horizon, row))

    updates = []
    for (sym, tf), items in list(groups.items())[:max_groups]:
        bi = INTERVAL_MAP.get(tf)
        tf_min = TF_MINUTES.get(tf, 60)
        if not bi:
            continue
        # 가장 오래된 신호의 구간 끝까지 덮을 캔들 수
        oldest = min(c for c, _, _ in items)
        span_min = (now - oldest).total_seconds() / 60
        limit = min(1500, int(span_min / tf_min) + 10)
        try:
            candles = fetch_klines(sym, bi, max(limit, 30))
        except Exception:
            continue

        for created, horizon, row in items:
            start_ms = created.timestamp() * 1000
            end_ms = (created + timedelta(minutes=horizon)).timestamp() * 1000
            window = [c for c in candles if start_ms <= c["time"] <= end_ms]
            res = _evaluate_window(row.get("position", ""), float(row.get("price") or 0), window)
            if res is None:
                # 구간 캔들 없음(너무 오래됨) → 평가 불가 처리
                fields = {
                    "evaluated_at": now.isoformat(), "exit_price": None,
                    "mfe_pct": None, "mae_pct": None, "return_pct": None, "outcome": "SKIP",
                }
            else:
                mfe, mae, ret, outcome = res
                fields = {
                    "evaluated_at": now.isoformat(),
                    "exit_price": window[-1]["close"] if window else None,
                    "mfe_pct": mfe, "mae_pct": mae, "return_pct": ret, "outcome": outcome,
                }
            updates.append((row["id"], fields))

    _write_eval(updates, mode)
    return len(updates)


# ═══════════════════════════════════════════════
# 공용 API — 통계
# ═══════════════════════════════════════════════

def _all_evaluated(symbol=None):
    if _ensure_db():
        conn = _get_conn()
        if conn:
            try:
                c = conn.cursor()
                q = ("SELECT " + ",".join(_COLS) +
                     " FROM rsi_wave_signals WHERE evaluated = TRUE")
                params = ()
                if symbol:
                    q += " AND symbol = %s"
                    params = (symbol,)
                c.execute(q, params)
                return [dict(zip(_COLS, row)) for row in c.fetchall()]
            except Exception:
                pass
            finally:
                conn.close()
    rows = [r for r in _local_load() if r.get("evaluated")]
    if symbol:
        rows = [r for r in rows if r.get("symbol") == symbol]
    return rows


def _agg(rows):
    """행 묶음 → 집계 지표"""
    directional = [r for r in rows if r.get("outcome") in ("WIN", "LOSS", "NEUTRAL")]
    wins = sum(1 for r in directional if r["outcome"] == "WIN")
    losses = sum(1 for r in directional if r["outcome"] == "LOSS")
    decided = wins + losses
    rets = [r["return_pct"] for r in directional if r.get("return_pct") is not None]
    mfes = [r["mfe_pct"] for r in directional if r.get("mfe_pct") is not None]
    maes = [r["mae_pct"] for r in directional if r.get("mae_pct") is not None]
    return {
        "n": len(directional),
        "win_rate": round(wins / decided * 100, 1) if decided else None,
        "avg_return": round(sum(rets) / len(rets), 3) if rets else None,
        "avg_mfe": round(sum(mfes) / len(mfes), 3) if mfes else None,
        "avg_mae": round(sum(maes) / len(maes), 3) if maes else None,
    }


def _group_agg(rows, key):
    groups = {}
    for r in rows:
        k = r.get(key) or "(없음)"
        groups.setdefault(k, []).append(r)
    return {k: _agg(v) for k, v in sorted(groups.items())}


def get_signal_stats(symbol=None):
    """평가된 신호 통계 — 전체 + signal_type/confidence/regime/timeframe별

    Returns:
        dict: {total_evaluated, overall, by_signal_type, by_confidence, by_regime, by_timeframe}
    """
    rows = _all_evaluated(symbol)
    return {
        "total_evaluated": len(rows),
        "overall": _agg(rows),
        "by_signal_type": _group_agg(rows, "signal_type"),
        "by_confidence": _group_agg(rows, "confidence"),
        "by_regime": _group_agg(rows, "regime"),
        "by_timeframe": _group_agg(rows, "timeframe"),
    }


# ═══════════════════════════════════════════════
# 공용 API — 가중치 조정 제안 (반자동, 사람 승인)
# ═══════════════════════════════════════════════

# 그룹별 최소 표본 — 이보다 적으면 노이즈라 제안 안 함
SUGGEST_MIN_SAMPLES = 20


def get_weight_suggestions(symbol=None, min_samples=SUGGEST_MIN_SAMPLES):
    """평가 통계 기반 가중치 조정 '제안' 생성 (자동 적용 X — 사람이 판단).

    충분한 표본(min_samples 이상)이 모인 signal_type/regime만 대상.
    적중률이 낮으면 하향, 높으면 상향 여지를 제안.

    Returns:
        dict: {
            ready: bool,          # 전체 표본이 최소치 도달했는지
            total: int,           # 평가 완료 신호 수
            needed: int,          # 제안 시작까지 남은 표본 수
            min_samples: int,
            suggestions: [ {target, direction, win_rate, avg_return, n, severity, message}, ... ]
        }
    """
    rows = _all_evaluated(symbol)
    total = len(rows)
    suggestions = []

    for key_label, key_col in (("신호유형", "signal_type"), ("레짐", "regime")):
        for name, v in _group_agg(rows, key_col).items():
            n = v.get("n") or 0
            wr = v.get("win_rate")
            ar = v.get("avg_return")
            if n < min_samples or wr is None:
                continue

            # 적중률 + 기대수익으로 방향 판단
            if wr < 40 or (ar is not None and ar < -0.5):
                severity = "high" if (wr < 30 or (ar is not None and ar < -1.0)) else "medium"
                suggestions.append({
                    "target": f"{key_label}: {name}",
                    "direction": "DOWN",
                    "win_rate": wr, "avg_return": ar, "n": n,
                    "severity": severity,
                    "message": f"적중률 {wr}%·평균 {ar}% (n={n}) → 가중치 하향 검토",
                })
            elif wr >= 60 and (ar is None or ar > 0):
                suggestions.append({
                    "target": f"{key_label}: {name}",
                    "direction": "UP",
                    "win_rate": wr, "avg_return": ar, "n": n,
                    "severity": "info",
                    "message": f"적중률 {wr}%·평균 {ar}% (n={n}) → 가중치 상향 여지",
                })

    sev_order = {"high": 0, "medium": 1, "info": 2}
    suggestions.sort(key=lambda s: (sev_order.get(s["severity"], 3), -s["n"]))

    return {
        "ready": total >= min_samples,
        "total": total,
        "needed": max(0, min_samples - total),
        "min_samples": min_samples,
        "suggestions": suggestions,
    }
