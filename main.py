# %%writefile app.py
"""
SPX Market Breadth Dashboard - Fast + Robust (Fix blank/slow charts)

Key fix:
- Rangebreaks default to weekends only (FAST).
- Optional holiday breaks are capped to avoid huge lists that freeze Plotly.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta, date
import requests
from io import StringIO
from typing import Optional, Tuple, List

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="SPX Dashboard", layout="wide")

SHEET_ID = "1faOXwIk7uR51IIeAMrrRPdorRsO7iJ3PDPn-mk5vc24"
GID = "564353266"

URL_PRIMARY  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
URL_FALLBACK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}"

st.title("📈 SPX Dashboard")

# -------------------------
# Helpers
# -------------------------
def _looks_like_html(text: str) -> bool:
    t = (text or "").lstrip().lower()
    return (
        t.startswith("<!doctype html")
        or t.startswith("<html")
        or ("servicelogin" in t)
        or ("accounts.google.com" in t)
    )

def _clean_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"nan": "", "None": "", "null": ""})
    s = (
        s.str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.replace("−", "-", regex=False)   # unicode minus
         .str.replace("—", "-", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")

def _normalize_percent(s: pd.Series) -> pd.Series:
    x = _clean_numeric_series(s)
    mx = x.max(skipna=True)
    # If data looks like 0..1, convert to %
    if pd.notna(mx) and mx <= 1.5:
        x = x * 100.0
    return x

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    return None

def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    Robust date parsing:
    - Try normal datetime parsing first
    - If mostly NaT, try Excel serial date conversion
    - Normalize to midnight (no tz)
    """
    ss = s.astype(str).str.strip()

    dt = pd.to_datetime(ss, errors="coerce", infer_datetime_format=True)
    ok = int(dt.notna().sum())

    # Fallback: Excel serial dates
    if ok < max(3, int(len(ss) * 0.2)):
        num = pd.to_numeric(ss, errors="coerce")
        if num.notna().sum() >= max(3, int(len(ss) * 0.5)) and num.median(skipna=True) > 10000:
            dt = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")

    # Remove tz if any, normalize
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
    except Exception:
        pass

    return dt.dt.normalize()

def _build_standard_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # name-based mapping
    col_date  = _pick_col(df, ["Date", "Datetime", "Time", "timestamp"])
    col_open  = _pick_col(df, ["Open"])
    col_high  = _pick_col(df, ["High"])
    col_low   = _pick_col(df, ["Low"])
    col_close = _pick_col(df, ["Close"])

    col_ma20  = _pick_col(df, ["MA20"])
    col_ma50  = _pick_col(df, ["MA50"])
    col_ma200 = _pick_col(df, ["MA200"])

    col_above = _pick_col(df, ["Percentage_Above_Both", "PctAbove", "% Above Both", "AboveBoth"])
    col_below = _pick_col(df, ["Percentage_Below_Both", "PctBelow", "% Below Both", "BelowBoth"])

    # positional fallback (A..S style)
    if not all([col_date, col_open, col_high, col_low, col_close]):
        def try_positional(offset: int = 0) -> dict:
            if df.shape[1] < 19 + offset:
                return {}
            cols = df.columns.tolist()
            return {
                "Date": cols[0 + offset],
                "Open": cols[1 + offset],
                "High": cols[2 + offset],
                "Low":  cols[3 + offset],
                "Close":cols[4 + offset],
                "MA20": cols[11 + offset],
                "MA50": cols[12 + offset],
                "MA200":cols[14 + offset],
                "PctAbove": cols[15 + offset],
                "PctBelow": cols[18 + offset],
            }

        mapping = try_positional(0) or try_positional(1)
        if not mapping:
            raise ValueError(f"Cannot map columns. Found {df.shape[1]} columns: {df.columns.tolist()}")

        out = pd.DataFrame({
            "Date": df[mapping["Date"]],
            "Open": df[mapping["Open"]],
            "High": df[mapping["High"]],
            "Low":  df[mapping["Low"]],
            "Close":df[mapping["Close"]],
            "MA20": df[mapping["MA20"]],
            "MA50": df[mapping["MA50"]],
            "MA200":df[mapping["MA200"]],
            "PctAbove": df[mapping["PctAbove"]],
            "PctBelow": df[mapping["PctBelow"]],
        })
    else:
        out = pd.DataFrame({
            "Date": df[col_date],
            "Open": df[col_open],
            "High": df[col_high],
            "Low":  df[col_low],
            "Close":df[col_close],
            "MA20": df[col_ma20] if col_ma20 else pd.NA,
            "MA50": df[col_ma50] if col_ma50 else pd.NA,
            "MA200":df[col_ma200] if col_ma200 else pd.NA,
            "PctAbove": df[col_above] if col_above else pd.NA,
            "PctBelow": df[col_below] if col_below else pd.NA,
        })

    out["Date"] = _parse_date_series(out["Date"])
    out = out.dropna(subset=["Date"]).sort_values("Date")

    for c in ["Open", "High", "Low", "Close", "MA20", "MA50", "MA200"]:
        out[c] = _clean_numeric_series(out[c])

    out["PctAbove"] = _normalize_percent(out["PctAbove"])
    out["PctBelow"] = _normalize_percent(out["PctBelow"])

    return out

def make_rangebreaks(dates: pd.Series, include_holidays: bool, max_holidays: int = 250):
    """
    FAST rangebreaks:
    - always skip weekends
    - optionally skip missing business days (holidays) BUT capped
    """
    rbs = [dict(bounds=["sat", "mon"])]

    if not include_holidays:
        return rbs

    dt = pd.to_datetime(dates, errors="coerce").dt.normalize()
    dt = dt.dropna()
    if dt.empty:
        return rbs

    obs = pd.DatetimeIndex(dt.unique())
    bdays = pd.date_range(obs.min(), obs.max(), freq="B")
    missing = bdays.difference(obs)

    # Cap to avoid Plotly freezing
    if len(missing) > 0 and len(missing) <= max_holidays:
        rbs.append(dict(values=list(missing)))

    return rbs

@st.cache_data(ttl=600, show_spinner=False)
def load_sheet_df(url_primary: str, url_fallback: str) -> Tuple[pd.DataFrame, str]:
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    last_err = None
    for url in (url_primary, url_fallback):
        try:
            r = session.get(url, headers=headers, timeout=20, allow_redirects=True)
            r.raise_for_status()
            txt = r.text or ""
            if _looks_like_html(txt):
                last_err = f"HTML/login page returned from: {url}"
                continue
            df = pd.read_csv(StringIO(txt), dtype=str)
            return df, url
        except Exception as e:
            last_err = str(e)

    raise ValueError(f"Google Sheet did not return CSV. Last error: {last_err}")

# -------------------------
# Controls
# -------------------------
# c1, c2, c3, c4, c5 = st.columns([1.2, 1.4, 3.2, 1.8, 1.4])
c1, c2, c3 = st.columns([1.2, 1.4, 3.2])

with c1:
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with c2:
    months_to_show = st.selectbox("Default window", [3, 6, 12], index=1)

# Set default values
include_holidays = False
disable_breaks = False

# with c4:
#     include_holidays = st.checkbox("Skip holidays (slower)", value=False)

# with c5:
#     disable_breaks = st.checkbox("Disable rangebreaks (debug)", value=False)

# -------------------------
# Load + Parse
# -------------------------
try:
    raw_df, used_url = load_sheet_df(URL_PRIMARY, URL_FALLBACK)
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

try:
    df = _build_standard_df(raw_df)
except Exception as e:
    st.error(f"Parse/mapping failed: {e}")
    with st.expander("Debug: raw columns + preview"):
        st.write(raw_df.columns.tolist())
        st.write(raw_df.head(10))
    st.stop()

if df.empty:
    st.warning("No valid rows after parsing Date. Check your Date column format.")
    with st.expander("Debug: raw preview"):
        st.write(raw_df.head(20))
    st.stop()

# Default date range
max_date = df["Date"].max()
min_date = df["Date"].min()

default_start = (max_date - timedelta(days=30 * int(months_to_show))).date()
default_end = max_date.date()

with c3:
    start_d, end_d = st.date_input(
        "Date range",
        value=(default_start, default_end),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    if isinstance(start_d, date) and isinstance(end_d, date) and start_d > end_d:
        start_d, end_d = end_d, start_d

start_ts = pd.Timestamp(start_d)
end_ts = pd.Timestamp(end_d)

dff = df.loc[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()

st.markdown("---")
st.write(f"Showing: **{start_d} → {end_d}**")

if dff.empty:
    st.warning("No data in the selected date range.")
    st.stop()

needed_ohlc = dff[["Open", "High", "Low", "Close"]].notna().all(axis=1).sum()
if needed_ohlc == 0:
    st.error("OHLC columns are not numeric (all NaN after cleaning).")
    with st.expander("Debug: cleaned OHLC sample"):
        st.write(dff[["Date", "Open", "High", "Low", "Close"]].head(30))
    st.stop()

# Rangebreaks (safe + capped)
rangebreaks = []
if not disable_breaks:
    rangebreaks = make_rangebreaks(dff["Date"], include_holidays=include_holidays, max_holidays=250)

# -------------------------
# Charts
# -------------------------

# Wrap S&P 500 Index in expander
with st.expander("📈 S&P 500 Index", expanded=True):
    fig1 = go.Figure(
        go.Candlestick(
            x=dff["Date"],
            open=dff["Open"],
            high=dff["High"],
            low=dff["Low"],
            close=dff["Close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )
    if rangebreaks:
        fig1.update_xaxes(rangebreaks=rangebreaks)
    fig1.update_layout(
        height=450,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# Wrap Market Breadth Analysis in expander
with st.expander("📊 Market Breadth Analysis", expanded=True):
    tab1, tab2 = st.tabs(["📊 Moving Averages", "📈 Double Moving Averages"])

    with tab1:
        fig2 = go.Figure()
        if dff["MA20"].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff["Date"], 
                y=dff["MA20"], 
                name="MA20", 
                line=dict(width=1.5, color="#26a69a")
            ))
        if dff["MA50"].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff["Date"], 
                y=dff["MA50"], 
                name="MA50", 
                line=dict(width=2, color="#ff9800")
            ))
        if dff["MA200"].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff["Date"], 
                y=dff["MA200"], 
                name="MA200", 
                line=dict(width=2.5, color="#ef5350")
            ))

        if rangebreaks:
            fig2.update_xaxes(rangebreaks=rangebreaks)
        fig2.update_layout(
            height=400,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        fig3 = go.Figure()

        if dff["PctAbove"].notna().any():
            fig3.add_trace(
                go.Scatter(
                    x=dff["Date"],
                    y=dff["PctAbove"],
                    name="% Price > 50-SMA and 200-SMA",
                    fill="tozeroy",
                    line=dict(width=1.5, color="#00ff00"),  # Green line
                    fillcolor="rgba(0, 255, 0, 0.2)",  # Green fill
                )
            )

        if dff["PctBelow"].notna().any():
            fig3.add_trace(
                go.Scatter(
                    x=dff["Date"],
                    y=dff["PctBelow"],
                    name="% Price < 50-SMA and 200-SMA",
                    fill="tozeroy",
                    line=dict(width=1.5, color="#ff0000"),  # Red line
                    fillcolor="rgba(255, 0, 0, 0.2)",
                )
            )

        if rangebreaks:
            fig3.update_xaxes(rangebreaks=rangebreaks)
        fig3.update_layout(
            height=400,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Percent",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")
st.caption(f"📊 Dashboard | {len(dff):,} points | {dff['Date'].min().date()} to {dff['Date'].max().date()}")