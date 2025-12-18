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
st.set_page_config(page_title="S&P500 Market Analysis", layout="wide")

SHEET_ID = "1faOXwIk7uR51IIeAMrrRPdorRsO7iJ3PDPn-mk5vc24"
GID = "564353266"

URL_PRIMARY  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
URL_FALLBACK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}"

st.title("📈 S&P500 Market Analysis")

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
    
    # New Highs/Lows columns
    col_nh20 = _pick_col(df, ["NH20"])
    col_nh250 = _pick_col(df, ["NH250"])
    col_nl20 = _pick_col(df, ["NL20"])
    col_nl250 = _pick_col(df, ["NL250"])
    
    # McClellan columns
    col_mcclellan_osc = _pick_col(df, ["McClellan_Oscillator", "McClellanOscillator"])
    col_mcclellan_sum = _pick_col(df, ["McClellan_Summation_Index", "McClellanSummationIndex"])
    
    # RSI columns
    col_rsi_over_70 = _pick_col(df, ["RSI_over_70", "RSIover70"])
    col_rsi_below_30 = _pick_col(df, ["RSI_below_30", "RSIbelow30"])
    
    # Advances-Declines Line
    col_ad_line = _pick_col(df, ["Advances-Declines Line", "AdvancesDeclines", "AD_Line", "ADLine"])

    # positional fallback (A..S style)
    if not all([col_date, col_open, col_high, col_low, col_close]):
        def try_positional(offset: int = 0) -> dict:
            if df.shape[1] < 49 + offset:
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
                "NH20": cols[19 + offset],
                "NH250": cols[23 + offset],
                "NL20": cols[24 + offset],
                "NL250": cols[28 + offset],
                "McClellan_Oscillator": cols[9 + offset],
                "McClellan_Summation_Index": cols[10 + offset],
                "RSI_over_70": cols[47 + offset],
                "RSI_below_30": cols[48 + offset],
                "AD_Line": cols[8 + offset],  # Column I (index 8)
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
            "NH20": df[mapping["NH20"]],
            "NH250": df[mapping["NH250"]],
            "NL20": df[mapping["NL20"]],
            "NL250": df[mapping["NL250"]],
            "McClellan_Oscillator": df[mapping["McClellan_Oscillator"]],
            "McClellan_Summation_Index": df[mapping["McClellan_Summation_Index"]],
            "RSI_over_70": df[mapping["RSI_over_70"]],
            "RSI_below_30": df[mapping["RSI_below_30"]],
            "AD_Line": df[mapping["AD_Line"]],
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
            "NH20": df[col_nh20] if col_nh20 else pd.NA,
            "NH250": df[col_nh250] if col_nh250 else pd.NA,
            "NL20": df[col_nl20] if col_nl20 else pd.NA,
            "NL250": df[col_nl250] if col_nl250 else pd.NA,
            "McClellan_Oscillator": df[col_mcclellan_osc] if col_mcclellan_osc else pd.NA,
            "McClellan_Summation_Index": df[col_mcclellan_sum] if col_mcclellan_sum else pd.NA,
            "RSI_over_70": df[col_rsi_over_70] if col_rsi_over_70 else pd.NA,
            "RSI_below_30": df[col_rsi_below_30] if col_rsi_below_30 else pd.NA,
            "AD_Line": df[col_ad_line] if col_ad_line else pd.NA,
        })

    out["Date"] = _parse_date_series(out["Date"])
    out = out.dropna(subset=["Date"]).sort_values("Date")

    for c in ["Open", "High", "Low", "Close", "MA20", "MA50", "MA200", "NH20", "NH250", "NL20", "NL250", "McClellan_Oscillator", "McClellan_Summation_Index", "RSI_over_70", "RSI_below_30", "AD_Line"]:
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

@st.cache_data(ttl=600, show_spinner=False)
def load_rrg_df(sheet_id: str, gid: str) -> pd.DataFrame:
    """Load RRG data from Google Sheets"""
    url_primary = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    url_fallback = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    
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
            df = pd.read_csv(StringIO(txt))
            return df
        except Exception as e:
            last_err = str(e)
    
    raise ValueError(f"RRG data load failed. Last error: {last_err}")

# -------------------------
# Controls
# -------------------------
c1, c2, c3 = st.columns([1.2, 1.4, 3.2])

with c1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with c2:
    months_to_show = st.selectbox("Time Period (month)", [3, 6, 12], index=1)

# Set default values
include_holidays = True
disable_breaks = False

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
        yaxis_title="S&P500 Index",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

with st.expander("📊 Market Breadth Analysis", expanded=True):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Moving Averages", 
        "📈 Double Moving Averages", 
        "📊 New Highs & Lows", 
        "📈 Advances-Declines Line",
        "📈 McClellan", 
        "📊 Relative Strength Index"
    ])

    with tab1:
        # Moving Averages chart (stays the same)
        fig2 = go.Figure()
        if dff["MA20"].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff["Date"], 
                y=dff["MA20"], 
                name="Percentage of Members Above 20 Day Moving Average",
                line=dict(width=1.5, color="#26a69a")
            ))
        if dff["MA50"].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff["Date"], 
                y=dff["MA50"], 
                name="Percentage of Members Above 50 Day Moving Average",
                line=dict(width=2, color="#ff9800")
            ))
        if dff["MA200"].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff["Date"], 
                y=dff["MA200"], 
                name="Percentage of Members Above 200 Day Moving Average",
                line=dict(width=2.5, color="#ef5350")
            ))

        if rangebreaks:
            fig2.update_xaxes(rangebreaks=rangebreaks)
        fig2.update_layout(
            height=400,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Moving Averages",
            yaxis=dict(range=[0, 100]),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        # Double Moving Averages chart (stays the same)
        fig3 = go.Figure()

        if dff["PctAbove"].notna().any():
            fig3.add_trace(
                go.Scatter(
                    x=dff["Date"],
                    y=dff["PctAbove"],
                    name="Percentage of Members Above 50-DMA and 200-DMA",
                    fill="tozeroy",
                    line=dict(width=1.5, color="#00ff00"),
                    fillcolor="rgba(0, 255, 0, 0.2)",
                )
            )

        if dff["PctBelow"].notna().any():
            fig3.add_trace(
                go.Scatter(
                    x=dff["Date"],
                    y=dff["PctBelow"],
                    name="Percentage of Members Below 50-DMA and 200-DMA",
                    fill="tozeroy",
                    line=dict(width=1.5, color="#ff0000"),
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
            yaxis_title="Double Moving Averages",
            yaxis=dict(range=[0, 100]),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        # New Highs & Lows charts (stays the same)
        # Chart 1: NH20 and NL20
        fig4a = go.Figure()

        if dff["NH20"].notna().any():
            fig4a.add_trace(
                go.Bar(
                    x=dff["Date"],
                    y=dff["NH20"],
                    name="Percentage of Members with New 4 Week Highs",
                    marker_color="lightgreen",
                )
            )

        if dff["NL20"].notna().any():
            fig4a.add_trace(
                go.Bar(
                    x=dff["Date"],
                    y=dff["NL20"] * -1,
                    name="Percentage of Members with New 4 Week Lows",
                    marker_color="salmon",
                )
            )

        if rangebreaks:
            fig4a.update_xaxes(rangebreaks=rangebreaks)
        fig4a.update_layout(
            height=350,
            template="plotly_dark",
            hovermode="x unified",
            barmode='relative',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="New Highs & New Lows (4 Week)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig4a, use_container_width=True, config={"displayModeBar": False})

        # Chart 2: NH250 and NL250
        fig4b = go.Figure()
        
        if dff["NH250"].notna().any():
            fig4b.add_trace(
                go.Bar(
                    x=dff["Date"],
                    y=dff["NH250"],
                    name="Percentage of Members with New 52 Week Highs",
                    marker_color="#4FD555",
                )
            )

        if dff["NL250"].notna().any():
            fig4b.add_trace(
                go.Bar(
                    x=dff["Date"],
                    y=dff["NL250"] * -1,
                    name="Percentage of Members with New 52 Week Lows",
                    marker_color="#E54141",
                )
            )

        if rangebreaks:
            fig4b.update_xaxes(rangebreaks=rangebreaks)
        fig4b.update_layout(
            height=350,
            template="plotly_dark",
            hovermode="x unified",
            barmode='relative',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="New Highs & New Lows (52 Week)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig4b, use_container_width=True, config={"displayModeBar": False})

    with tab4:
        # Advances-Declines Line chart (moved from tab6 to tab4)
        fig7 = go.Figure()

        if dff["AD_Line"].notna().any():
            fig7.add_trace(
                go.Scatter(
                    x=dff["Date"],
                    y=dff["AD_Line"],
                    name="Advances-Declines Line",
                    line=dict(width=2, color="#0000ff"),
                    mode='lines'
                )
            )

        if rangebreaks:
            fig7.update_xaxes(rangebreaks=rangebreaks)
        fig7.update_layout(
            height=400,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Advances-Declines Line",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig7, use_container_width=True, config={"displayModeBar": False})

    with tab5:
        # McClellan charts (moved from tab4 to tab5)
        # Chart 1: McClellan Oscillator
        fig5a = go.Figure()

        if dff["McClellan_Oscillator"].notna().any():
            fig5a.add_trace(
                go.Scatter(
                    x=dff["Date"],
                    y=dff["McClellan_Oscillator"],
                    name="McClellan Oscillator",
                    line=dict(width=2, color="#00ff00"),
                    mode='lines'
                )
            )

        if rangebreaks:
            fig5a.update_xaxes(rangebreaks=rangebreaks)
        fig5a.update_layout(
            height=350,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="McClellan Oscillator",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig5a, use_container_width=True, config={"displayModeBar": False})

        # Chart 2: McClellan Summation Index
        fig5b = go.Figure()

        if dff["McClellan_Summation_Index"].notna().any():
            fig5b.add_trace(
                go.Scatter(
                    x=dff["Date"],
                    y=dff["McClellan_Summation_Index"],
                    name="McClellan Summation Index",
                    line=dict(width=2, color="#0000ff"),
                    mode='lines'
                )
            )

        if rangebreaks:
            fig5b.update_xaxes(rangebreaks=rangebreaks)
        fig5b.update_layout(
            height=350,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="McClellan Summation Index",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig5b, use_container_width=True, config={"displayModeBar": False})

    with tab6:
        # Relative Strength Index chart (moved from tab5 to tab6)
        fig6 = go.Figure()

        if dff["RSI_over_70"].notna().any():
            fig6.add_trace(
                go.Scatter(
                    x=dff["Date"],
                    y=dff["RSI_over_70"],
                    name="Percentage of Members with 14-Day RSI Above 70 ",
                    line=dict(width=2, color="#00ff00"),
                    mode='lines'
                )
            )

        if dff["RSI_below_30"].notna().any():
            fig6.add_trace(
                go.Scatter(
                    x=dff["Date"],
                    y=dff["RSI_below_30"],
                    name="Percentage of Members with 14-Day RSI Below 30",
                    line=dict(width=2, color="#ff0000"),
                    mode='lines'
                )
            )

        if rangebreaks:
            fig6.update_xaxes(rangebreaks=rangebreaks)
        fig6.update_layout(
            height=400,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Relative Strength Index",
            yaxis=dict(range=[0, 100]),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# -------------------------
# RRG Section
# -------------------------
RRG_GID = "1041193797"

with st.expander("📊 Relative Rotation Graph", expanded=True):
    try:
        rrg_df = load_rrg_df(SHEET_ID, RRG_GID)
        
        # Parse date column
        rrg_df['Date'] = pd.to_datetime(rrg_df['Date'], errors='coerce')
        rrg_df = rrg_df.dropna(subset=['Date']).sort_values('Date')
        
        if rrg_df.empty:
            st.warning("No valid RRG data available")
        else:
            # Get latest date data
            latest_date = rrg_df['Date'].max()
            latest_data = rrg_df[rrg_df['Date'] == latest_date].iloc[0]
            
            st.write(f"Latest RRG Data: **{latest_date.date()}**")
            
            # Define sectors and their corresponding columns
            sectors = {
                'Consumer Discretionary': ('Consumer Discretionary JdK RS-Momentum', 'Consumer Discretionary JdK RS-Ratio'),
                'Consumer Staples': ('Consumer Staples JdK RS-Momentum', 'Consumer Staples JdK RS-Ratio'),
                'Health Care': ('Health Care JdK RS-Momentum', 'Health Care JdK RS-Ratio'),
                'Industrials': ('Industrials JdK RS-Momentum', 'Industrials JdK RS-Ratio'),
                'Information Technology': ('Information Technology JdK RS-Momentum', 'Information Technology JdK RS-Ratio'),
                'Materials': ('Materials JdK RS-Momentum', 'Materials JdK RS-Ratio'),
                'Real Estate': ('Real Estate JdK RS-Momentum', 'Real Estate JdK RS-Ratio'),
                'Communication Services': ('Communication Services JdK RS-Momentum', 'Communication Services JdK RS-Ratio'),
                'Utilities': ('Utilities JdK RS-Momentum', 'Utilities JdK RS-Ratio'),
                'Financials': ('Financials JdK RS-Momentum', 'Financials JdK RS-Ratio'),
                'Energy': ('Energy JdK RS-Momentum', 'Energy JdK RS-Ratio'),
            }
            
            # Prepare data for plotting
            plot_data = []
            for sector_name, (momentum_col, ratio_col) in sectors.items():
                try:
                    momentum = pd.to_numeric(latest_data[momentum_col], errors='coerce')
                    ratio = pd.to_numeric(latest_data[ratio_col], errors='coerce')
                    
                    if pd.notna(momentum) and pd.notna(ratio):
                        plot_data.append({
                            'Sector': sector_name,
                            'RS_Momentum': momentum,
                            'RS_Ratio': ratio
                        })
                except Exception as e:
                    st.warning(f"Could not parse {sector_name}: {e}")
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Create RRG scatter plot
                fig_rrg = go.Figure()
                
                # Determine quadrant colors for markers
                colors = []
                for _, row in plot_df.iterrows():
                    if row['RS_Ratio'] >= 100 and row['RS_Momentum'] >= 100:
                        colors.append('#2ecc71')  # Leading - Green
                    elif row['RS_Ratio'] < 100 and row['RS_Momentum'] >= 100:
                        colors.append('#3498db')  # Improving - Blue
                    elif row['RS_Ratio'] < 100 and row['RS_Momentum'] < 100:
                        colors.append('#e74c3c')  # Lagging - Red
                    else:
                        colors.append('#f39c12')  # Weakening - Orange
                
                # Add scatter plot with colored markers
                fig_rrg.add_trace(
                    go.Scatter(
                        x=plot_df['RS_Ratio'],
                        y=plot_df['RS_Momentum'],
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color=colors,
                            line=dict(width=2, color='#34495e')
                        ),
                        text=plot_df['Sector'],
                        textposition='top center',
                        textfont=dict(size=9, color='#2c3e50'),
                        hovertemplate='<b>%{text}</b><br>' +
                                     'RS-Ratio: %{x:.2f}<br>' +
                                     'RS-Momentum: %{y:.2f}<br>' +
                                     '<extra></extra>',
                        name='Sectors'
                    )
                )
                
                # Add quadrant lines at 100
                fig_rrg.add_hline(y=100, line_dash="solid", line_color="#7f8c8d", line_width=2, opacity=0.8)
                fig_rrg.add_vline(x=100, line_dash="solid", line_color="#7f8c8d", line_width=2, opacity=0.8)
                
                # Fixed axis ranges
                x_min, x_max = 95, 105
                y_min, y_max = 95, 105
                
                # Add quadrant background colors (lighter shades)
                fig_rrg.add_shape(type="rect", x0=100, y0=100, x1=x_max, y1=y_max,
                                 fillcolor="#d5f4e6", opacity=0.3, layer="below", line_width=0)  # Light green
                fig_rrg.add_shape(type="rect", x0=x_min, y0=100, x1=100, y1=y_max,
                                 fillcolor="#d6eaf8", opacity=0.3, layer="below", line_width=0)  # Light blue
                fig_rrg.add_shape(type="rect", x0=x_min, y0=y_min, x1=100, y1=100,
                                 fillcolor="#fadbd8", opacity=0.3, layer="below", line_width=0)  # Light red
                fig_rrg.add_shape(type="rect", x0=100, y0=y_min, x1=x_max, y1=100,
                                 fillcolor="#fdeaa8", opacity=0.3, layer="below", line_width=0)  # Light orange/yellow
                
                # Add quadrant labels with darker text
                fig_rrg.add_annotation(x=104, y=104, text="Leading", showarrow=False,
                                      font=dict(size=18, color="#27ae60", family="Arial Black"))
                fig_rrg.add_annotation(x=96, y=104, text="Improving", showarrow=False,
                                      font=dict(size=18, color="#2980b9", family="Arial Black"))
                fig_rrg.add_annotation(x=96, y=96, text="Lagging", showarrow=False,
                                      font=dict(size=18, color="#c0392b", family="Arial Black"))
                fig_rrg.add_annotation(x=104, y=96, text="Weakening", showarrow=False,
                                      font=dict(size=18, color="#d68910", family="Arial Black"))
                
                fig_rrg.update_layout(
                    height=700,
                    template="plotly_white",  # Changed to white template
                    plot_bgcolor='#f8f9fa',  # Light gray background
                    paper_bgcolor='white',
                    xaxis_title="JdK RS-Ratio",
                    yaxis_title="JdK RS-Momentum",
                    xaxis=dict(
                        range=[x_min, x_max], 
                        zeroline=False,
                        gridcolor='#ecf0f1',
                        title_font=dict(color='#2c3e50', size=14)
                    ),
                    yaxis=dict(
                        range=[y_min, y_max], 
                        zeroline=False,
                        gridcolor='#ecf0f1',
                        title_font=dict(color='#2c3e50', size=14)
                    ),
                    hovermode="closest",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10),
                    font=dict(color='#2c3e50')
                )
                
                st.plotly_chart(fig_rrg, use_container_width=True, config={"displayModeBar": True})
                
                # Show data table
                with st.expander("📊 Sector Data Table"):
                    display_df = plot_df.copy()
                    display_df['RS_Ratio'] = display_df['RS_Ratio'].round(2)
                    display_df['RS_Momentum'] = display_df['RS_Momentum'].round(2)
                    
                    # Add quadrant column
                    def get_quadrant(row):
                        if row['RS_Ratio'] >= 100 and row['RS_Momentum'] >= 100:
                            return 'Leading'
                        elif row['RS_Ratio'] < 100 and row['RS_Momentum'] >= 100:
                            return 'Improving'
                        elif row['RS_Ratio'] < 100 and row['RS_Momentum'] < 100:
                            return 'Lagging'
                        else:
                            return 'Weakening'
                    
                    display_df['Quadrant'] = display_df.apply(get_quadrant, axis=1)
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No valid sector data to plot")
                
    except Exception as e:
        st.error(f"Failed to load RRG data: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")
st.caption(f"📊 Dashboard | {len(dff):,} points | {dff['Date'].min().date()} to {dff['Date'].max().date()}")