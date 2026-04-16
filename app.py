"""
app.py — High-Convexity "Doubler" Stock Screener v2.0
Run:  streamlit run app.py
"""

import datetime as dt
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from engine import run_full_scan, make_sparkline_data, compute_exit_signals, run_backtest

st.set_page_config(page_title="Doubler Screener v2 — S&P 1500", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
:root{--bg-primary:#0a0e17;--bg-card:#111827;--border:#1e2d3d;--text-primary:#e2e8f0;--text-secondary:#94a3b8;--accent-green:#22c55e;--accent-red:#ef4444;--accent-blue:#3b82f6;--accent-gold:#f59e0b;--accent-purple:#a855f7}
.stApp{background-color:var(--bg-primary)!important;color:var(--text-primary)!important;font-family:'DM Sans',sans-serif!important}
h1,h2,h3,h4{font-family:'DM Sans',sans-serif!important;color:var(--text-primary)!important}
section[data-testid="stSidebar"]{background-color:#0d1117!important;border-right:1px solid var(--border)!important}
div[data-testid="stMetric"]{background-color:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:16px 20px}
div[data-testid="stMetric"] label{color:var(--text-secondary)!important;font-size:.8rem!important;text-transform:uppercase;letter-spacing:.05em}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:var(--accent-gold)!important;font-family:'JetBrains Mono',monospace!important;font-weight:700}
div[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden}
.stButton>button{background:linear-gradient(135deg,#3b82f6,#a855f7)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;padding:.6rem 1.5rem!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 4px 20px rgba(59,130,246,.4)!important}
.stDownloadButton>button{background:linear-gradient(135deg,#22c55e,#16a34a)!important;color:white!important;border:none!important;border-radius:10px!important}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{background-color:var(--bg-card)!important;color:var(--text-secondary)!important;border-radius:8px!important;border:1px solid var(--border)!important;padding:8px 20px!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#3b82f6,#a855f7)!important;color:white!important;border-color:transparent!important}
#MainMenu{visibility:hidden}header{visibility:hidden}footer{visibility:hidden}
.score-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.95rem}
.score-high{background:rgba(34,197,94,.15);color:#22c55e}.score-mid{background:rgba(245,158,11,.15);color:#f59e0b}.score-low{background:rgba(239,68,68,.15);color:#ef4444}
.regime-badge{display:inline-block;padding:6px 16px;border-radius:20px;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.85rem}
.regime-bull{background:rgba(34,197,94,.2);color:#22c55e;border:1px solid #22c55e}.regime-caution{background:rgba(245,158,11,.2);color:#f59e0b;border:1px solid #f59e0b}.regime-bear{background:rgba(239,68,68,.2);color:#ef4444;border:1px solid #ef4444}
.ensemble-tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.7rem;font-weight:600;background:rgba(168,85,247,.2);color:#a855f7;border:1px solid #a855f7;margin-left:6px}
</style>""", unsafe_allow_html=True)

st.markdown("""<div style="text-align:center;padding:1.5rem 0 .5rem;"><h1 style="margin:0;font-size:2.4rem;background:linear-gradient(135deg,#3b82f6,#a855f7,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🚀 HIGH-CONVEXITY DOUBLER SCREENER v2</h1><p style="color:#94a3b8;font-size:1.05rem;margin-top:6px;">S&amp;P 1500 · Regime-Aware · Ensemble-Validated · Backtested</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🔄 Scan Market Now", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    top_n = st.slider("Leaderboard Size", 10, 50, 20, 5)
    min_score = st.slider("Minimum Composite Score", 0.0, 10.0, 5.0, 0.5)
    sector_cap = st.slider("Max Stocks per Sector", 1, 10, 4, 1)
    st.markdown("---")
    st.markdown("### 📊 V2 Methodology")
    st.markdown("""| Factor | Weight |\n|--------|--------|\n| Trend Intensity | 30% |\n| Relative Strength | 25% |\n| Volume Persistence | 20% |\n| Volatility Quality | 15% |\n| Catalyst Proximity | 10% |\n\n**Hard Filters:** Price < 200 SMA → ❌ · Price > 2× 200 SMA → ❌\n\n**Bonuses:** SMA Stack → +0.5 · Ensemble → ✅\n\n**Regime:** Bear → scores × 0.5–0.65""")
    st.markdown("---")
    st.caption(f"Last scan cached for 24h · {dt.datetime.now():%Y-%m-%d %H:%M}")

@st.cache_data(ttl=86400, show_spinner=False)
def cached_scan():
    progress_bar = st.progress(0)
    status_text = st.empty()
    def _progress(pct, msg=""):
        progress_bar.progress(min(pct, 1.0))
        if msg:
            status_text.markdown(f"<p style='color:#94a3b8;text-align:center;'>{msg}</p>", unsafe_allow_html=True)
    scored, price_data, regime_info = run_full_scan(progress_callback=_progress)
    progress_bar.empty()
    status_text.empty()
    # Only cache price data for scored stocks to save memory
    scored_tickers = set()
    if not scored.empty:
        if "ticker" in scored.columns:
            scored_tickers = set(scored["ticker"].tolist())
        scored_tickers.add("SPY")
    filtered_price = {tk: df.to_dict() for tk, df in price_data.items() if tk in scored_tickers or len(scored_tickers) == 0}
    return scored, filtered_price, regime_info

with st.spinner("Running full S&P 1500 scan…"):
    scored_df, price_data_dict, regime_info = cached_scan()
price_data = {tk: pd.DataFrame(d) for tk, d in price_data_dict.items()}
if scored_df.empty:
    st.error("No data returned. Check your internet connection and try again.")
    st.stop()

regime = regime_info.get("regime", "UNKNOWN")
regime_label = {"BULL": ("🟢 BULL MARKET", "regime-bull"), "BULL_VOLATILE": ("🟡 BULL — HIGH VOLATILITY", "regime-caution"), "BEAR_CALM": ("🔴 BEAR MARKET — CALM", "regime-bear"), "BEAR_VOLATILE": ("🔴 BEAR MARKET — VOLATILE", "regime-bear")}.get(regime, ("⚪ UNKNOWN", "regime-caution"))
mult = regime_info.get("score_multiplier", 1.0)
vol = regime_info.get("realized_vol", 0)
vol_pct = regime_info.get("vol_percentile", 0)
st.markdown(f"""<div style="text-align:center;margin-bottom:16px;"><span class="regime-badge {regime_label[1]}">{regime_label[0]}</span><span style="color:#94a3b8;font-size:.85rem;margin-left:16px;">Score Multiplier: <b style="color:#f59e0b;">{mult:.2f}×</b> &nbsp;|&nbsp; Realized Vol: <b>{vol:.1f}%</b> &nbsp;|&nbsp; Vol Percentile: <b>{vol_pct:.0%}</b></span></div>""", unsafe_allow_html=True)

qualified = scored_df[scored_df["composite_score"] >= min_score].copy()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Stocks Scanned", f"{len(scored_df):,}")
col2.metric("Above 200 SMA", f"{scored_df['above_200sma'].sum():,}")
col3.metric("SMA Stack Aligned", f"{scored_df['sma_stack_aligned'].sum():,}")
ensemble_count = scored_df["ensemble_consensus"].sum() if "ensemble_consensus" in scored_df.columns else 0
col4.metric("Ensemble Consensus", f"{ensemble_count:,}")
col5.metric("Qualifying Doublers", f"{len(qualified):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Leaderboard", "📈 Deep Dive", "🛡️ Risk & Exit Signals", "🔬 Backtest", "📋 Full Results"])

with tab1:
    if qualified.empty:
        st.warning("No stocks meet the minimum score threshold. Lower the filter in the sidebar.")
    else:
        leaders = qualified.head(top_n).copy()
        if "ticker" not in leaders.columns:
            leaders["ticker"] = leaders.index
        if "ticker" not in leaders.columns:
            leaders = leaders.reset_index()
            if "Rank" in leaders.columns:
                leaders = leaders.rename(columns={"Rank": "rank_orig"})
        for i, row in leaders.iterrows():
            tk = row.get("ticker", "N/A")
            score = row["composite_score"]
            price = row.get("price", 0)
            rs_avg = row.get("rs_avg_decile", 0)
            trend_d = row.get("trend_decile", 0)
            quality_d = row.get("quality_decile", 0)
            sector = row.get("sector", "")
            is_ensemble = row.get("ensemble_consensus", False)
            has_earnings = row.get("earnings_within_30d", False)
            badge_class = "score-high" if score >= 8 else ("score-mid" if score >= 6 else "score-low")
            ensemble_tag = '<span class="ensemble-tag">ENSEMBLE ✓</span>' if is_ensemble else ""
            earnings_tag = '<span style="color:#f59e0b;font-size:.7rem;margin-left:4px;">📅 EARNINGS</span>' if has_earnings else ""
            with st.container():
                c1, c2, c3, c4, c5, c6 = st.columns([1.2, 2.2, 1.5, 1.5, 1.5, 3])
                with c1:
                    st.markdown(f'<div style="padding-top:8px;"><span class="score-badge {badge_class}">{score:.1f}</span></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div style="padding-top:4px;"><span style="font-family:'JetBrains Mono',monospace;font-weight:700;font-size:1.1rem;color:#e2e8f0;">{tk}</span>{ensemble_tag}{earnings_tag}<br><span style="color:#94a3b8;font-size:.8rem;">${price:,.2f} · {sector}</span></div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div style="padding-top:4px;"><span style="color:#94a3b8;font-size:.7rem;">TREND</span><br><span style="color:#22c55e;font-weight:600;">{trend_d:.0f}</span><span style="color:#94a3b8;font-size:.7rem;"> / 10</span></div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""<div style="padding-top:4px;"><span style="color:#94a3b8;font-size:.7rem;">REL STR</span><br><span style="color:#3b82f6;font-weight:600;">{rs_avg:.1f}</span><span style="color:#94a3b8;font-size:.7rem;"> / 10</span></div>""", unsafe_allow_html=True)
                with c5:
                    st.markdown(f"""<div style="padding-top:4px;"><span style="color:#94a3b8;font-size:.7rem;">QUALITY</span><br><span style="color:#a855f7;font-weight:600;">{quality_d:.1f}</span><span style="color:#94a3b8;font-size:.7rem;"> / 10</span></div>""", unsafe_allow_html=True)
                with c6:
                    spark = make_sparkline_data(price_data, tk, days=60)
                    if spark:
                        fig_spark = go.Figure(go.Scatter(y=spark, mode="lines", line=dict(color="#3b82f6", width=2), fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
                        fig_spark.update_layout(height=55, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
                        st.plotly_chart(fig_spark, width="stretch", config={"displayModeBar": False})
                st.markdown("<hr style='border-color:#1e2d3d;margin:4px 0;'>", unsafe_allow_html=True)

with tab2:
    if qualified.empty:
        st.info("No qualifying stocks to analyze.")
    else:
        ticker_list = qualified["ticker"].tolist() if "ticker" in qualified.columns else qualified.index.tolist()
        if isinstance(ticker_list, pd.Index):
            ticker_list = ticker_list.tolist()
        selected = st.selectbox("Select a stock for deep analysis", ticker_list, key="deep_dive_select")
        if selected and selected in price_data:
            df = price_data[selected]
            close = df["Close"]
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444"))
            fig.add_trace(go.Scatter(x=df.index, y=close.rolling(50).mean(), name="50 SMA", line=dict(color="#3b82f6", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=close.rolling(150).mean(), name="150 SMA", line=dict(color="#a855f7", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=close.rolling(200).mean(), name="200 SMA", line=dict(color="#f59e0b", width=1.5, dash="dot")))
            fig.update_layout(title=dict(text=f"{selected} — Candlestick + SMA Overlays", font=dict(size=18, color="#e2e8f0")), height=550, paper_bgcolor="#0a0e17", plot_bgcolor="#111827", font=dict(color="#94a3b8", family="DM Sans"), xaxis=dict(rangeslider_visible=False, gridcolor="#1e2d3d"), yaxis=dict(gridcolor="#1e2d3d", title="Price ($)"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), xaxis_rangeselector=dict(buttons=[dict(count=1,label="1M",step="month",stepmode="backward"),dict(count=3,label="3M",step="month",stepmode="backward"),dict(count=6,label="6M",step="month",stepmode="backward"),dict(step="all",label="ALL")], font=dict(color="#e2e8f0")))
            st.plotly_chart(fig, width="stretch")
            row = None
            if "ticker" in qualified.columns:
                match = qualified[qualified["ticker"] == selected]
                if not match.empty: row = match.iloc[0]
            else:
                if selected in qualified.index: row = qualified.loc[selected]
            if row is not None:
                st.markdown("#### Score Breakdown")
                b1,b2,b3,b4,b5 = st.columns(5)
                b1.metric("Composite", f"{row['composite_score']:.1f}")
                b2.metric("Trend", f"{row['trend_decile']:.0f} / 10")
                b3.metric("RS Avg", f"{row['rs_avg_decile']:.1f} / 10")
                b4.metric("Volume", f"{row['vol_decile']:.0f} / 10")
                quality = row.get("quality_decile", "N/A")
                b5.metric("Quality", f"{quality}" if isinstance(quality, str) else f"{quality:.1f} / 10")
                st.markdown("#### Technical Stats")
                t1,t2,t3,t4 = st.columns(4)
                t1.metric("Price", f"${row['price']:,.2f}")
                t2.metric("52W High", f"${row['high_52w']:,.2f}")
                t3.metric("% from High", f"{row['pct_from_high']*100:.1f}%")
                t4.metric("Vol Up/Down", f"{row['vol_ratio']:.2f}")
                st.markdown("#### Returns & Risk")
                s1,s2,s3,s4 = st.columns(4)
                s1.metric("3M Return", f"{row.get('ret_3m',0)*100:+.1f}%")
                s2.metric("6M Return", f"{row.get('ret_6m',0)*100:+.1f}%")
                s3.metric("12M Return", f"{row.get('ret_12m',0)*100:+.1f}%")
                atr_pct = row.get("atr_pct", 0)
                s4.metric("ATR%", f"{atr_pct*100:.2f}%" if atr_pct else "N/A")
                st.markdown("#### Additional Info")
                x1,x2,x3,x4 = st.columns(4)
                x1.metric("Sector", str(row.get("sector", "Unknown")))
                x2.metric("Ensemble?", "✅ Yes" if row.get("ensemble_consensus", False) else "No")
                x3.metric("Earnings Soon?", "📅 Yes" if row.get("earnings_within_30d", False) else "No")
                dd = row.get("max_dd_60d", None)
                x4.metric("60d Max DD", f"{dd*100:.1f}%" if dd is not None else "N/A")

with tab3:
    st.markdown("### 🛡️ Position Risk Management")
    st.markdown("<p style='color:#94a3b8;'>Select a stock to see trailing stop levels and exit signals.</p>", unsafe_allow_html=True)
    if qualified.empty:
        st.info("No qualifying stocks.")
    else:
        ticker_list_exit = qualified["ticker"].tolist() if "ticker" in qualified.columns else qualified.index.tolist()
        if isinstance(ticker_list_exit, pd.Index): ticker_list_exit = ticker_list_exit.tolist()
        selected_exit = st.selectbox("Select stock for exit analysis", ticker_list_exit, key="exit_select")
        if selected_exit:
            exits = compute_exit_signals(price_data, selected_exit)
            if exits:
                st.markdown(f"#### {selected_exit} — Exit Signal Dashboard")
                e1,e2,e3,e4 = st.columns(4)
                e1.metric("Current Price", f"${exits['current_price']:,.2f}")
                e2.metric("Trailing Stop", f"${exits['trailing_stop']:,.2f}")
                e3.metric("ATR Stop (2×)", f"${exits['atr_stop']:,.2f}")
                e4.metric("15% Stop", f"${exits['pct_stop_15']:,.2f}")
                st.markdown("#### Signal Status")
                sig1,sig2,sig3 = st.columns(3)
                sig1.markdown(f"**ATR Stop:** {'🔴 TRIGGERED' if exits.get('exit_signal_atr',False) else '🟢 Safe'}")
                sig2.markdown(f"**50 SMA Stop:** {'🔴 TRIGGERED' if exits.get('exit_signal_sma50',False) else '🟢 Safe'}")
                sig3.markdown(f"**15% Stop:** {'🔴 TRIGGERED' if exits.get('exit_signal_pct',False) else '🟢 Safe'}")
                if selected_exit in price_data:
                    exit_df = price_data[selected_exit]
                    recent_close = exit_df["Close"].iloc[-60:]
                    fig_exit = go.Figure()
                    fig_exit.add_trace(go.Scatter(x=recent_close.index, y=recent_close.values, name="Price", line=dict(color="#e2e8f0", width=2)))
                    fig_exit.add_hline(y=exits["trailing_stop"], line_dash="dash", line_color="#ef4444", annotation_text="Trailing Stop")
                    fig_exit.add_hline(y=exits["atr_stop"], line_dash="dot", line_color="#f59e0b", annotation_text="ATR Stop")
                    if exits.get("sma50_stop"):
                        fig_exit.add_hline(y=exits["sma50_stop"], line_dash="dot", line_color="#3b82f6", annotation_text="50 SMA")
                    fig_exit.update_layout(title=f"{selected_exit} — Price vs Stop Levels (60d)", height=400, paper_bgcolor="#0a0e17", plot_bgcolor="#111827", font=dict(color="#94a3b8"), xaxis=dict(gridcolor="#1e2d3d"), yaxis=dict(gridcolor="#1e2d3d", title="Price ($)"))
                    st.plotly_chart(fig_exit, width="stretch")

with tab4:
    st.markdown("### 🔬 Walk-Forward Backtest")
    st.markdown("<p style='color:#94a3b8;'>Pick any historical date. The screener runs as-of that date, selects top stocks, then measures actual forward returns.</p>", unsafe_allow_html=True)
    bc1,bc2,bc3,bc4 = st.columns(4)
    with bc1:
        bt_date = st.date_input("Scan Date", value=dt.date(2024, 1, 2), min_value=dt.date(2020, 1, 1), max_value=dt.date.today() - dt.timedelta(days=100))
    with bc2:
        bt_hold = st.selectbox("Hold Period", [3, 6], index=1, format_func=lambda x: f"{x} months")
    with bc3:
        bt_top_n = st.slider("Top N Picks", 5, 30, 20, 5, key="bt_top_n")
    with bc4:
        bt_min_score = st.slider("Min Score", 0.0, 10.0, 5.0, 0.5, key="bt_min_score")
    if st.button("▶️ Run Backtest", width="stretch", key="bt_run"):
        bt_progress = st.progress(0)
        bt_status = st.empty()
        def _bt_progress(pct, msg=""):
            bt_progress.progress(min(pct, 1.0))
            if msg: bt_status.markdown(f"<p style='color:#94a3b8;text-align:center;'>{msg}</p>", unsafe_allow_html=True)
        with st.spinner("Running historical backtest…"):
            bt_results = run_backtest(scan_date=bt_date.strftime("%Y-%m-%d"), hold_months=bt_hold, top_n=bt_top_n, min_score=bt_min_score, progress_callback=_bt_progress)
        bt_progress.empty()
        bt_status.empty()
        if "error" in bt_results:
            st.error(bt_results["error"])
        else:
            st.markdown(f"#### Results: {bt_results['scan_date']} → {bt_results['exit_date']}")
            r1,r2,r3,r4,r5 = st.columns(5)
            r1.metric("Portfolio Return", f"{bt_results['portfolio_return']*100:+.1f}%")
            r2.metric("SPY Return", f"{bt_results['spy_return']*100:+.1f}%")
            alpha = bt_results["alpha"]
            r3.metric("Alpha", f"{alpha*100:+.1f}%", delta=f"{'Outperformed' if alpha > 0 else 'Underperformed'}")
            r4.metric("Win Rate", f"{bt_results['win_rate']*100:.0f}%")
            r5.metric("Doubler Rate", f"{bt_results['doubler_rate']*100:.0f}%")
            bt_regime = bt_results.get("regime", {})
            st.markdown(f"**Regime at scan:** {bt_regime.get('regime','N/A')} (multiplier: {bt_regime.get('score_multiplier','N/A')})")
            bw1,bw2 = st.columns(2)
            bw1.metric("Best Pick", f"{bt_results['best_pick']}", delta=f"{bt_results['best_return']*100:+.1f}%")
            bw2.metric("Worst Pick", f"{bt_results['worst_pick']}", delta=f"{bt_results['worst_return']*100:+.1f}%")
            picks_df = bt_results.get("picks", pd.DataFrame())
            if not picks_df.empty and "forward_return" in picks_df.columns:
                st.markdown("#### Individual Pick Returns")
                picks_sorted = picks_df.dropna(subset=["forward_return"]).sort_values("forward_return", ascending=True)
                colors = ["#22c55e" if r > 0 else "#ef4444" for r in picks_sorted["forward_return"]]
                fig_bt = go.Figure(go.Bar(y=picks_sorted["ticker"], x=picks_sorted["forward_return"]*100, orientation="h", marker_color=colors, text=[f"{r*100:+.1f}%" for r in picks_sorted["forward_return"]], textposition="outside"))
                fig_bt.update_layout(title=f"Forward {bt_hold}-Month Returns by Pick", height=max(400, len(picks_sorted)*28), paper_bgcolor="#0a0e17", plot_bgcolor="#111827", font=dict(color="#94a3b8"), xaxis=dict(title="Return (%)", gridcolor="#1e2d3d", zeroline=True, zerolinecolor="#94a3b8"), yaxis=dict(gridcolor="#1e2d3d"))
                st.plotly_chart(fig_bt, width="stretch")
                display_picks = picks_sorted[["ticker","composite_score","entry_price","exit_price","forward_return"]].copy()
                display_picks.columns = ["Ticker","Score","Entry $","Exit $","Return"]
                st.dataframe(display_picks.style.format({"Score":"{:.1f}","Entry $":"${:,.2f}","Exit $":"${:,.2f}","Return":"{:+.1%}"}), width="stretch")

with tab5:
    st.markdown("### 📋 Monthly Top 20 Picks — Full History")
    st.markdown("<p style='color:#94a3b8;'>Top 20 stocks on the 1st trading day of each month, ranked by composite score.</p>", unsafe_allow_html=True)
    
    import os
    picks_file = None
    for f in ["monthly_picks.csv", "data/monthly_picks.csv", "/mount/src/doubler-screener/monthly_picks.csv"]:
        if os.path.exists(f):
            picks_file = f
            break
    
    if picks_file:
        mp = pd.read_csv(picks_file)
        # Keep only top 10 per month
        mp_top10 = mp[mp["Rank"] <= 20].copy()
        
        available_months = sorted(mp_top10["Scan_Date"].unique(), reverse=True)
        
        # Month filter
        view_option = st.radio("View", ["All Months", "Single Month"], horizontal=True, key="results_view")
        
        if view_option == "Single Month":
            sel_month = st.selectbox("Select Month", available_months, key="results_month")
            show_data = mp_top10[mp_top10["Scan_Date"] == sel_month].copy()
        else:
            show_data = mp_top10.copy()
        
        if show_data.empty:
            st.warning("No picks available for the selected period.")
        else:
            # Display columns
            display_cols = ["Scan_Date","Rank","Ticker","Score","Price","Trend","RS","Volume","Quality","Regime"]
            has_fwd = show_data["Fwd_3M"].notna().any()
            if has_fwd:
                display_cols += ["Fwd_3M","Fwd_6M","Alpha_3M","Alpha_6M"]
            
            avail = [c for c in display_cols if c in show_data.columns]
            show_df = show_data[avail].copy()
            
            fmt = {"Score":"{:.1f}","Price":"${:,.2f}","RS":"{:.1f}","Volume":"{:.1f}","Quality":"{:.1f}"}
            if "Fwd_3M" in avail: fmt["Fwd_3M"] = "{:+.1f}%"
            if "Fwd_6M" in avail: fmt["Fwd_6M"] = "{:+.1f}%"
            if "Alpha_3M" in avail: fmt["Alpha_3M"] = "{:+.1f}%"
            if "Alpha_6M" in avail: fmt["Alpha_6M"] = "{:+.1f}%"
            if "Fwd_12M" in avail: fmt["Fwd_12M"] = "{:+.1f}%"
            if "Alpha_12M" in avail: fmt["Alpha_12M"] = "{:+.1f}%"
            
            st.dataframe(show_df.style.format(fmt), height=min(800, 40 + len(show_df) * 35))
            
            # Download
            csv_data = show_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Monthly Picks (CSV)",
                data=csv_data,
                file_name=f"monthly_top10_picks.csv",
                mime="text/csv",
            )
            
            # Summary
            st.markdown("---")
            st.markdown("#### Performance Summary (Top 20 Picks)")
            v3 = show_data.dropna(subset=["Fwd_3M"])
            v6 = show_data.dropna(subset=["Fwd_6M"])
            if len(v3) > 0:
                s1,s2,s3,s4 = st.columns(4)
                s1.metric("Avg 3M Return", f"{v3.Fwd_3M.mean():+.1f}%")
                s2.metric("SPY 3M", f"{v3.SPY_3M.mean():+.1f}%")
                s3.metric("3M Alpha", f"{v3.Alpha_3M.mean():+.1f}%")
                s4.metric("3M Win Rate", f"{(v3.Fwd_3M>0).mean()*100:.0f}%")
            if len(v6) > 0:
                s5,s6,s7,s8 = st.columns(4)
                s5.metric("Avg 6M Return", f"{v6.Fwd_6M.mean():+.1f}%")
                s6.metric("SPY 6M", f"{v6.SPY_6M.mean():+.1f}%")
                s7.metric("6M Alpha", f"{v6.Alpha_6M.mean():+.1f}%")
                s8.metric("6M Win Rate", f"{(v6.Fwd_6M>0).mean()*100:.0f}%")
            v12 = show_data.dropna(subset=["Fwd_12M"])
            if len(v12) > 0:
                s9,s10,s11,s12 = st.columns(4)
                s9.metric("Avg 12M Return", f"{v12.Fwd_12M.mean():+.1f}%")
                s10.metric("SPY 12M", f"{v12.SPY_12M.mean():+.1f}%")
                s11.metric("12M Alpha", f"{v12.Alpha_12M.mean():+.1f}%")
                s12.metric("12M Win Rate", f"{(v12.Fwd_12M>0).mean()*100:.0f}%")
    else:
        st.info("No monthly picks data found. Generate monthly_picks.csv locally first.")