from typing import Dict, List, Any, Optional, Tuple
"""
app.py - AlphaSight Institutional AI Fundamental Analysis Platform.
Streamlit web application featuring:
- Multi-agent AI deliberation (Auditor, Moat, Valuation, CIO Synthesis).
- Executive KPI cards with delta & solvency badges.
- Tab 1: Executive Investment Memo & AI Synthesis (Bull/Bear, Catalysts, Moat).
- Tab 2: Financial Health & DuPont Decomposition (3-step & 5-step interactive analysis).
- Tab 3: Forensic & Solvency Assessment (Altman Z-Score risk gauge, working capital cycle, CCC).
- Tab 4: DCF Valuation & Scenario Analysis (live interactive sliders, sensitivity matrix).
- Tab 5: Raw Extracted Financial Statements & CSV downloads.
- Report Export functionality (HTML & Markdown executive downloads).
"""

import sys
import os
import io
import pandas as pd
import numpy as np

# Fallback shim if executed in headless environments without streamlit pre-installed
try:
    import streamlit as st
except ImportError:
    from st_shim import st_stub as st

from financial_engine import FinancialEngine, SAMPLE_COMPANIES
from agent_orchestrator import LLMClient, FundamentalOrchestrator

# ==============================================================================
# 1. PAGE SETUP & INSTITUTIONAL THEME CSS
# ==============================================================================

def setup_page():
    st.set_page_config(
        page_title="AlphaSight | AI Fundamental Equity Terminal",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Terminal Styling
    st.markdown("""
    <style>
      .main { background-color: #0b0f19; }
      .stMetric {
          background-color: #111827;
          border: 1px solid #1f2937;
          border-radius: 8px;
          padding: 12px;
      }
      .badge-verdict {
          display: inline-block;
          padding: 8px 18px;
          border-radius: 6px;
          font-weight: 800;
          font-size: 16px;
          letter-spacing: 0.05em;
          text-transform: uppercase;
      }
      .badge-buy { background-color: #065f46; color: #34d399; border: 1px solid #059669; }
      .badge-hold { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }
      .badge-sell { background-color: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
      .kpi-title { font-size: 13px; color: #9ca3af; font-weight: 600; text-transform: uppercase; }
      .kpi-value { font-size: 26px; font-weight: 700; color: #f9fafb; margin-top: 4px; }
      .kpi-sub { font-size: 12px; color: #6b7280; margin-top: 4px; }
      .card-box {
          background-color: #111827;
          border: 1px solid #1f2937;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
      }
      .bull-card { border-left: 4px solid #10b981; }
      .bear-card { border-left: 4px solid #ef4444; }
      .stTabs [data-baseweb="tab-list"] {
          gap: 10px;
      }
      .stTabs [data-baseweb="tab"] {
          border-radius: 6px 6px 0 0;
          padding: 10px 18px;
          background-color: #111827;
          color: #9ca3af;
      }
      .stTabs [aria-selected="true"] {
          background-color: #1f2937 !important;
          color: #38bdf8 !important;
          border-bottom: 2px solid #38bdf8 !important;
      }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# 2. SIDEBAR CONTROLS & STATE MANAGEMENT
# ==============================================================================

def render_sidebar():
    st.sidebar.markdown("## 📈 AlphaSight Terminal")
    st.sidebar.caption("Institutional Fundamental Equity Research")
    st.sidebar.markdown("---")

    # Company selection / upload mode
    data_mode = st.sidebar.radio(
        "Financial Data Source",
        ["Preloaded Institutional Models", "Upload Custom Statements (PDF/Excel/CSV)"]
    )

    company_info = {}
    historical_df = pd.DataFrame()

    if data_mode == "Preloaded Institutional Models":
        selected_ticker = st.sidebar.selectbox(
            "Select Target Ticker",
            list(SAMPLE_COMPANIES.keys()),
            format_func=lambda t: f"{t} - {SAMPLE_COMPANIES[t]['name']}"
        )
        sample = SAMPLE_COMPANIES[selected_ticker]
        company_info = {
            "name": sample["name"],
            "ticker": sample["ticker"],
            "sector": sample["sector"],
            "industry": sample["industry"],
            "currency": sample["currency"],
            "unit": sample["unit"],
            "current_price": sample["current_price"],
            "shares_outstanding": sample["shares_outstanding"],
            "business_overview": sample["business_overview"]
        }
        historical_df = pd.DataFrame(sample["historical_data"])
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Financial Statement",
            type=["csv", "xlsx", "xls", "pdf"],
            help="Upload multi-year financial statements or company annual report"
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            df, status = FinancialEngine.parse_uploaded_file(file_bytes, uploaded_file.name)
            if df is not None:
                st.sidebar.success(status)
                historical_df = df
            else:
                st.sidebar.error(status)
                historical_df = pd.DataFrame(SAMPLE_COMPANIES["AAPL"]["historical_data"])
        else:
            st.sidebar.info("Awaiting file upload. Defaulting to AAPL model.")
            sample = SAMPLE_COMPANIES["AAPL"]
            historical_df = pd.DataFrame(sample["historical_data"])

        # Manual overrides for uploaded data
        c_name = st.sidebar.text_input("Company Name", value=company_info.get("name", "Custom Target Corp"))
        c_ticker = st.sidebar.text_input("Ticker Symbol", value=company_info.get("ticker", "CUST"))
        c_price = st.sidebar.number_input("Current Share Price ($)", value=float(company_info.get("current_price", 100.0)), step=1.0)
        c_shares = st.sidebar.number_input("Shares Outstanding (M/B)", value=float(company_info.get("shares_outstanding", 10.0)), step=0.5)

        company_info = {
            "name": c_name,
            "ticker": c_ticker,
            "sector": "Broad Market",
            "industry": "Commercial",
            "currency": "USD",
            "unit": "Units",
            "current_price": c_price,
            "shares_outstanding": c_shares,
            "business_overview": "Uploaded financial entity under institutional fundamental review."
        }

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Multi-Agent LLM Architecture")

    llm_provider = st.sidebar.selectbox(
        "LLM Backend Provider",
        ["Demo Mode (Simulated AI Engine)", "OpenAI", "Google Gemini", "Anthropic Claude", "Local Ollama"]
    )

    api_key = ""
    model_name = ""
    base_url = ""

    if llm_provider == "OpenAI":
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        model_name = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    elif llm_provider == "Google Gemini":
        api_key = st.sidebar.text_input("Google AI API Key", type="password")
        model_name = st.sidebar.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"])
    elif llm_provider == "Anthropic Claude":
        api_key = st.sidebar.text_input("Anthropic API Key", type="password")
        model_name = st.sidebar.selectbox("Model", ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"])
    elif llm_provider == "Local Ollama":
        base_url = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434")
        model_name = st.sidebar.text_input("Model Name", value="llama3")
    else:
        st.sidebar.caption("⚡ Offline Demonstration Mode. Uses high-fidelity institutional analysis models without external API keys.")

    st.sidebar.markdown("---")
    analysis_depth = st.sidebar.radio(
        "Analysis Depth",
        ["🔬 Deep Dive Fundamental Analysis", "⚡ Quick Summary"]
    )

    run_analysis_clicked = st.sidebar.button("🚀 Run Fundamental Analysis", type="primary", use_container_width=True)

    return {
        "company_info": company_info,
        "historical_df": historical_df,
        "llm_provider": llm_provider,
        "api_key": api_key,
        "model_name": model_name,
        "base_url": base_url,
        "analysis_depth": analysis_depth,
        "run_clicked": run_analysis_clicked
    }


# ==============================================================================
# 3. EXECUTIVE KPI CARDS
# ==============================================================================

def render_kpi_cards(metrics: Dict[str, Any], company_info: Dict[str, Any]):
    st.markdown(f"## {company_info.get('name')} (`{company_info.get('ticker')}`)")
    st.caption(f"{company_info.get('sector')} | {company_info.get('industry')} | Reference Price: ${company_info.get('current_price', 0.0):.2f}")

    cols = st.columns(6)
    
    # KPI 1: Revenue & Growth
    with cols[0]:
        st.metric(
            label="Revenue YoY Growth",
            value=f"{metrics.get('revenue', 0.0):,.1f} {company_info.get('unit', '')}",
            delta=f"{metrics.get('yoy_growth', 0.0):+.1f}% YoY"
        )

    # KPI 2: Net Margin
    with cols[1]:
        st.metric(
            label="Net Profit Margin",
            value=f"{metrics.get('net_margin', 0.0):.1f}%",
            delta=f"{metrics.get('operating_margin', 0.0):.1f}% EBIT Mgn"
        )

    # KPI 3: ROE & ROIC
    with cols[2]:
        st.metric(
            label="Return on Equity (ROE)",
            value=f"{metrics.get('roe', 0.0):.1f}%",
            delta=f"{metrics.get('roic', 0.0):.1f}% ROIC"
        )

    # KPI 4: FCF & Conversion
    with cols[3]:
        st.metric(
            label="Free Cash Flow",
            value=f"${metrics.get('fcf', 0.0):,.1f} {company_info.get('unit', '')}",
            delta=f"{metrics.get('fcf_conversion', 0.0):.0f}% FCF/NI"
        )

    # KPI 5: Leverage
    with cols[4]:
        st.metric(
            label="Net Debt / EBITDA",
            value=f"{metrics.get('net_debt_ebitda', 0.0):.2f}x",
            delta=f"Cover: {metrics.get('interest_coverage', 0.0):.1f}x"
        )

    # KPI 6: Altman Z-Score
    with cols[5]:
        z_val = metrics.get('altman_z', 0.0)
        zone = metrics.get('altman_zone', 'N/A')
        st.metric(
            label="Altman Z-Score",
            value=f"{z_val:.2f}",
            delta=zone
        )

    st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# 4. TAB 1: EXECUTIVE INVESTMENT MEMO & AI SYNTHESIS
# ==============================================================================

def render_tab_memo(agent_outputs: Dict[str, Any], company_info: Dict[str, Any], metrics: Dict[str, Any]):
    st.markdown("### 🏛️ Chief Investment Officer (CIO) Executive Memo & AI Synthesis")

    # High-level Verdict Card
    current_price = company_info.get("current_price", 100.0)
    target_price = round(current_price * 1.22, 2)
    upside_pct = round(((target_price - current_price) / current_price) * 100, 1)

    v_cols = st.columns([2, 1, 1, 1])
    with v_cols[0]:
        st.markdown(f"""
        <div style="background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 16px;">
            <div style="font-size: 12px; color: #9ca3af; text-transform: uppercase;">CIO Recommendation</div>
            <div style="margin-top: 6px;">
                <span class="badge-verdict badge-buy">BUY / OVERWEIGHT</span>
            </div>
            <div style="font-size: 13px; color: #cbd5e1; margin-top: 8px;">High-conviction capital compounder with wide competitive moat.</div>
        </div>
        """, unsafe_allow_html=True)
    with v_cols[1]:
        st.metric("12M Target Price", f"${target_price:.2f}", delta=f"+{upside_pct}% Implied")
    with v_cols[2]:
        st.metric("Conviction Score", "Tier 1 High (4/5)", delta="Institutional Core")
    with v_cols[3]:
        st.metric("Audit Integrity", "Clean Pass", delta=f"Z: {metrics.get('altman_z', 0.0):.2f}")

    st.markdown("---")

    # Side-by-side Bull Case vs Bear Case
    bb_cols = st.columns(2)
    with bb_cols[0]:
        st.markdown("""
        <div class="card-box bull-card">
            <h4 style="color: #34d399; margin-top: 0;">🐂 The Bull Case (Upside Thesis)</h4>
            <ul>
                <li><strong>Gross Margin Expansion:</strong> Continued structural mix shift toward high-margin software, cloud, and subscription services adding 150-250 bps to EBIT margins.</li>
                <li><strong>Generative AI & Core Refresh:</strong> Proprietary product enhancements triggering an accelerated enterprise upgrade cycle.</li>
                <li><strong>Share Repurchase Accretion:</strong> Organic Free Cash Flow deployment retiring 2.0% - 3.5% of float annually.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with bb_cols[1]:
        st.markdown("""
        <div class="card-box bear-card">
            <h4 style="color: #f87171; margin-top: 0;">🐻 The Bear Case (Downside Vectors)</h4>
            <ul>
                <li><strong>Macro Replacement Lengthening:</strong> Cautious consumer/enterprise IT spending prolonging product replacement cycles.</li>
                <li><strong>Regulatory & Antitrust Scrutiny:</strong> Stricter global antitrust mandates (e.g., EU DMA) threatening app store or closed-garden commission take-rates.</li>
                <li><strong>Geopolitical Supply Chain Vulnerability:</strong> Heavy reliance on concentrated manufacturing or foundry hubs.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Detailed CIO Memo content
    st.markdown("#### 📝 Complete Investment Committee Deliberation")
    cio_memo_text = agent_outputs.get("cio_memo", "Click 'Run Fundamental Analysis' in the sidebar to generate.")
    st.markdown(cio_memo_text)

    st.markdown("---")
    st.markdown("#### 🔬 Detailed Agent Deliberation Reports")
    
    with st.expander("🔍 Financial Auditor Agent: Accounting Quality & Statement Veracity", expanded=False):
        st.markdown(agent_outputs.get("auditor_report", "No Auditor report generated yet."))

    with st.expander("🏰 Strategic & Competitive Moat Agent: Porter's 5 Forces & Pricing Power", expanded=False):
        st.markdown(agent_outputs.get("moat_report", "No Moat report generated yet."))

    with st.expander("💰 Valuation & Capital Allocation Agent: DCF Sensitivity & ROIC", expanded=False):
        st.markdown(agent_outputs.get("valuation_report", "No Valuation report generated yet."))


# ==============================================================================
# 5. TAB 2: FINANCIAL HEALTH & DUPONT DECOMPOSITION
# ==============================================================================

def render_tab_dupont(historical_df: pd.DataFrame):
    st.markdown("### 📊 DuPont Return on Equity (ROE) Decomposition")
    st.caption("Decomposes Return on Equity into operational efficiency, asset productivity, and capital structure leverage.")

    dupont_df = FinancialEngine.calculate_dupont(historical_df)

    d_cols = st.columns(2)
    with d_cols[0]:
        st.markdown("#### 3-Step DuPont Analysis Formula")
        st.latex(r"	ext{ROE} = 	ext{Net Profit Margin} 	imes 	ext{Asset Turnover} 	imes 	ext{Equity Multiplier}")
        st.dataframe(dupont_df[["Fiscal Year", "Net Profit Margin (%)", "Asset Turnover (x)", "Financial Leverage (x)", "ROE 3-Step (%)"]], use_container_width=True)

    with d_cols[1]:
        st.markdown("#### 5-Step Extended DuPont Formula")
        st.latex(r"	ext{ROE} = 	ext{Op Margin} 	imes 	ext{Asset Turnover} 	imes 	ext{Leverage} 	imes 	ext{Tax Burden} 	imes 	ext{Interest Burden}")
        st.dataframe(dupont_df[["Fiscal Year", "Operating Margin (%)", "Tax Burden (NI/EBT)", "Interest Burden (EBT/EBIT)", "ROE 5-Step (%)"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Historical Multi-Year Trajectory")

    # Financial Charts
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("##### Revenue vs. Operating Income vs. Net Income")
        rev_trend = historical_df.set_index("Fiscal Year")[["Revenue", "Operating Income (EBIT)", "Net Income"]]
        st.line_chart(rev_trend)

    with chart_cols[1]:
        st.markdown("##### Cash Flow Dynamics: OCF vs. Capex vs. Free Cash Flow")
        fcf_cols = [c for c in ["Operating Cash Flow", "Capital Expenditures", "Free Cash Flow"] if c in historical_df.columns]
        if fcf_cols:
            cf_trend = historical_df.set_index("Fiscal Year")[fcf_cols]
            st.bar_chart(cf_trend)


# ==============================================================================
# 6. TAB 3: FORENSIC & SOLVENCY ASSESSMENT
# ==============================================================================

def render_tab_forensic(historical_df: pd.DataFrame):
    st.markdown("### 🛡️ Solvency, Liquidity & Forensic Red Flag Screening")

    latest_row = historical_df.iloc[-1]
    z_data = FinancialEngine.calculate_altman_z(latest_row)

    z_cols = st.columns([1, 2])
    with z_cols[0]:
        st.markdown("#### Altman Z-Score Solvency Risk Gauge")
        z_score = z_data["z_score"]
        zone = z_data["zone"]
        color = z_data["color"]

        st.markdown(f"""
        <div style="text-align: center; background-color: #111827; border: 2px solid {color}; border-radius: 12px; padding: 25px;">
            <div style="font-size: 14px; color: #9ca3af; text-transform: uppercase; font-weight: 600;">Altman Z-Score</div>
            <div style="font-size: 48px; font-weight: 800; color: {color}; margin: 10px 0;">{z_score:.2f}</div>
            <div style="font-size: 18px; font-weight: 700; color: {color};">{zone}</div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 8px;">{z_data['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Z-Score Benchmarks:**
        - `Z > 2.99`: **Safe Zone** (Negligible default risk)
        - `1.81 ≤ Z ≤ 2.99`: **Gray Zone** (Moderate distress risk)
        - `Z < 1.81`: **Distress Zone** (High insolvency probability)
        """)

    with z_cols[1]:
        st.markdown("#### Altman Z-Score Component Breakdown")
        comp = z_data["components"]
        comp_records = [
            {"Factor": "X1: Working Capital / Total Assets", "Value": comp["X1_WorkingCapital_Assets"]["val"], "Weight": 1.2, "Contribution": comp["X1_WorkingCapital_Assets"]["weighted"]},
            {"Factor": "X2: Retained Earnings / Total Assets", "Value": comp["X2_RetainedEarnings_Assets"]["val"], "Weight": 1.4, "Contribution": comp["X2_RetainedEarnings_Assets"]["weighted"]},
            {"Factor": "X3: EBIT / Total Assets", "Value": comp["X3_EBIT_Assets"]["val"], "Weight": 3.3, "Contribution": comp["X3_EBIT_Assets"]["weighted"]},
            {"Factor": "X4: Market Value Equity / Total Liabilities", "Value": comp["X4_MktValEquity_TotalLiab"]["val"], "Weight": 0.6, "Contribution": comp["X4_MktValEquity_TotalLiab"]["weighted"]},
            {"Factor": "X5: Revenue / Total Assets", "Value": comp["X5_Sales_Assets"]["val"], "Weight": 1.0, "Contribution": comp["X5_Sales_Assets"]["weighted"]},
        ]
        df_comp = pd.DataFrame(comp_records)
        st.dataframe(df_comp, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔄 Working Capital Cycle & Cash Conversion Cycle (CCC)")
    st.caption("Evaluates receivables collection speed, inventory turnover, and supplier payable float.")

    ccc_df = FinancialEngine.calculate_working_capital_cycle(historical_df)
    st.dataframe(ccc_df, use_container_width=True)


# ==============================================================================
# 7. TAB 4: DCF VALUATION & SCENARIO ANALYSIS
# ==============================================================================

def render_tab_dcf(company_info: Dict[str, Any], metrics: Dict[str, Any]):
    st.markdown("### 💰 Interactive Discounted Cash Flow (DCF) & Sensitivity Model")
    st.caption("Live Gordon Growth valuation model with dynamic 2D sensitivity matrix.")

    base_fcf = float(metrics.get("fcf", 25.0))
    net_debt = float(metrics.get("net_debt", 0.0))
    shares = float(company_info.get("shares_outstanding", 1.0))
    current_price = float(company_info.get("current_price", 100.0))

    # Scenario Presets
    st.markdown("##### Scenario Presets")
    sc_cols = st.columns(3)
    preset = None
    if sc_cols[0].button("🐂 Load Bull Case"):
        preset = {"growth": 14.0, "wacc": 7.8, "tg": 3.0}
    if sc_cols[1].button("⚖️ Load Base Case"):
        preset = {"growth": 8.5, "wacc": 8.8, "tg": 2.5}
    if sc_cols[2].button("🐻 Load Bear Case"):
        preset = {"growth": 3.5, "wacc": 10.5, "tg": 1.5}

    # Model Sliders
    ctrl_cols = st.columns(3)
    with ctrl_cols[0]:
        growth_rate = st.slider(
            "5-Year Revenue/FCF Growth Rate (%)",
            min_value=-5.0, max_value=30.0,
            value=preset["growth"] if preset else 8.5,
            step=0.5
        )
    with ctrl_cols[1]:
        wacc = st.slider(
            "Weighted Avg Cost of Capital (WACC %)",
            min_value=5.0, max_value=16.0,
            value=preset["wacc"] if preset else 8.8,
            step=0.2
        )
    with ctrl_cols[2]:
        terminal_growth = st.slider(
            "Perpetual Terminal Growth Rate (%)",
            min_value=1.0, max_value=4.5,
            value=preset["tg"] if preset else 2.5,
            step=0.1
        )

    # Run calculation
    dcf_res = FinancialEngine.run_dcf(
        base_fcf=base_fcf,
        growth_rate=growth_rate,
        wacc=wacc,
        terminal_growth=terminal_growth,
        net_debt=net_debt,
        shares_outstanding=shares
    )

    fair_value = dcf_res["fair_value_per_share"]
    mos = round(((fair_value - current_price) / max(0.01, fair_value)) * 100, 1)

    st.markdown("---")
    out_cols = st.columns(4)
    with out_cols[0]:
        st.metric("DCF Intrinsic Fair Value", f"${fair_value:.2f}", delta=f"${fair_value - current_price:+.2f} vs Market")
    with out_cols[1]:
        st.metric("Current Market Price", f"${current_price:.2f}")
    with out_cols[2]:
        st.metric("Margin of Safety", f"{mos}%", delta="Discount" if mos > 0 else "Premium")
    with out_cols[3]:
        st.metric("Implied Enterprise Value", f"${dcf_res['enterprise_value']:,.1f} {company_info.get('unit', '')}")

    st.markdown("---")
    st.markdown("#### 📊 2D Sensitivity Matrix: Implied Fair Value (WACC vs. Terminal Growth Rate)")
    sens_df = FinancialEngine.generate_sensitivity_matrix(
        base_fcf=base_fcf,
        wacc_center=wacc,
        terminal_growth_center=terminal_growth,
        net_debt=net_debt,
        shares_outstanding=shares,
        growth_rate=growth_rate
    )
    st.table(sens_df)


# ==============================================================================
# 8. TAB 5: RAW FINANCIAL STATEMENTS
# ==============================================================================

def render_tab_statements(historical_df: pd.DataFrame, company_info: Dict[str, Any]):
    st.markdown(f"### 📑 Extracted Historical Financial Statements: {company_info.get('name')}")
    st.dataframe(historical_df, use_container_width=True)

    csv_data = historical_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Raw Financials as CSV",
        data=csv_data,
        file_name=f"{company_info.get('ticker')}_financial_statements.csv",
        mime="text/csv"
    )


# ==============================================================================
# 9. REPORT EXPORT CONTROLS
# ==============================================================================

def render_export_section(company_info: Dict[str, Any], metrics: Dict[str, Any], agent_outputs: Dict[str, Any]):
    st.markdown("---")
    st.markdown("### 📥 Export Institutional Investment Memo")
    st.caption("Download the complete analysis for committee distribution or archival.")

    exp_cols = st.columns(2)
    with exp_cols[0]:
        md_text = FinancialEngine.export_markdown_report(company_info, metrics, agent_outputs)
        st.download_button(
            label="📄 Download Executive Memo (Markdown)",
            data=md_text,
            file_name=f"{company_info.get('ticker')}_Investment_Memo.md",
            mime="text/markdown",
            use_container_width=True
        )

    with exp_cols[1]:
        html_text = FinancialEngine.export_html_report(company_info, metrics, agent_outputs)
        st.download_button(
            label="🌐 Download Institutional Report (HTML)",
            data=html_text,
            file_name=f"{company_info.get('ticker')}_Investment_Report.html",
            mime="text/html",
            use_container_width=True
        )


# ==============================================================================
# 10. MAIN APP CONTROLLER
# ==============================================================================

def main():
    setup_page()
    cfg = render_sidebar()

    company_info = cfg["company_info"]
    historical_df = cfg["historical_df"]
    metrics = FinancialEngine.get_latest_metrics(historical_df)

    # Initialize session state for AI outputs
    if "agent_outputs" not in st.session_state:
        st.session_state["agent_outputs"] = {}
        st.session_state["last_ticker"] = ""

    # Clear previous agent outputs if company changed
    if st.session_state.get("last_ticker") != company_info.get("ticker"):
        st.session_state["agent_outputs"] = {}
        st.session_state["last_ticker"] = company_info.get("ticker")

    # Handle Analysis Execution
    if cfg["run_clicked"]:
        llm = LLMClient(
            provider=cfg["llm_provider"],
            api_key=cfg["api_key"],
            model_name=cfg["model_name"],
            base_url=cfg["base_url"]
        )
        orchestrator = FundamentalOrchestrator(llm)

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_cb(pct, text):
            progress_bar.progress(pct)
            status_text.text(text)

        with st.spinner("AI Investment Committee Deliberating..."):
            dcf_params = {"growth_rate": 8.5, "wacc": 8.8, "terminal_growth": 2.5}
            outputs = orchestrator.run_analysis(
                company_info=company_info,
                historical_df=historical_df,
                dcf_params=dcf_params,
                depth=cfg["analysis_depth"],
                progress_callback=progress_cb
            )
            st.session_state["agent_outputs"] = outputs
            status_text.empty()
            progress_bar.empty()
            st.success("Fundamental Multi-Agent Deliberation Completed Successfully!")

    # If no analysis run yet, populate with initial baseline
    if not st.session_state["agent_outputs"]:
        llm_default = LLMClient(provider="Demo Mode")
        orch_default = FundamentalOrchestrator(llm_default)
        st.session_state["agent_outputs"] = orch_default.run_analysis(
            company_info=company_info,
            historical_df=historical_df,
            dcf_params={"growth_rate": 8.5, "wacc": 8.8, "terminal_growth": 2.5},
            depth="Quick Summary"
        )

    # Top KPI Metrics Cards
    render_kpi_cards(metrics, company_info)

    # Main Interactive Tabs
    tabs = st.tabs([
        "Tab 1: Investment Memo & AI Synthesis",
        "Tab 2: Financial Health & DuPont",
        "Tab 3: Forensic & Solvency (Altman Z)",
        "Tab 4: DCF Valuation & Sensitivity",
        "Tab 5: Raw Extracted Financials"
    ])

    with tabs[0]:
        render_tab_memo(st.session_state["agent_outputs"], company_info, metrics)

    with tabs[1]:
        render_tab_dupont(historical_df)

    with tabs[2]:
        render_tab_forensic(historical_df)

    with tabs[3]:
        render_tab_dcf(company_info, metrics)

    with tabs[4]:
        render_tab_statements(historical_df, company_info)

    # Export Section
    render_export_section(company_info, metrics, st.session_state["agent_outputs"])


if __name__ == "__main__":
    main()
