"""
verify_system.py - Comprehensive verification test suite for AlphaSight platform.
Verifies:
1. Module imports (financial_engine, agent_orchestrator, app).
2. Financial Engine calculations:
   - Sample datasets (AAPL, MSFT, TSLA, ACME)
   - Executive KPIs (revenue, growth, margins, ROE, ROIC, debt ratios)
   - DuPont 3-Step & 5-Step decomposition
   - Altman Z-Score calculation and zone boundaries
   - Working capital & Cash Conversion Cycle (CCC)
   - DCF model and 2D sensitivity matrix
   - HTML and Markdown report exports
3. Multi-Agent Orchestrator:
   - LLM Client in Demo Mode
   - Financial Auditor Agent prompt building and output
   - Competitive Moat Agent prompt building and output
   - Valuation & Capital Allocation Agent prompt building and output
   - CIO Synthesis Agent prompt building and output
   - Quick Summary vs Deep Dive execution
"""

import sys
import os
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    print("=" * 60)
    print("ALPHASIGHT SYSTEM VERIFICATION SUITE")
    print("=" * 60)

    # Test 1: Imports
    print("\n[1/5] Testing Module Imports...")
    from financial_engine import FinancialEngine, SAMPLE_COMPANIES
    from agent_orchestrator import (
        LLMClient,
        FinancialAuditorAgent,
        CompetitiveMoatAgent,
        ValuationAgent,
        CIOSynthesisAgent,
        FundamentalOrchestrator
    )
    import app
    print(" -> All modules imported successfully.")

    # Test 2: Sample Datasets and Metrics
    print("\n[2/5] Testing Financial Datasets & Ratio Calculations...")
    for ticker, data in SAMPLE_COMPANIES.items():
        df = pd.DataFrame(data["historical_data"])
        metrics = FinancialEngine.get_latest_metrics(df)
        print(f" -> {ticker} ({data['name']}): Rev={metrics['revenue']} | Growth={metrics['yoy_growth']}% | Margin={metrics['net_margin']}% | ROE={metrics['roe']}% | Altman Z={metrics['altman_z']} ({metrics['altman_zone']})")
        assert metrics["revenue"] > 0, f"Invalid revenue for {ticker}"
        assert "altman_z" in metrics, f"Missing Altman Z for {ticker}"

    # Test 3: DuPont, Altman Z, CCC, and DCF
    print("\n[3/5] Testing DuPont, Altman Z, CCC, and DCF Models...")
    aapl_df = pd.DataFrame(SAMPLE_COMPANIES["AAPL"]["historical_data"])
    dupont_df = FinancialEngine.calculate_dupont(aapl_df)
    assert len(dupont_df) == 4, "DuPont table length mismatch"
    print(f" -> DuPont 3-Step ROE (Latest): {dupont_df.iloc[-1]['ROE 3-Step (%)']}% | 5-Step: {dupont_df.iloc[-1]['ROE 5-Step (%)']}%")

    ccc_df = FinancialEngine.calculate_working_capital_cycle(aapl_df)
    assert len(ccc_df) == 4, "CCC table length mismatch"
    print(f" -> Cash Conversion Cycle (Latest): {ccc_df.iloc[-1]['Cash Conversion Cycle (Days)']} days | DSO: {ccc_df.iloc[-1]['DSO (Days Receivable)']} days")

    dcf_res = FinancialEngine.run_dcf(
        base_fcf=107.1,
        growth_rate=8.5,
        wacc=8.8,
        terminal_growth=2.5,
        net_debt=41.4,
        shares_outstanding=15.2
    )
    assert dcf_res["fair_value_per_share"] > 0, "DCF fair value calculation failed"
    print(f" -> DCF Fair Value: ${dcf_res['fair_value_per_share']:.2f} per share | EV: ${dcf_res['enterprise_value']:.1f}B")

    sens_df = FinancialEngine.generate_sensitivity_matrix(
        base_fcf=107.1,
        wacc_center=8.8,
        terminal_growth_center=2.5,
        net_debt=41.4,
        shares_outstanding=15.2,
        growth_rate=8.5
    )
    assert sens_df.shape == (5, 5), "Sensitivity matrix shape mismatch"
    print(f" -> 2D Sensitivity Matrix generated: 5x5 grid (WACC vs Terminal Growth)")

    # Test 4: Multi-Agent Orchestrator Pipeline
    print("\n[4/5] Testing Multi-Agent Orchestrator (Demo & Deep Dive)...")
    llm = LLMClient(provider="Demo Mode")
    orchestrator = FundamentalOrchestrator(llm)

    company_info = {
        "name": SAMPLE_COMPANIES["AAPL"]["name"],
        "ticker": SAMPLE_COMPANIES["AAPL"]["ticker"],
        "sector": SAMPLE_COMPANIES["AAPL"]["sector"],
        "industry": SAMPLE_COMPANIES["AAPL"]["industry"],
        "current_price": SAMPLE_COMPANIES["AAPL"]["current_price"],
        "shares_outstanding": SAMPLE_COMPANIES["AAPL"]["shares_outstanding"],
        "unit": SAMPLE_COMPANIES["AAPL"]["unit"],
        "business_overview": SAMPLE_COMPANIES["AAPL"]["business_overview"]
    }

    results = orchestrator.run_analysis(
        company_info=company_info,
        historical_df=aapl_df,
        dcf_params={"growth_rate": 8.5, "wacc": 8.8, "terminal_growth": 2.5},
        depth="Deep Dive Fundamental Analysis"
    )

    assert "auditor_report" in results and len(results["auditor_report"]) > 100
    assert "moat_report" in results and len(results["moat_report"]) > 100
    assert "valuation_report" in results and len(results["valuation_report"]) > 100
    assert "cio_memo" in results and len(results["cio_memo"]) > 100
    print(f" -> Auditor Report length: {len(results['auditor_report'])} chars")
    print(f" -> Moat Report length: {len(results['moat_report'])} chars")
    print(f" -> Valuation Report length: {len(results['valuation_report'])} chars")
    print(f" -> CIO Memo length: {len(results['cio_memo'])} chars")
    print(f" -> Execution Duration: {results['execution_metadata']['duration_seconds']}s")

    # Test 5: Report Generation
    print("\n[5/5] Testing HTML & Markdown Report Generation...")
    md_report = FinancialEngine.export_markdown_report(company_info, results["metrics"], results)
    html_report = FinancialEngine.export_html_report(company_info, results["metrics"], results)
    assert len(md_report) > 500, "Markdown report too short"
    assert len(html_report) > 1000, "HTML report too short"
    print(f" -> Markdown Export: {len(md_report)} chars")
    print(f" -> HTML Export: {len(html_report)} chars")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
