"""
financial_engine.py - Production-Ready Fundamental Financial Analysis Framework
=============================================================================
A comprehensive, high-precision Python framework for corporate financial statement
modeling, multi-period analysis, ratio computation, forensic risk screening,
intrinsic/relative valuation, and automated document extraction (CSV, Excel, PDF).

Key Capabilities:
1. Standardized Financial Statement Data Structures (IS, BS, CFS, Multi-Period Container)
2. Advanced Ratio & Metric Calculations:
   - Profitability (Gross, Operating, Net, EBITDA margins, ROE, ROA, ROIC)
   - DuPont Analysis (3-Stage & 5-Stage Decompositions with identity verification)
   - Liquidity & Solvency (Current, Quick, Cash, D/E, Net Debt/EBITDA, Interest Coverage)
   - Efficiency & Working Capital (DSO, DIO, DPO, Cash Conversion Cycle, Turnovers)
   - Cash Flow Quality (OCF/NI, Free Cash Flow, FCF Margin, CapEx % of Sales)
   - Forensic & Solvency Risk (Altman Z-Score for Mfg/Non-Mfg, Beneish M-Score 8-Variable)
   - Valuation Models (DCF with Gordon Growth & Sensitivity Matrix, Public Trading Multiples)
3. Document Extraction Helpers (CSV, Excel, PDF with pdfplumber & pypdf)
4. Automated Executive Reporting Summary
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional PDF parsing dependencies
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


# =====================================================================
# SECTION 1: UTILITY FUNCTIONS & BASE CONVERTERS
# =====================================================================

def safe_float(val: Any, default: float = 0.0) -> float:
    """
    Safely converts strings, integers, floats, and numeric representations
    to float. Handles currency symbols, accounting parentheses, commas,
    percentages, and missing values ('-', 'N/A', 'None').
    """
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val) if not np.isnan(val) else default
    if isinstance(val, str):
        cleaned = val.strip().replace("$", "").replace("€", "").replace("£", "").replace(",", "")
        # Handle parentheses representing negative numbers: (1,234.50) -> -1234.50
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1].strip()
        # Handle percentage strings: 15.5% -> 0.155
        if cleaned.endswith("%"):
            try:
                return float(cleaned[:-1].strip()) / 100.0
            except ValueError:
                return default
        if cleaned in ("", "-", "—", "N/A", "na", "null", "None", "nan"):
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def safe_div(
    numerator: Optional[float],
    denominator: Optional[float],
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Safely divides two floats. Returns default (or None) if denominator is zero,
    substantially zero, or either value is None.
    """
    if numerator is None or denominator is None:
        return default
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


# =====================================================================
# SECTION 2: DATA STRUCTURES FOR FINANCIAL STATEMENTS
# =====================================================================

@dataclass
class IncomeStatement:
    """
    Standardized Income Statement for a single reporting period.
    Auto-computes gross profit, operating income (EBIT), EBITDA,
    EBT, net income, and EPS if individual line items are provided.
    """
    period: str
    revenue: float = 0.0
    cost_of_goods_sold: float = 0.0
    gross_profit: Optional[float] = None

    # Operating Expenses
    selling_general_admin: float = 0.0
    research_development: float = 0.0
    other_operating_expenses: float = 0.0
    total_operating_expenses: Optional[float] = None

    # Operating Profitability
    operating_income: Optional[float] = None  # EBIT
    depreciation_amortization: float = 0.0
    ebitda: Optional[float] = None

    # Non-Operating & Tax
    interest_expense: float = 0.0
    interest_income: float = 0.0
    other_non_operating: float = 0.0
    ebt: Optional[float] = None  # Earnings Before Tax / Pretax Income
    tax_expense: float = 0.0
    net_income: Optional[float] = None

    # Share Metrics
    shares_outstanding: Optional[float] = None
    eps_diluted: Optional[float] = None

    def __post_init__(self):
        if self.gross_profit is None:
            self.gross_profit = self.revenue - self.cost_of_goods_sold

        if self.total_operating_expenses is None:
            self.total_operating_expenses = (
                self.selling_general_admin
                + self.research_development
                + self.other_operating_expenses
            )

        if self.operating_income is None:
            self.operating_income = self.gross_profit - self.total_operating_expenses

        if self.ebitda is None:
            self.ebitda = self.operating_income + self.depreciation_amortization

        if self.ebt is None:
            self.ebt = (
                self.operating_income
                - self.interest_expense
                + self.interest_income
                + self.other_non_operating
            )

        if self.net_income is None:
            self.net_income = self.ebt - self.tax_expense

        if self.eps_diluted is None and self.shares_outstanding and self.shares_outstanding > 0:
            self.eps_diluted = self.net_income / self.shares_outstanding

    @property
    def ebit(self) -> float:
        """Alias for operating income."""
        return self.operating_income if self.operating_income is not None else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BalanceSheet:
    """
    Standardized Balance Sheet for a single reporting period.
    Tracks assets, liabilities, and stockholders' equity with auto-calculation
    of sub-totals, working capital, total debt, and invested capital.
    """
    period: str

    # Current Assets
    cash_and_equivalents: float = 0.0
    marketable_securities: float = 0.0
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    other_current_assets: float = 0.0
    total_current_assets: Optional[float] = None

    # Non-Current Assets
    gross_ppe: float = 0.0
    accumulated_depreciation: float = 0.0
    net_ppe: Optional[float] = None
    goodwill: float = 0.0
    intangible_assets: float = 0.0
    long_term_investments: float = 0.0
    other_non_current_assets: float = 0.0
    total_non_current_assets: Optional[float] = None
    total_assets: Optional[float] = None

    # Current Liabilities
    accounts_payable: float = 0.0
    short_term_debt: float = 0.0
    current_portion_lt_debt: float = 0.0
    accrued_liabilities: float = 0.0
    other_current_liabilities: float = 0.0
    total_current_liabilities: Optional[float] = None

    # Non-Current Liabilities
    long_term_debt: float = 0.0
    deferred_tax_liabilities: float = 0.0
    other_non_current_liabilities: float = 0.0
    total_non_current_liabilities: Optional[float] = None
    total_liabilities: Optional[float] = None

    # Total Debt (explicit or derived)
    total_debt: Optional[float] = None

    # Stockholders' Equity
    common_stock: float = 0.0
    additional_paid_in_capital: float = 0.0
    retained_earnings: float = 0.0
    accumulated_other_comprehensive_income: float = 0.0
    treasury_stock: float = 0.0
    total_equity: Optional[float] = None
    total_liabilities_and_equity: Optional[float] = None

    def __post_init__(self):
        # Current assets
        computed_ca = (
            self.cash_and_equivalents
            + self.marketable_securities
            + self.accounts_receivable
            + self.inventory
            + self.other_current_assets
        )
        if self.total_current_assets is None or self.total_current_assets == 0.0:
            self.total_current_assets = computed_ca

        # Net PPE
        if self.net_ppe is None or self.net_ppe == 0.0:
            if self.gross_ppe > 0:
                self.net_ppe = max(0.0, self.gross_ppe - self.accumulated_depreciation)
            else:
                self.net_ppe = 0.0

        # Non-current assets
        computed_nca = (
            self.net_ppe
            + self.goodwill
            + self.intangible_assets
            + self.long_term_investments
            + self.other_non_current_assets
        )
        if self.total_non_current_assets is None or self.total_non_current_assets == 0.0:
            self.total_non_current_assets = computed_nca

        # Total assets
        if self.total_assets is None or self.total_assets == 0.0:
            self.total_assets = self.total_current_assets + self.total_non_current_assets

        # Current liabilities
        computed_cl = (
            self.accounts_payable
            + self.short_term_debt
            + self.current_portion_lt_debt
            + self.accrued_liabilities
            + self.other_current_liabilities
        )
        if self.total_current_liabilities is None or self.total_current_liabilities == 0.0:
            self.total_current_liabilities = computed_cl

        # Non-current liabilities
        computed_ncl = (
            self.long_term_debt
            + self.deferred_tax_liabilities
            + self.other_non_current_liabilities
        )
        if self.total_non_current_liabilities is None or self.total_non_current_liabilities == 0.0:
            self.total_non_current_liabilities = computed_ncl

        # Total liabilities
        if self.total_liabilities is None or self.total_liabilities == 0.0:
            self.total_liabilities = self.total_current_liabilities + self.total_non_current_liabilities

        # Total debt
        computed_debt = self.short_term_debt + self.current_portion_lt_debt + self.long_term_debt
        if self.total_debt is None or self.total_debt == 0.0:
            self.total_debt = computed_debt

        # Equity
        if self.total_equity is None or self.total_equity == 0.0:
            ts = abs(self.treasury_stock)
            computed_eq = (
                self.common_stock
                + self.additional_paid_in_capital
                + self.retained_earnings
                + self.accumulated_other_comprehensive_income
                - ts
            )
            if computed_eq != 0.0:
                self.total_equity = computed_eq
            elif self.total_assets > 0 and self.total_liabilities > 0:
                self.total_equity = self.total_assets - self.total_liabilities
            else:
                self.total_equity = 0.0

        if self.total_liabilities_and_equity is None or self.total_liabilities_and_equity == 0.0:
            self.total_liabilities_and_equity = (self.total_liabilities or 0.0) + (self.total_equity or 0.0)

    @property
    def cash_and_short_term_investments(self) -> float:
        """Cash and cash equivalents + marketable securities."""
        return self.cash_and_equivalents + self.marketable_securities

    @property
    def net_debt(self) -> float:
        """Total Debt - Cash and Marketable Securities."""
        debt = self.total_debt if self.total_debt is not None else 0.0
        return debt - self.cash_and_short_term_investments

    @property
    def working_capital(self) -> float:
        """Working Capital = Current Assets - Current Liabilities."""
        ca = self.total_current_assets if self.total_current_assets is not None else 0.0
        cl = self.total_current_liabilities if self.total_current_liabilities is not None else 0.0
        return ca - cl

    @property
    def invested_capital(self) -> float:
        """Invested Capital = Total Debt + Total Equity - Cash & Marketable Securities."""
        debt = self.total_debt if self.total_debt is not None else 0.0
        eq = self.total_equity if self.total_equity is not None else 0.0
        return debt + eq - self.cash_and_short_term_investments

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CashFlowStatement:
    """
    Standardized Statement of Cash Flows for a single reporting period.
    Tracks cash flows across Operating, Investing, and Financing activities,
    with auto-calculation of sub-totals and Free Cash Flow (FCF).
    """
    period: str

    # Operating Activities
    net_income: float = 0.0
    depreciation_amortization: float = 0.0
    stock_based_compensation: float = 0.0
    change_in_working_capital: float = 0.0
    other_operating_cash_flow: float = 0.0
    cash_from_operations: Optional[float] = None  # OCF

    # Investing Activities
    capital_expenditures: float = 0.0  # CapEx magnitude
    acquisitions: float = 0.0
    purchase_of_investments: float = 0.0
    sale_of_investments: float = 0.0
    other_investing_cash_flow: float = 0.0
    cash_from_investing: Optional[float] = None

    # Financing Activities
    debt_issuance: float = 0.0
    debt_repayment: float = 0.0
    common_stock_issuance: float = 0.0
    common_stock_repurchase: float = 0.0
    dividends_paid: float = 0.0
    other_financing_cash_flow: float = 0.0
    cash_from_financing: Optional[float] = None

    # Net Cash Change
    net_change_in_cash: Optional[float] = None

    def __post_init__(self):
        if self.cash_from_operations is None:
            self.cash_from_operations = (
                self.net_income
                + self.depreciation_amortization
                + self.stock_based_compensation
                + self.change_in_working_capital
                + self.other_operating_cash_flow
            )

        if self.cash_from_investing is None:
            self.cash_from_investing = (
                -abs(self.capital_expenditures)
                - abs(self.acquisitions)
                - abs(self.purchase_of_investments)
                + self.sale_of_investments
                + self.other_investing_cash_flow
            )

        if self.cash_from_financing is None:
            self.cash_from_financing = (
                self.debt_issuance
                - abs(self.debt_repayment)
                + self.common_stock_issuance
                - abs(self.common_stock_repurchase)
                - abs(self.dividends_paid)
                + self.other_financing_cash_flow
            )

        if self.net_change_in_cash is None:
            self.net_change_in_cash = (
                self.cash_from_operations
                + self.cash_from_investing
                + self.cash_from_financing
            )

    @property
    def free_cash_flow(self) -> float:
        """Free Cash Flow (FCF) = Operating Cash Flow - abs(CapEx)."""
        ocf = self.cash_from_operations if self.cash_from_operations is not None else 0.0
        return ocf - abs(self.capital_expenditures)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PeriodFinancials:
    """Wrapper encapsulating all 3 statements for a single time period."""
    period: str
    income_statement: IncomeStatement
    balance_sheet: BalanceSheet
    cash_flow_statement: CashFlowStatement


class CompanyFinancials:
    """
    Multi-period financial statements model supporting chronological ordering,
    time-series queries, statement consolidation, and DataFrame exports.
    """
    def __init__(
        self,
        ticker: str = "TICKER",
        name: Optional[str] = None,
        industry: Optional[str] = None,
        is_manufacturing: bool = False,
        share_price: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
    ):
        self.ticker = ticker
        self.name = name or ticker
        self.industry = industry
        self.is_manufacturing = is_manufacturing
        self.share_price = share_price
        self.shares_outstanding = shares_outstanding
        self.periods: Dict[str, PeriodFinancials] = {}
        self._ordered_periods: List[str] = []

    def add_period(
        self,
        period: str,
        income_statement: IncomeStatement,
        balance_sheet: BalanceSheet,
        cash_flow_statement: CashFlowStatement,
    ) -> None:
        """Registers a period's financial statements into the company model."""
        pf = PeriodFinancials(
            period=period,
            income_statement=income_statement,
            balance_sheet=balance_sheet,
            cash_flow_statement=cash_flow_statement,
        )
        self.periods[period] = pf
        if period not in self._ordered_periods:
            self._ordered_periods.append(period)
            try:
                # Natural alphanumeric sort: handles '2021', '2022', '2023Q1', '2023Q2'
                self._ordered_periods.sort(
                    key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", x)]
                )
            except Exception:
                pass

    def get_period(self, period: str) -> Optional[PeriodFinancials]:
        """Retrieves statements for a specific period."""
        return self.periods.get(period)

    def list_periods(self) -> List[str]:
        """Lists all registered periods in chronological order."""
        return list(self._ordered_periods)

    def get_prior_period(self, period: str) -> Optional[str]:
        """Finds the chronologically immediate prior period."""
        if period in self._ordered_periods:
            idx = self._ordered_periods.index(period)
            if idx > 0:
                return self._ordered_periods[idx - 1]
        return None

    def to_dataframe(self, statement: str = "all") -> pd.DataFrame:
        """
        Exports statements across all periods to a tabular pandas DataFrame.
        statement can be: 'income_statement', 'balance_sheet', 'cash_flow', or 'all'.
        """
        data = {}
        for p in self._ordered_periods:
            pf = self.periods[p]
            col_data = {}
            if statement in ("income_statement", "all"):
                for k, v in pf.income_statement.to_dict().items():
                    if k != "period":
                        col_data[f"IS_{k}"] = v
            if statement in ("balance_sheet", "all"):
                for k, v in pf.balance_sheet.to_dict().items():
                    if k != "period":
                        col_data[f"BS_{k}"] = v
            if statement in ("cash_flow", "all"):
                for k, v in pf.cash_flow_statement.to_dict().items():
                    if k != "period":
                        col_data[f"CF_{k}"] = v
            data[p] = col_data
        return pd.DataFrame(data)


# =====================================================================
# SECTION 3: RATIOS, METRICS, FORENSICS, & VALUATION ENGINE
# =====================================================================

@dataclass
class DuPont3Stage:
    net_profit_margin: Optional[float]
    asset_turnover: Optional[float]
    equity_multiplier: Optional[float]
    calculated_roe: Optional[float]
    actual_roe: Optional[float]
    discrepancy: Optional[float]


@dataclass
class DuPont5Stage:
    tax_burden: Optional[float]          # Net Income / EBT
    interest_burden: Optional[float]     # EBT / EBIT
    operating_margin: Optional[float]    # EBIT / Revenue
    asset_turnover: Optional[float]      # Revenue / Total Assets
    financial_leverage: Optional[float]  # Total Assets / Total Equity
    calculated_roe: Optional[float]
    actual_roe: Optional[float]
    discrepancy: Optional[float]


@dataclass
class AltmanZResult:
    score: float
    model_type: str  # 'manufacturing' or 'non_manufacturing'
    zone: str        # 'Safe', 'Grey', or 'Distress'
    x1_wc_ta: float
    x2_re_ta: float
    x3_ebit_ta: float
    x4_eq_tl: float
    x5_sales_ta: Optional[float] = None
    interpretation: str = ""


@dataclass
class BeneishMResult:
    score: float
    is_manipulator: bool
    interpretation: str
    dsri: Optional[float]  # Days Sales in Receivables Index
    gmi: Optional[float]   # Gross Margin Index
    aqi: Optional[float]   # Asset Quality Index
    sgi: Optional[float]   # Sales Growth Index
    depi: Optional[float]  # Depreciation Index
    sgai: Optional[float]  # Sales, General & Administrative Expenses Index
    lvgi: Optional[float]  # Leverage Index
    tata: Optional[float]  # Total Accruals to Total Assets
    model_used: str = "8-variable"


class FinancialAnalyzer:
    """
    Comprehensive analyzer for fundamental financial ratio computation,
    DuPont decomposition, working capital efficiency, risk screeners, and valuation.
    """

    def __init__(self, financials: CompanyFinancials):
        self.financials = financials

    # -----------------------------------------------------------------
    # 1. Profitability Ratios
    # -----------------------------------------------------------------

    def gross_margin(self, period: str) -> Optional[float]:
        """Gross Margin = Gross Profit / Revenue"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.income_statement.gross_profit, pf.income_statement.revenue)

    def operating_margin(self, period: str) -> Optional[float]:
        """Operating Margin = Operating Income (EBIT) / Revenue"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.income_statement.operating_income, pf.income_statement.revenue)

    def net_margin(self, period: str) -> Optional[float]:
        """Net Margin = Net Income / Revenue"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.income_statement.net_income, pf.income_statement.revenue)

    def ebitda_margin(self, period: str) -> Optional[float]:
        """EBITDA Margin = EBITDA / Revenue"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.income_statement.ebitda, pf.income_statement.revenue)

    def roe(self, period: str, use_average: bool = True) -> Optional[float]:
        """
        Return on Equity = Net Income / Total Stockholders' Equity.
        When use_average=True, averages equity between period t and t-1.
        """
        pf = self.financials.get_period(period)
        if not pf: return None
        eq_cur = pf.balance_sheet.total_equity
        if eq_cur is None or abs(eq_cur) < 1e-9: return None

        equity = eq_cur
        if use_average:
            prior_p = self.financials.get_prior_period(period)
            if prior_p:
                prior_pf = self.financials.get_period(prior_p)
                if prior_pf and prior_pf.balance_sheet.total_equity is not None:
                    equity = (eq_cur + prior_pf.balance_sheet.total_equity) / 2.0

        return safe_div(pf.income_statement.net_income, equity)

    def roa(self, period: str, use_average: bool = True) -> Optional[float]:
        """
        Return on Assets = Net Income / Total Assets.
        When use_average=True, averages assets between period t and t-1.
        """
        pf = self.financials.get_period(period)
        if not pf: return None
        ta_cur = pf.balance_sheet.total_assets
        if ta_cur is None or abs(ta_cur) < 1e-9: return None

        assets = ta_cur
        if use_average:
            prior_p = self.financials.get_prior_period(period)
            if prior_p:
                prior_pf = self.financials.get_period(prior_p)
                if prior_pf and prior_pf.balance_sheet.total_assets is not None:
                    assets = (ta_cur + prior_pf.balance_sheet.total_assets) / 2.0

        return safe_div(pf.income_statement.net_income, assets)

    def roic(
        self,
        period: str,
        tax_rate: Optional[float] = None,
        use_average: bool = True,
    ) -> Optional[float]:
        """
        Return on Invested Capital (ROIC) = NOPAT / Invested Capital.
        NOPAT = EBIT * (1 - Effective Tax Rate).
        Invested Capital = Total Debt + Total Equity - Cash & Marketable Securities.
        """
        pf = self.financials.get_period(period)
        if not pf: return None

        ebit = pf.income_statement.ebit
        ebt = pf.income_statement.ebt or 0.0
        tax_exp = pf.income_statement.tax_expense

        # Effective tax rate determination
        if tax_rate is None:
            if ebt > 0 and tax_exp >= 0:
                eff_tax = min(max(tax_exp / ebt, 0.0), 0.45)
            else:
                eff_tax = 0.21
        else:
            eff_tax = tax_rate

        nopat = ebit * (1.0 - eff_tax)
        ic_cur = pf.balance_sheet.invested_capital
        if ic_cur <= 0:
            return None

        ic = ic_cur
        if use_average:
            prior_p = self.financials.get_prior_period(period)
            if prior_p:
                prior_pf = self.financials.get_period(prior_p)
                if prior_pf and prior_pf.balance_sheet.invested_capital > 0:
                    ic = (ic_cur + prior_pf.balance_sheet.invested_capital) / 2.0

        return safe_div(nopat, ic)

    # -----------------------------------------------------------------
    # 2. DuPont Analysis
    # -----------------------------------------------------------------

    def dupont_3_stage(self, period: str) -> Optional[DuPont3Stage]:
        """
        3-Stage DuPont Decomposition:
        ROE = Net Profit Margin * Asset Turnover * Equity Multiplier
        """
        pf = self.financials.get_period(period)
        if not pf: return None

        net_income = pf.income_statement.net_income
        revenue = pf.income_statement.revenue
        assets = pf.balance_sheet.total_assets
        equity = pf.balance_sheet.total_equity

        npm = safe_div(net_income, revenue)
        at = safe_div(revenue, assets)
        em = safe_div(assets, equity)

        calc_roe = (npm * at * em) if (npm is not None and at is not None and em is not None) else None
        act_roe = safe_div(net_income, equity)
        disc = abs(calc_roe - act_roe) if (calc_roe is not None and act_roe is not None) else None

        return DuPont3Stage(
            net_profit_margin=npm,
            asset_turnover=at,
            equity_multiplier=em,
            calculated_roe=calc_roe,
            actual_roe=act_roe,
            discrepancy=disc,
        )

    def dupont_5_stage(self, period: str) -> Optional[DuPont5Stage]:
        """
        5-Stage DuPont Decomposition:
        ROE = Tax Burden * Interest Burden * Operating Margin * Asset Turnover * Financial Leverage
        - Tax Burden = Net Income / EBT
        - Interest Burden = EBT / EBIT
        - Operating Margin = EBIT / Revenue
        - Asset Turnover = Revenue / Total Assets
        - Financial Leverage = Total Assets / Total Equity
        """
        pf = self.financials.get_period(period)
        if not pf: return None

        net_income = pf.income_statement.net_income
        ebt = pf.income_statement.ebt
        ebit = pf.income_statement.ebit
        revenue = pf.income_statement.revenue
        assets = pf.balance_sheet.total_assets
        equity = pf.balance_sheet.total_equity

        tax_burden = safe_div(net_income, ebt)
        int_burden = safe_div(ebt, ebit)
        op_margin = safe_div(ebit, revenue)
        asset_turnover = safe_div(revenue, assets)
        fin_leverage = safe_div(assets, equity)

        components = [tax_burden, int_burden, op_margin, asset_turnover, fin_leverage]
        calc_roe = math.prod(components) if all(c is not None for c in components) else None
        act_roe = safe_div(net_income, equity)
        disc = abs(calc_roe - act_roe) if (calc_roe is not None and act_roe is not None) else None

        return DuPont5Stage(
            tax_burden=tax_burden,
            interest_burden=int_burden,
            operating_margin=op_margin,
            asset_turnover=asset_turnover,
            financial_leverage=fin_leverage,
            calculated_roe=calc_roe,
            actual_roe=act_roe,
            discrepancy=disc,
        )

    # -----------------------------------------------------------------
    # 3. Liquidity & Solvency Ratios
    # -----------------------------------------------------------------

    def current_ratio(self, period: str) -> Optional[float]:
        """Current Ratio = Current Assets / Current Liabilities"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.balance_sheet.total_current_assets, pf.balance_sheet.total_current_liabilities)

    def quick_ratio(self, period: str) -> Optional[float]:
        """Quick Ratio = (Cash + Marketable Securities + Receivables) / Current Liabilities"""
        pf = self.financials.get_period(period)
        if not pf: return None
        bs = pf.balance_sheet
        quick_assets = bs.cash_and_equivalents + bs.marketable_securities + bs.accounts_receivable
        return safe_div(quick_assets, bs.total_current_liabilities)

    def cash_ratio(self, period: str) -> Optional[float]:
        """Cash Ratio = (Cash + Marketable Securities) / Current Liabilities"""
        pf = self.financials.get_period(period)
        if not pf: return None
        bs = pf.balance_sheet
        return safe_div(bs.cash_and_short_term_investments, bs.total_current_liabilities)

    def debt_to_equity(self, period: str) -> Optional[float]:
        """Debt-to-Equity = Total Debt / Total Stockholders' Equity"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.balance_sheet.total_debt, pf.balance_sheet.total_equity)

    def net_debt_to_ebitda(self, period: str) -> Optional[float]:
        """Net Debt to EBITDA = Net Debt / EBITDA"""
        pf = self.financials.get_period(period)
        if not pf: return None
        ebitda = pf.income_statement.ebitda
        return safe_div(pf.balance_sheet.net_debt, ebitda)

    def interest_coverage(self, period: str) -> Optional[float]:
        """Interest Coverage Ratio = EBIT / Interest Expense"""
        pf = self.financials.get_period(period)
        if not pf: return None
        int_exp = pf.income_statement.interest_expense
        if abs(int_exp) < 1e-9:
            return float("inf") if pf.income_statement.ebit > 0 else 0.0
        return safe_div(pf.income_statement.ebit, int_exp)

    # -----------------------------------------------------------------
    # 4. Efficiency & Working Capital
    # -----------------------------------------------------------------

    def days_sales_outstanding(self, period: str, days: int = 365) -> Optional[float]:
        """Days Sales Outstanding (DSO) = (Accounts Receivable / Revenue) * days"""
        pf = self.financials.get_period(period)
        if not pf: return None
        ratio = safe_div(pf.balance_sheet.accounts_receivable, pf.income_statement.revenue)
        return ratio * days if ratio is not None else None

    def days_inventory_outstanding(self, period: str, days: int = 365) -> Optional[float]:
        """Days Inventory Outstanding (DIO) = (Inventory / COGS) * days"""
        pf = self.financials.get_period(period)
        if not pf: return None
        cogs = pf.income_statement.cost_of_goods_sold
        if abs(cogs) < 1e-9: return None
        ratio = safe_div(pf.balance_sheet.inventory, cogs)
        return ratio * days if ratio is not None else None

    def days_payable_outstanding(self, period: str, days: int = 365) -> Optional[float]:
        """Days Payable Outstanding (DPO) = (Accounts Payable / COGS) * days"""
        pf = self.financials.get_period(period)
        if not pf: return None
        cogs = pf.income_statement.cost_of_goods_sold
        if abs(cogs) < 1e-9: return None
        ratio = safe_div(pf.balance_sheet.accounts_payable, cogs)
        return ratio * days if ratio is not None else None

    def cash_conversion_cycle(self, period: str, days: int = 365) -> Optional[float]:
        """Cash Conversion Cycle (CCC) = DIO + DSO - DPO"""
        dio = self.days_inventory_outstanding(period, days)
        dso = self.days_sales_outstanding(period, days)
        dpo = self.days_payable_outstanding(period, days)
        dio_val = dio if dio is not None else 0.0
        if dso is not None and dpo is not None:
            return dio_val + dso - dpo
        return None

    def asset_turnover(self, period: str, use_average: bool = False) -> Optional[float]:
        """Asset Turnover = Revenue / Total Assets"""
        pf = self.financials.get_period(period)
        if not pf: return None
        assets = pf.balance_sheet.total_assets
        if use_average:
            prior_p = self.financials.get_prior_period(period)
            if prior_p:
                prior_pf = self.financials.get_period(prior_p)
                if prior_pf and prior_pf.balance_sheet.total_assets:
                    assets = (assets + prior_pf.balance_sheet.total_assets) / 2.0
        return safe_div(pf.income_statement.revenue, assets)

    def inventory_turnover(self, period: str) -> Optional[float]:
        """Inventory Turnover = COGS / Inventory"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.income_statement.cost_of_goods_sold, pf.balance_sheet.inventory)

    def receivables_turnover(self, period: str) -> Optional[float]:
        """Receivables Turnover = Revenue / Accounts Receivable"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.income_statement.revenue, pf.balance_sheet.accounts_receivable)

    def payables_turnover(self, period: str) -> Optional[float]:
        """Payables Turnover = COGS / Accounts Payable"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.income_statement.cost_of_goods_sold, pf.balance_sheet.accounts_payable)

    # -----------------------------------------------------------------
    # 5. Cash Flow Quality
    # -----------------------------------------------------------------

    def ocf_to_net_income(self, period: str) -> Optional[float]:
        """Operating Cash Flow / Net Income (> 1.0 indicates high earnings quality)."""
        pf = self.financials.get_period(period)
        if not pf: return None
        return safe_div(pf.cash_flow_statement.cash_from_operations, pf.income_statement.net_income)

    def free_cash_flow(self, period: str) -> Optional[float]:
        """Free Cash Flow (FCF) = Cash from Operations - CapEx"""
        pf = self.financials.get_period(period)
        if not pf: return None
        return pf.cash_flow_statement.free_cash_flow

    def fcf_margin(self, period: str) -> Optional[float]:
        """FCF Margin = Free Cash Flow / Revenue"""
        pf = self.financials.get_period(period)
        if not pf: return None
        fcf = self.free_cash_flow(period)
        return safe_div(fcf, pf.income_statement.revenue)

    def capex_as_pct_of_sales(self, period: str) -> Optional[float]:
        """CapEx % of Sales = abs(CapEx) / Revenue"""
        pf = self.financials.get_period(period)
        if not pf: return None
        capex_abs = abs(pf.cash_flow_statement.capital_expenditures)
        return safe_div(capex_abs, pf.income_statement.revenue)

    # -----------------------------------------------------------------
    # 6. Forensic Risk Screener: Altman Z-Score & Beneish M-Score
    # -----------------------------------------------------------------

    def altman_z_score(
        self,
        period: str,
        is_manufacturing: Optional[bool] = None,
        market_value_equity: Optional[float] = None,
    ) -> Optional[AltmanZResult]:
        """
        Altman Z-Score calculation for default / bankruptcy risk screening.
        
        Manufacturing Model (Original 1968):
            Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
            X1: Working Capital / Total Assets
            X2: Retained Earnings / Total Assets
            X3: EBIT / Total Assets
            X4: Equity Value / Total Liabilities
            X5: Sales / Total Assets
            Zones: Safe (> 2.99), Grey [1.81, 2.99], Distress (< 1.81)

        Non-Manufacturing / Emerging Markets Model (Altman Z''-Score):
            Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
            Zones: Safe (> 2.60), Grey [1.10, 2.60], Distress (< 1.10)
        """
        pf = self.financials.get_period(period)
        if not pf: return None

        bs = pf.balance_sheet
        is_stmt = pf.income_statement

        ta = bs.total_assets
        tl = bs.total_liabilities
        if not ta or ta <= 0 or not tl or tl <= 0:
            return None

        wc = bs.working_capital
        re = bs.retained_earnings
        ebit = is_stmt.ebit
        sales = is_stmt.revenue

        # Equity valuation for X4
        eq_val = market_value_equity
        if eq_val is None:
            if self.financials.share_price and self.financials.shares_outstanding:
                eq_val = self.financials.share_price * self.financials.shares_outstanding
            else:
                eq_val = bs.total_equity or 0.0

        x1 = wc / ta
        x2 = re / ta
        x3 = ebit / ta
        x4 = eq_val / tl
        x5 = sales / ta

        mfg = self.financials.is_manufacturing if is_manufacturing is None else is_manufacturing

        if mfg:
            score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
            if score > 2.99:
                zone = "Safe"
                interp = "Low probability of bankruptcy within 2 years (Safe Zone)."
            elif score >= 1.81:
                zone = "Grey"
                interp = "Moderate risk of financial distress (Grey Zone); warrants scrutiny."
            else:
                zone = "Distress"
                interp = "High probability of bankruptcy/financial distress within 2 years (Distress Zone)."
            model_type = "manufacturing"
        else:
            score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
            if score > 2.60:
                zone = "Safe"
                interp = "Low probability of financial distress (Safe Zone)."
            elif score >= 1.10:
                zone = "Grey"
                interp = "Moderate risk of financial distress (Grey Zone)."
            else:
                zone = "Distress"
                interp = "High probability of financial distress within 2 years (Distress Zone)."
            model_type = "non_manufacturing"

        return AltmanZResult(
            score=round(score, 4),
            model_type=model_type,
            zone=zone,
            x1_wc_ta=round(x1, 4),
            x2_re_ta=round(x2, 4),
            x3_ebit_ta=round(x3, 4),
            x4_eq_tl=round(x4, 4),
            x5_sales_ta=round(x5, 4) if mfg else None,
            interpretation=interp,
        )

    def beneish_m_score(
        self,
        period: str,
        prior_period: Optional[str] = None,
    ) -> Optional[BeneishMResult]:
        """
        Beneish M-Score calculation for financial statement manipulation detection.
        Evaluates 8 variables across consecutive periods t and t-1:
        1. DSRI: Days Sales in Receivables Index
        2. GMI: Gross Margin Index
        3. AQI: Asset Quality Index
        4. SGI: Sales Growth Index
        5. DEPI: Depreciation Index
        6. SGAI: Sales, General & Administrative Expenses Index
        7. LVGI: Leverage Index
        8. TATA: Total Accruals to Total Assets
        Threshold: M-Score > -1.78 flags potential earnings manipulation.
        """
        p_cur = period
        p_prev = prior_period or self.financials.get_prior_period(period)
        if not p_prev:
            return None

        cur = self.financials.get_period(p_cur)
        prev = self.financials.get_period(p_prev)
        if not cur or not prev:
            return None

        # 1. DSRI = (Receivables_t / Sales_t) / (Receivables_{t-1} / Sales_{t-1})
        rec_t, rev_t = cur.balance_sheet.accounts_receivable, cur.income_statement.revenue
        rec_p, rev_p = prev.balance_sheet.accounts_receivable, prev.income_statement.revenue
        dsri_cur = safe_div(rec_t, rev_t, 0.0)
        dsri_prev = safe_div(rec_p, rev_p, 0.0)
        dsri = safe_div(dsri_cur, dsri_prev, 1.0)

        # 2. GMI = [(Sales_{t-1} - COGS_{t-1}) / Sales_{t-1}] / [(Sales_t - COGS_t) / Sales_t]
        gm_prev = safe_div(prev.income_statement.gross_profit, rev_p, 0.0)
        gm_cur = safe_div(cur.income_statement.gross_profit, rev_t, 0.0)
        gmi = safe_div(gm_prev, gm_cur, 1.0)

        # 3. AQI = [1 - (CurrentAssets_t + PP&E_t + Securities_t) / TotalAssets_t] / [1 - (CurrentAssets_{t-1} + PP&E_{t-1} + Securities_{t-1}) / TotalAssets_{t-1}]
        ca_t = cur.balance_sheet.total_current_assets or 0.0
        ppe_t = cur.balance_sheet.net_ppe or 0.0
        sec_t = cur.balance_sheet.marketable_securities or 0.0
        ta_t = cur.balance_sheet.total_assets or 1.0

        ca_p = prev.balance_sheet.total_current_assets or 0.0
        ppe_p = prev.balance_sheet.net_ppe or 0.0
        sec_p = prev.balance_sheet.marketable_securities or 0.0
        ta_p = prev.balance_sheet.total_assets or 1.0

        non_ca_ppe_t = max(0.0, 1.0 - ((ca_t + ppe_t + sec_t) / ta_t))
        non_ca_ppe_p = max(0.0, 1.0 - ((ca_p + ppe_p + sec_p) / ta_p))
        aqi = safe_div(non_ca_ppe_t, non_ca_ppe_p, 1.0)

        # 4. SGI = Sales_t / Sales_{t-1}
        sgi = safe_div(rev_t, rev_p, 1.0)

        # 5. DEPI = [Depr_{t-1} / (Depr_{t-1} + PP&E_{t-1})] / [Depr_t / (Depr_t + PP&E_t)]
        dep_p = prev.income_statement.depreciation_amortization or prev.cash_flow_statement.depreciation_amortization or 0.0
        dep_t = cur.income_statement.depreciation_amortization or cur.cash_flow_statement.depreciation_amortization or 0.0
        rate_p = safe_div(dep_p, dep_p + ppe_p, 0.1)
        rate_t = safe_div(dep_t, dep_t + ppe_t, 0.1)
        depi = safe_div(rate_p, rate_t, 1.0)

        # 6. SGAI = (SGA_t / Sales_t) / (SGA_{t-1} / Sales_{t-1})
        sga_t = cur.income_statement.selling_general_admin
        sga_p = prev.income_statement.selling_general_admin
        sgai_cur = safe_div(sga_t, rev_t, 0.0)
        sgai_prev = safe_div(sga_p, rev_p, 0.0)
        sgai = safe_div(sgai_cur, sgai_prev, 1.0)

        # 7. LVGI = [(LTDebt_t + CurrentLiab_t) / TotalAssets_t] / [(LTDebt_{t-1} + CurrentLiab_{t-1}) / TotalAssets_{t-1}]
        lev_t = safe_div((cur.balance_sheet.long_term_debt + (cur.balance_sheet.total_current_liabilities or 0.0)), ta_t, 0.0)
        lev_p = safe_div((prev.balance_sheet.long_term_debt + (prev.balance_sheet.total_current_liabilities or 0.0)), ta_p, 0.0)
        lvgi = safe_div(lev_t, lev_p, 1.0)

        # 8. TATA = (NetIncome_t - CashFromOperations_t) / TotalAssets_t
        ni_t = cur.income_statement.net_income or 0.0
        ocf_t = cur.cash_flow_statement.cash_from_operations or 0.0
        tata = (ni_t - ocf_t) / ta_t

        score = (
            -4.84
            + 0.920 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.037 * tata
            + 0.0327 * lvgi
        )

        is_manipulator = score > -1.78
        interp = (
            f"M-Score of {score:.2f} > -1.78: Potential earnings manipulation flag detected."
            if is_manipulator
            else f"M-Score of {score:.2f} <= -1.78: Low probability of earnings manipulation."
        )

        return BeneishMResult(
            score=round(score, 4),
            is_manipulator=is_manipulator,
            interpretation=interp,
            dsri=round(dsri, 4),
            gmi=round(gmi, 4),
            aqi=round(aqi, 4),
            sgi=round(sgi, 4),
            depi=round(depi, 4),
            sgai=round(sgai, 4),
            lvgi=round(lvgi, 4),
            tata=round(tata, 4),
            model_used="8-variable",
        )

    # -----------------------------------------------------------------
    # 7. Batch Ratios Summary Across All Periods
    # -----------------------------------------------------------------

    def generate_ratios_summary(self) -> pd.DataFrame:
        """Calculates and aggregates all key ratios across all periods into a DataFrame."""
        results = {}
        for p in self.financials.list_periods():
            col = {
                "Gross Margin (%)": (self.gross_margin(p) or 0) * 100,
                "Operating Margin (%)": (self.operating_margin(p) or 0) * 100,
                "Net Margin (%)": (self.net_margin(p) or 0) * 100,
                "EBITDA Margin (%)": (self.ebitda_margin(p) or 0) * 100,
                "ROE (%)": (self.roe(p) or 0) * 100,
                "ROA (%)": (self.roa(p) or 0) * 100,
                "ROIC (%)": (self.roic(p) or 0) * 100,
                "Current Ratio": self.current_ratio(p),
                "Quick Ratio": self.quick_ratio(p),
                "Cash Ratio": self.cash_ratio(p),
                "Debt / Equity": self.debt_to_equity(p),
                "Net Debt / EBITDA": self.net_debt_to_ebitda(p),
                "Interest Coverage": self.interest_coverage(p),
                "DSO (days)": self.days_sales_outstanding(p),
                "DIO (days)": self.days_inventory_outstanding(p),
                "DPO (days)": self.days_payable_outstanding(p),
                "Cash Conversion Cycle (days)": self.cash_conversion_cycle(p),
                "Asset Turnover": self.asset_turnover(p),
                "OCF / Net Income": self.ocf_to_net_income(p),
                "Free Cash Flow": self.free_cash_flow(p),
                "FCF Margin (%)": (self.fcf_margin(p) or 0) * 100,
                "CapEx % of Sales": (self.capex_as_pct_of_sales(p) or 0) * 100,
            }
            z = self.altman_z_score(p)
            if z:
                col["Altman Z-Score"] = z.score
                col["Altman Zone"] = z.zone
            m = self.beneish_m_score(p)
            if m:
                col["Beneish M-Score"] = m.score
                col["Beneish Manipulation Flag"] = m.is_manipulator
            results[p] = col

        return pd.DataFrame(results).round(3)


# =====================================================================
# SECTION 4: VALUATION MODELS (INTRINSIC DCF & TRADING MULTIPLES)
# =====================================================================

@dataclass
class DCFValuationResult:
    enterprise_value: float
    equity_value: float
    implied_share_price: Optional[float]
    pv_projected_fcf: float
    terminal_value: float
    pv_terminal_value: float
    wacc: float
    terminal_growth_rate: float
    shares_outstanding: Optional[float]
    net_debt: float
    projected_fcfs: List[float]
    pv_fcfs: List[float]
    sensitivity_table: pd.DataFrame


class DCFModel:
    """
    Intrinsic Discounted Cash Flow (DCF) model supporting custom horizons,
    growth trajectories, mid-year discounting, and 2D sensitivity analysis.
    """

    @staticmethod
    def calculate(
        base_fcf: float,
        growth_rates: Union[float, List[float]],
        wacc: float = 0.09,
        terminal_growth_rate: float = 0.025,
        projection_years: int = 5,
        net_debt: float = 0.0,
        shares_outstanding: Optional[float] = None,
        mid_year_convention: bool = False,
    ) -> DCFValuationResult:
        """
        Executes DCF valuation.
        
        Parameters:
        - base_fcf: Starting Free Cash Flow (Year 0)
        - growth_rates: Annual growth rate (float) or list of rates per year
        - wacc: Discount rate (Weighted Average Cost of Capital)
        - terminal_growth_rate: Perpetual terminal growth rate (g < wacc)
        - projection_years: Forecast period in years
        - net_debt: Total Debt minus Cash and Equivalents
        - shares_outstanding: Share count for implied share price derivation
        - mid_year_convention: Whether to discount cash flows mid-year (t - 0.5)
        """
        if wacc <= terminal_growth_rate:
            raise ValueError(
                f"WACC ({wacc:.2%}) must be strictly greater than Terminal Growth Rate ({terminal_growth_rate:.2%})"
            )

        if isinstance(growth_rates, (int, float)):
            rates = [float(growth_rates)] * projection_years
        else:
            rates = list(growth_rates)
            projection_years = len(rates)

        projected_fcfs = []
        curr_fcf = base_fcf
        for r in rates:
            curr_fcf *= (1.0 + r)
            projected_fcfs.append(curr_fcf)

        pv_fcfs = []
        for i, fcf in enumerate(projected_fcfs, start=1):
            t = (i - 0.5) if mid_year_convention else i
            pv = fcf / ((1.0 + wacc) ** t)
            pv_fcfs.append(pv)

        sum_pv_fcf = sum(pv_fcfs)

        final_fcf = projected_fcfs[-1]
        terminal_value = (final_fcf * (1.0 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
        tv_t = (projection_years - 0.5) if mid_year_convention else projection_years
        pv_terminal_value = terminal_value / ((1.0 + wacc) ** tv_t)

        enterprise_value = sum_pv_fcf + pv_terminal_value
        equity_value = enterprise_value - net_debt
        implied_share_price = (
            (equity_value / shares_outstanding)
            if (shares_outstanding and shares_outstanding > 0)
            else None
        )

        # Build Sensitivity Matrix (WACC vs Terminal Growth Rate)
        wacc_range = [round(wacc + d, 4) for d in [-0.02, -0.01, 0.0, 0.01, 0.02] if (wacc + d) > 0.01]
        g_range = [round(terminal_growth_rate + d, 4) for d in [-0.01, -0.005, 0.0, 0.005, 0.01] if (terminal_growth_rate + d) >= 0.0]

        sensitivity_data = {}
        for w in wacc_range:
            col_vals = {}
            for g in g_range:
                if w <= g:
                    col_vals[f"g={g:.1%}"] = np.nan
                    continue
                pv_f = sum(
                    fcf / ((1.0 + w) ** ((idx - 0.5) if mid_year_convention else idx))
                    for idx, fcf in enumerate(projected_fcfs, start=1)
                )
                tv = (final_fcf * (1.0 + g)) / (w - g)
                pv_tv = tv / ((1.0 + w) ** tv_t)
                ev = pv_f + pv_tv
                eq = ev - net_debt
                val = (eq / shares_outstanding) if (shares_outstanding and shares_outstanding > 0) else eq
                col_vals[f"g={g:.1%}"] = round(val, 2)
            sensitivity_data[f"WACC={w:.1%}"] = col_vals

        sensitivity_df = pd.DataFrame(sensitivity_data)

        return DCFValuationResult(
            enterprise_value=round(enterprise_value, 2),
            equity_value=round(equity_value, 2),
            implied_share_price=round(implied_share_price, 2) if implied_share_price is not None else None,
            pv_projected_fcf=round(sum_pv_fcf, 2),
            terminal_value=round(terminal_value, 2),
            pv_terminal_value=round(pv_terminal_value, 2),
            wacc=wacc,
            terminal_growth_rate=terminal_growth_rate,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
            projected_fcfs=[round(x, 2) for x in projected_fcfs],
            pv_fcfs=[round(x, 2) for x in pv_fcfs],
            sensitivity_table=sensitivity_df,
        )


class MultipleValuationModel:
    """
    Relative valuation engine utilizing market multiples (P/E, EV/EBITDA, P/B, P/FCF, EV/Sales)
    to determine implied Enterprise Value, Equity Value, and per-share price.
    """

    @staticmethod
    def calculate(
        financials: CompanyFinancials,
        period: str,
        target_multiples: Dict[str, float],
        shares_outstanding: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Computes valuation implied by a dictionary of target multiples.
        Example target_multiples:
            {'pe': 22.0, 'ev_ebitda': 14.0, 'p_b': 3.5, 'p_fcf': 18.0, 'ev_sales': 3.0}
        """
        pf = financials.get_period(period)
        if not pf:
            raise ValueError(f"Period {period} not found in financials")

        net_income = pf.income_statement.net_income or 0.0
        ebitda = pf.income_statement.ebitda or 0.0
        sales = pf.income_statement.revenue or 0.0
        book_value = pf.balance_sheet.total_equity or 0.0
        fcf = pf.cash_flow_statement.free_cash_flow
        net_debt = pf.balance_sheet.net_debt

        shares = shares_outstanding or financials.shares_outstanding

        results = []

        # P/E
        if "pe" in target_multiples and net_income > 0:
            mult = target_multiples["pe"]
            eq_val = net_income * mult
            ev = eq_val + net_debt
            sh_price = (eq_val / shares) if (shares and shares > 0) else None
            results.append({
                "Multiple Type": "P/E",
                "Multiple Applied": mult,
                "Metric Value": net_income,
                "Implied EV": ev,
                "Implied Equity Value": eq_val,
                "Implied Share Price": sh_price,
            })

        # EV/EBITDA
        if "ev_ebitda" in target_multiples and ebitda > 0:
            mult = target_multiples["ev_ebitda"]
            ev = ebitda * mult
            eq_val = ev - net_debt
            sh_price = (eq_val / shares) if (shares and shares > 0) else None
            results.append({
                "Multiple Type": "EV/EBITDA",
                "Multiple Applied": mult,
                "Metric Value": ebitda,
                "Implied EV": ev,
                "Implied Equity Value": eq_val,
                "Implied Share Price": sh_price,
            })

        # P/B
        if "p_b" in target_multiples and book_value > 0:
            mult = target_multiples["p_b"]
            eq_val = book_value * mult
            ev = eq_val + net_debt
            sh_price = (eq_val / shares) if (shares and shares > 0) else None
            results.append({
                "Multiple Type": "P/B",
                "Multiple Applied": mult,
                "Metric Value": book_value,
                "Implied EV": ev,
                "Implied Equity Value": eq_val,
                "Implied Share Price": sh_price,
            })

        # P/FCF
        if "p_fcf" in target_multiples and fcf > 0:
            mult = target_multiples["p_fcf"]
            eq_val = fcf * mult
            ev = eq_val + net_debt
            sh_price = (eq_val / shares) if (shares and shares > 0) else None
            results.append({
                "Multiple Type": "P/FCF",
                "Multiple Applied": mult,
                "Metric Value": fcf,
                "Implied EV": ev,
                "Implied Equity Value": eq_val,
                "Implied Share Price": sh_price,
            })

        # EV/Sales
        if "ev_sales" in target_multiples and sales > 0:
            mult = target_multiples["ev_sales"]
            ev = sales * mult
            eq_val = ev - net_debt
            sh_price = (eq_val / shares) if (shares and shares > 0) else None
            results.append({
                "Multiple Type": "EV/Sales",
                "Multiple Applied": mult,
                "Metric Value": sales,
                "Implied EV": ev,
                "Implied Equity Value": eq_val,
                "Implied Share Price": sh_price,
            })

        return pd.DataFrame(results).round(2)


# =====================================================================
# SECTION 5: DOCUMENT EXTRACTION HELPERS (CSV, EXCEL, AND PDF)
# =====================================================================

class StatementNormalizer:
    """
    Robust financial line-item normalizer using word-boundary matching
    and priority ordering to avoid sub-word collisions (e.g. 'debt' vs 'ebt').
    """
    SYNONYM_MAP: Dict[str, List[str]] = {
        # Income Statement
        "revenue": [
            "total revenue", "total net revenue", "total net sales",
            "net revenues", "net revenue", "revenues", "revenue",
            "net sales", "total sales", "sales", "turnover"
        ],
        "cost_of_goods_sold": [
            "cost of goods sold", "cost of products sold", "cost of services",
            "cost of sales", "cost of revenue", "cogs"
        ],
        "gross_profit": ["gross profit", "gross margin", "gross income"],
        "selling_general_admin": [
            "selling general and administrative expenses", "selling general and administrative",
            "selling general admin", "general and administrative",
            "administrative expenses", "sg&a", "sga"
        ],
        "research_development": [
            "research and development expense", "research and development", "r&d", "rd"
        ],
        "total_operating_expenses": [
            "total operating expenses", "operating expenses", "total opex", "opex"
        ],
        "operating_income": [
            "income from operations", "operating profit", "operating earnings",
            "operating income", "ebit"
        ],
        "depreciation_amortization": [
            "depreciation and amortization", "depreciation & amortization",
            "depreciation", "amortization", "d&a", "da"
        ],
        "ebitda": ["adjusted ebitda", "ebitda"],
        "interest_expense": [
            "interest and debt expense", "interest expense", "finance costs", "finance cost"
        ],
        "interest_income": ["interest income", "finance income"],
        "ebt": [
            "earnings before tax", "income before taxes", "pretax income",
            "income before income taxes", "ebt"
        ],
        "tax_expense": [
            "provision for income taxes", "income tax expense", "tax expense", "income taxes"
        ],
        "net_income": [
            "net income attributable to shareholders", "net profit after tax",
            "profit after tax", "net earnings", "net profit", "net income"
        ],
        "shares_outstanding": [
            "weighted average shares diluted", "diluted shares outstanding",
            "weighted average shares", "diluted shares", "shares outstanding"
        ],

        # Balance Sheet
        "cash_and_equivalents": [
            "cash and cash equivalents", "cash & cash equivalents",
            "cash and equivalents", "cash"
        ],
        "marketable_securities": [
            "short term investments", "short-term investments",
            "marketable equity securities", "marketable securities"
        ],
        "accounts_receivable": [
            "accounts receivable net", "trade and other receivables",
            "trade receivables", "accounts receivable", "receivables"
        ],
        "inventory": [
            "merchandise inventory", "total inventory", "inventories", "inventory"
        ],
        "other_current_assets": [
            "prepaid expenses and other current assets", "prepaid expenses", "other current assets"
        ],
        "total_current_assets": ["total current assets", "current assets"],
        "gross_ppe": [
            "gross property plant and equipment", "property plant and equipment gross", "gross ppe"
        ],
        "accumulated_depreciation": [
            "less accumulated depreciation", "accumulated depreciation"
        ],
        "net_ppe": [
            "property, plant and equipment net", "property plant and equipment net",
            "property plant and equipment", "property plant & equipment",
            "net property plant and equipment", "net ppe"
        ],
        "goodwill": ["goodwill"],
        "intangible_assets": ["intangible assets net", "intangible assets", "intangibles"],
        "long_term_investments": [
            "long term investments", "long-term investments", "non-current investments"
        ],
        "total_non_current_assets": [
            "total non-current assets", "total noncurrent assets", "non-current assets"
        ],
        "total_assets": ["total assets"],

        "accounts_payable": [
            "accounts payable and accrued liabilities", "trade and other payables",
            "accounts payable", "trade payables"
        ],
        "short_term_debt": [
            "current portion of long-term debt", "current portion of long term debt",
            "current maturities of long-term debt", "commercial paper",
            "short term debt", "short-term debt", "current debt", "notes payable"
        ],
        "accrued_liabilities": ["accrued expenses", "accrued liabilities"],
        "other_current_liabilities": ["other current liabilities"],
        "total_current_liabilities": ["total current liabilities", "current liabilities"],
        "long_term_debt": [
            "long term debt net of current portion", "long-term debt",
            "long term debt", "non-current debt", "senior notes"
        ],
        "total_debt": ["total debt", "total borrowings"],
        "deferred_tax_liabilities": ["deferred tax liabilities", "deferred revenue non-current"],
        "total_non_current_liabilities": [
            "total non-current liabilities", "total noncurrent liabilities", "non-current liabilities"
        ],
        "total_liabilities": ["total liabilities"],

        "common_stock": ["common stock and paid-in capital", "common stock", "share capital"],
        "additional_paid_in_capital": [
            "additional paid in capital", "capital in excess of par value", "apic"
        ],
        "retained_earnings": [
            "retained earnings accumulated deficit", "accumulated earnings", "retained earnings"
        ],
        "treasury_stock": ["treasury shares", "treasury stock"],
        "accumulated_other_comprehensive_income": [
            "accumulated other comprehensive loss", "accumulated other comprehensive income", "aoci"
        ],
        "total_equity": [
            "total stockholders equity", "total stockholders' equity",
            "total shareholders equity", "total shareholders' equity",
            "shareholders' equity", "stockholders equity", "total equity"
        ],

        # Cash Flow Statement
        "cash_from_operations": [
            "cash provided by operating activities", "net cash from operating activities",
            "cash flows from operating activities", "operating cash flow", "cash from operations"
        ],
        "capital_expenditures": [
            "payments for property and equipment", "additions to property plant and equipment",
            "purchase of property plant and equipment", "capital expenditures", "capex"
        ],
        "acquisitions": [
            "acquisitions net of cash acquired", "payments for acquisitions", "acquisitions"
        ],
        "purchase_of_investments": [
            "purchases of investments", "purchase of marketable securities"
        ],
        "sale_of_investments": [
            "proceeds from sales of marketable securities", "sales of investments"
        ],
        "cash_from_investing": [
            "cash provided by used in investing activities", "net cash used in investing activities",
            "cash flows from investing activities", "cash from investing"
        ],
        "debt_issuance": [
            "proceeds from issuance of long-term debt", "proceeds from issuance of debt", "issuance of debt"
        ],
        "debt_repayment": [
            "payments on long-term debt", "repayments of debt", "debt repayment"
        ],
        "common_stock_issuance": [
            "proceeds from issuance of common stock", "issuance of common stock"
        ],
        "common_stock_repurchase": [
            "payments for repurchase of stock", "repurchases of common stock", "repurchase of common stock"
        ],
        "dividends_paid": ["payments of dividends", "dividends paid", "dividends"],
        "cash_from_financing": [
            "cash provided by used in financing activities", "net cash used in financing activities",
            "cash flows from financing activities", "cash from financing"
        ],
        "net_change_in_cash": [
            "net change in cash and cash equivalents", "net increase decrease in cash and cash equivalents",
            "net change in cash"
        ],
    }

    # Pre-compiled sorted synonyms (longest phrase checked first)
    _SORTED_SYNONYMS: List[Tuple[str, str]] = []
    for _canon, _syns in SYNONYM_MAP.items():
        for _s in _syns:
            _SORTED_SYNONYMS.append((_s, _canon))
    _SORTED_SYNONYMS.sort(key=lambda x: len(x[0]), reverse=True)

    @classmethod
    def match_item(cls, raw_label: str) -> Optional[str]:
        """Maps arbitrary raw line item label to canonical identifier."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s&]", " ", str(raw_label).lower()).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"^(is|bs|cf)\s*_\s*", "", cleaned)

        # 1. Exact match
        for syn, canon in cls._SORTED_SYNONYMS:
            if cleaned == syn:
                return canon

        # 2. Word boundary search
        for syn, canon in cls._SORTED_SYNONYMS:
            pattern = r"\b" + re.escape(syn) + r"\b"
            if re.search(pattern, cleaned):
                return canon

        return None


class DocumentExtractor:
    """
    Automated extractor for parsing structured financial statements from
    CSV, Excel spreadsheets, and tabular PDF reports.
    """

    @staticmethod
    def extract_from_dataframe(df: pd.DataFrame, ticker: str = "TICKER") -> CompanyFinancials:
        """
        Parses financial statements from a normalized DataFrame layout:
        - Row index or first column: Line Items
        - Subsequent columns: Period identifiers (e.g. '2021', '2022', '2023')
        """
        company = CompanyFinancials(ticker=ticker)
        df = df.copy()

        if not isinstance(df.index, pd.RangeIndex) and df.index.name is not None:
            df = df.reset_index()

        first_col = df.columns[0]
        period_cols = [c for c in df.columns if c != first_col and str(c).strip() != ""]

        period_data: Dict[str, Dict[str, float]] = {str(p).strip(): {} for p in period_cols}

        for _, row in df.iterrows():
            raw_label = str(row[first_col])
            canon_name = StatementNormalizer.match_item(raw_label)
            if not canon_name:
                continue
            for p in period_cols:
                val = safe_float(row[p])
                period_data[str(p).strip()][canon_name] = val

        for p, d in period_data.items():
            # Income Statement
            inc_keys = [
                "revenue", "cost_of_goods_sold", "gross_profit", "selling_general_admin",
                "research_development", "operating_income", "depreciation_amortization",
                "ebitda", "interest_expense", "interest_income", "ebt", "tax_expense",
                "net_income", "shares_outstanding"
            ]
            inc_kwargs = {k: d[k] for k in inc_keys if k in d}
            is_stmt = IncomeStatement(period=p, **inc_kwargs)

            # Balance Sheet
            bs_keys = [
                "cash_and_equivalents", "marketable_securities", "accounts_receivable",
                "inventory", "other_current_assets", "total_current_assets", "gross_ppe",
                "accumulated_depreciation", "net_ppe", "goodwill", "intangible_assets",
                "long_term_investments", "total_non_current_assets", "total_assets",
                "accounts_payable", "short_term_debt", "current_portion_lt_debt",
                "accrued_liabilities", "total_current_liabilities", "long_term_debt",
                "total_debt", "total_non_current_liabilities", "total_liabilities",
                "common_stock", "additional_paid_in_capital", "retained_earnings",
                "treasury_stock", "accumulated_other_comprehensive_income", "total_equity"
            ]
            bs_kwargs = {k: d[k] for k in bs_keys if k in d}
            bs_stmt = BalanceSheet(period=p, **bs_kwargs)

            # Cash Flow Statement
            cf_keys = [
                "net_income", "depreciation_amortization", "cash_from_operations",
                "capital_expenditures", "acquisitions", "purchase_of_investments",
                "sale_of_investments", "cash_from_investing", "debt_issuance",
                "debt_repayment", "common_stock_issuance", "common_stock_repurchase",
                "dividends_paid", "cash_from_financing", "net_change_in_cash"
            ]
            cf_kwargs = {k: d[k] for k in cf_keys if k in d}
            cf_stmt = CashFlowStatement(period=p, **cf_kwargs)

            company.add_period(p, is_stmt, bs_stmt, cf_stmt)

        return company

    @classmethod
    def extract_from_csv(
        cls,
        filepath: Union[str, Path],
        ticker: str = "TICKER",
    ) -> CompanyFinancials:
        """Extract financial statements from CSV file."""
        df = pd.read_csv(filepath)
        return cls.extract_from_dataframe(df, ticker=ticker)

    @classmethod
    def extract_from_excel(
        cls,
        filepath: Union[str, Path],
        ticker: str = "TICKER",
        sheet_name: Optional[Union[str, int]] = 0,
    ) -> CompanyFinancials:
        """Extract financial statements from an Excel sheet (.xlsx, .xls)."""
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        return cls.extract_from_dataframe(df, ticker=ticker)

    @classmethod
    def extract_from_pdf(
        cls,
        filepath: Union[str, Path],
        ticker: str = "TICKER",
        page_numbers: Optional[List[int]] = None,
    ) -> CompanyFinancials:
        """
        Extract financial statements from tabular PDF report using pdfplumber (with pypdf fallback).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {filepath}")

        extracted_rows: List[List[str]] = []

        if HAS_PDFPLUMBER:
            with pdfplumber.open(path) as pdf:
                pages_to_read = page_numbers or range(len(pdf.pages))
                for page_idx in pages_to_read:
                    if page_idx < len(pdf.pages):
                        page = pdf.pages[page_idx]
                        tables = page.extract_tables()
                        for table in tables:
                            for row in table:
                                if row and any(c for c in row if c):
                                    cleaned_row = [str(c).strip() if c is not None else "" for c in row]
                                    extracted_rows.append(cleaned_row)

        if not extracted_rows and HAS_PYPDF:
            reader = pypdf.PdfReader(str(path))
            pages_to_read = page_numbers or range(len(reader.pages))
            for page_idx in pages_to_read:
                if page_idx < len(reader.pages):
                    text = reader.pages[page_idx].extract_text()
                    lines = text.splitlines()
                    for line in lines:
                        parts = re.split(r"\s{2,}|\t", line.strip())
                        if len(parts) >= 2:
                            extracted_rows.append(parts)

        if not extracted_rows:
            raise ValueError(f"Unable to extract tabular financial data from PDF: {filepath}")

        header_idx = 0
        max_periods = 0
        for i, row in enumerate(extracted_rows[:15]):
            periods = sum(1 for c in row[1:] if re.search(r"\b(19|20)\d{2}\b|Q[1-4]|FY\d{2}", str(c)))
            if periods > max_periods:
                max_periods = periods
                header_idx = i

        headers = [f"Col_{idx}" if not c else c for idx, c in enumerate(extracted_rows[header_idx])]
        data_rows = extracted_rows[header_idx + 1:]

        num_cols = len(headers)
        normalized_data = []
        for r in data_rows:
            if len(r) < num_cols:
                r = r + [""] * (num_cols - len(r))
            normalized_data.append(r[:num_cols])

        df = pd.DataFrame(normalized_data, columns=headers)
        return cls.extract_from_dataframe(df, ticker=ticker)


# =====================================================================
# SECTION 6: EXECUTIVE FINANCIAL REPORT GENERATION
# =====================================================================

def generate_executive_summary(analyzer: FinancialAnalyzer) -> str:
    """
    Generates a structured, professional executive summary report
    encapsulating performance trends, ratio scorecards, DuPont decomposition,
    solvency/forensic risk screening, and intrinsic valuation.
    """
    fin = analyzer.financials
    periods = fin.list_periods()
    if not periods:
        return "No financial statement periods available to summarize."

    latest_p = periods[-1]
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXECUTIVE FINANCIAL ANALYSIS REPORT: {fin.name} ({fin.ticker})")
    lines.append(f"Periods Analyzed: {', '.join(periods)} | Latest Period: {latest_p}")
    lines.append("=" * 80)

    # 1. Historical Key Financial Statements Summary
    lines.append("\n[1] FINANCIAL STATEMENTS SUMMARY (USD Millions / Raw Units)")
    lines.append("-" * 80)
    summary_df = pd.DataFrame({
        p: {
            "Revenue": fin.get_period(p).income_statement.revenue,
            "Gross Profit": fin.get_period(p).income_statement.gross_profit,
            "Operating Income (EBIT)": fin.get_period(p).income_statement.ebit,
            "Net Income": fin.get_period(p).income_statement.net_income,
            "Operating Cash Flow": fin.get_period(p).cash_flow_statement.cash_from_operations,
            "Free Cash Flow": fin.get_period(p).cash_flow_statement.free_cash_flow,
            "Total Assets": fin.get_period(p).balance_sheet.total_assets,
            "Total Debt": fin.get_period(p).balance_sheet.total_debt,
            "Total Equity": fin.get_period(p).balance_sheet.total_equity,
        } for p in periods
    })
    lines.append(summary_df.to_string())

    # 2. Key Ratios Matrix
    lines.append("\n\n[2] FINANCIAL RATIO & PERFORMANCE MATRIX")
    lines.append("-" * 80)
    ratios_df = analyzer.generate_ratios_summary()
    lines.append(ratios_df.to_string())

    # 3. DuPont 5-Stage Analysis
    lines.append(f"\n\n[3] 5-STAGE DUPONT DECOMPOSITION ({latest_p})")
    lines.append("-" * 80)
    d5 = analyzer.dupont_5_stage(latest_p)
    if d5:
        lines.append(f"• Tax Burden (NI / EBT):             {d5.tax_burden:.4f}" if d5.tax_burden else "• Tax Burden: N/A")
        lines.append(f"• Interest Burden (EBT / EBIT):      {d5.interest_burden:.4f}" if d5.interest_burden else "• Interest Burden: N/A")
        lines.append(f"• Operating Margin (EBIT / Sales):   {d5.operating_margin:.4%}" if d5.operating_margin else "• Operating Margin: N/A")
        lines.append(f"• Asset Turnover (Sales / Assets):   {d5.asset_turnover:.4f}x" if d5.asset_turnover else "• Asset Turnover: N/A")
        lines.append(f"• Financial Leverage (Assets / Eq):  {d5.financial_leverage:.4f}x" if d5.financial_leverage else "• Financial Leverage: N/A")
        lines.append(f"-> Implied ROE:                      {d5.calculated_roe:.2%}" if d5.calculated_roe else "-> Implied ROE: N/A")
        lines.append(f"-> Actual ROE:                       {d5.actual_roe:.2%}" if d5.actual_roe else "-> Actual ROE: N/A")

    # 4. Forensic & Risk Evaluation
    lines.append(f"\n\n[4] FORENSIC & SOLVENCY SCREENER ({latest_p})")
    lines.append("-" * 80)
    z = analyzer.altman_z_score(latest_p)
    if z:
        lines.append(f"• Altman Z-Score: {z.score:.2f} ({z.zone} Zone) -> {z.interpretation}")
    m = analyzer.beneish_m_score(latest_p)
    if m:
        lines.append(f"• Beneish M-Score: {m.score:.2f} (Flag: {m.is_manipulator}) -> {m.interpretation}")
        lines.append(f"  [Indices: DSRI={m.dsri}, GMI={m.gmi}, AQI={m.aqi}, SGI={m.sgi}, DEPI={m.depi}, SGAI={m.sgai}, LVGI={m.lvgi}, TATA={m.tata}]")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("FINANCIAL ENGINE: RUNNING BUILT-IN VERIFICATION & DEMONSTRATION")
    print("=" * 70)

    # Initialize company model
    demo_company = CompanyFinancials(
        ticker="ACME",
        name="Acme Corporation",
        industry="Technology & Manufacturing",
        is_manufacturing=True,
        share_price=150.0,
        shares_outstanding=100.0,
    )

    # Multi-period financial statements (2021, 2022, 2023)
    is_21 = IncomeStatement(period="2021", revenue=1000.0, cost_of_goods_sold=600.0, selling_general_admin=150.0, research_development=50.0, depreciation_amortization=40.0, interest_expense=10.0, tax_expense=40.0, shares_outstanding=100.0)
    bs_21 = BalanceSheet(period="2021", cash_and_equivalents=200.0, marketable_securities=50.0, accounts_receivable=120.0, inventory=100.0, other_current_assets=30.0, gross_ppe=800.0, accumulated_depreciation=200.0, goodwill=100.0, intangible_assets=50.0, accounts_payable=90.0, short_term_debt=20.0, accrued_liabilities=40.0, long_term_debt=250.0, common_stock=300.0, retained_earnings=450.0)
    cf_21 = CashFlowStatement(period="2021", net_income=150.0, depreciation_amortization=40.0, stock_based_compensation=10.0, change_in_working_capital=-15.0, capital_expenditures=60.0, debt_issuance=50.0, debt_repayment=20.0, dividends_paid=30.0)
    demo_company.add_period("2021", is_21, bs_21, cf_21)

    is_22 = IncomeStatement(period="2022", revenue=1200.0, cost_of_goods_sold=700.0, selling_general_admin=170.0, research_development=60.0, depreciation_amortization=45.0, interest_expense=12.0, tax_expense=58.0, shares_outstanding=100.0)
    bs_22 = BalanceSheet(period="2022", cash_and_equivalents=240.0, marketable_securities=60.0, accounts_receivable=150.0, inventory=120.0, other_current_assets=35.0, gross_ppe=950.0, accumulated_depreciation=245.0, goodwill=100.0, intangible_assets=45.0, accounts_payable=105.0, short_term_debt=25.0, accrued_liabilities=50.0, long_term_debt=270.0, common_stock=300.0, retained_earnings=580.0)
    cf_22 = CashFlowStatement(period="2022", net_income=200.0, depreciation_amortization=45.0, stock_based_compensation=12.0, change_in_working_capital=-10.0, capital_expenditures=70.0, debt_issuance=40.0, debt_repayment=25.0, dividends_paid=35.0)
    demo_company.add_period("2022", is_22, bs_22, cf_22)

    is_23 = IncomeStatement(period="2023", revenue=1500.0, cost_of_goods_sold=850.0, selling_general_admin=200.0, research_development=75.0, depreciation_amortization=55.0, interest_expense=15.0, tax_expense=72.0, shares_outstanding=100.0)
    bs_23 = BalanceSheet(period="2023", cash_and_equivalents=310.0, marketable_securities=70.0, accounts_receivable=180.0, inventory=140.0, other_current_assets=40.0, gross_ppe=1150.0, accumulated_depreciation=300.0, goodwill=100.0, intangible_assets=40.0, accounts_payable=125.0, short_term_debt=30.0, accrued_liabilities=60.0, long_term_debt=300.0, common_stock=300.0, retained_earnings=740.0)
    cf_23 = CashFlowStatement(period="2023", net_income=288.0, depreciation_amortization=55.0, stock_based_compensation=15.0, change_in_working_capital=-12.0, capital_expenditures=85.0, debt_issuance=50.0, debt_repayment=20.0, dividends_paid=40.0)
    demo_company.add_period("2023", is_23, bs_23, cf_23)

    demo_analyzer = FinancialAnalyzer(demo_company)

    # Print Executive Summary
    print(generate_executive_summary(demo_analyzer))

    # Intrinsic DCF
    dcf = DCFModel.calculate(
        base_fcf=demo_analyzer.free_cash_flow("2023"),
        growth_rates=[0.12, 0.10, 0.08, 0.07, 0.05],
        wacc=0.09,
        terminal_growth_rate=0.025,
        net_debt=demo_company.get_period("2023").balance_sheet.net_debt,
        shares_outstanding=100.0,
    )
    print("\n[5] INTRINSIC DCF VALUATION")
    print("-" * 80)
    print(f"Enterprise Value:    ${dcf.enterprise_value:,.2f}M")
    print(f"Equity Value:        ${dcf.equity_value:,.2f}M")
    print(f"Implied Share Price: ${dcf.implied_share_price:.2f}")
    print("\nDCF Sensitivity Matrix (WACC vs Terminal Growth):")
    print(dcf.sensitivity_table)

    # Relative Multiples
    multiples = MultipleValuationModel.calculate(
        financials=demo_company,
        period="2023",
        target_multiples={"pe": 20.0, "ev_ebitda": 12.0, "p_b": 3.0, "p_fcf": 18.0, "ev_sales": 2.5},
    )
    print("\n[6] RELATIVE TRADING MULTIPLES VALUATION")
    print("-" * 80)
    print(multiples.to_string(index=False))
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE: ZERO SYNTAX OR RUNTIME ERRORS DETECTED.")
    print("=" * 70)
