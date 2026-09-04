# 📈 AlphaSight: Institutional AI Fundamental Equity Research Platform

AlphaSight is an institutional-grade fundamental equity analysis terminal built with **Streamlit** and an autonomous **multi-agent orchestration architecture**. It automates forensic accounting audits, competitive moat identification (Porter's Five Forces), capital allocation assessment, and DCF intrinsic valuation.

---

## 🌟 Key Capabilities & Features

### 1. Multi-Agent Fundamental Deliberation Architecture (`agent_orchestrator.py`)
- **🔍 Financial Auditor Agent:** Scrutinizes statement veracity, accrual quality ($OCF / NI$), revenue recognition manipulation (DSO / DSRI), expense capitalization vs. OpEx, and bankruptcy risk.
- **🏰 Strategic & Competitive Moat Agent:** Evaluates business model durability, economic moat rating (Wide, Narrow, None), Porter's Five Forces, pricing power, customer concentration, and MD&A risk trends.
- **💰 Valuation & Capital Allocation Agent:** Evaluates management stewardship (ROIC vs. WACC spread, Mauboussin capital allocation framework), share buybacks vs. dilution, dividend sustainability, and DCF intrinsic valuation.
- **🏛️ Chief Investment Officer (CIO) Synthesis Agent:** Presides over the Investment Committee, resolving trade-offs between valuation, accounting veracity, and competitive moat to formulate a definitive **Investment Memo** (Bull Case, Bear Case, Key Catalysts, Institutional Risk Matrix, and Final Verdict).

### 2. Multi-LLM Provider Support with Graceful Fallback
- **OpenAI:** GPT-4o, GPT-4o-mini, GPT-3.5-turbo.
- **Google Gemini:** Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash.
- **Anthropic:** Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku.
- **Local Ollama:** Llama 3, Mistral, DeepSeek-R1, Qwen 2.5.
- **Demonstration / Offline Mode:** High-fidelity simulated fundamental intelligence requiring zero API keys, providing immediate institutional deliberations grounded in the target company's financials.

### 3. Quantitative Financial Engine (`financial_engine.py`)
- **Institutional DuPont Decomposition:**
  - **3-Step:** $\text{ROE} = \text{Net Profit Margin} \times \text{Asset Turnover} \times \text{Equity Multiplier}$
  - **5-Step Extended:** $\text{ROE} = \text{Operating Margin} \times \text{Asset Turnover} \times \text{Leverage} \times \text{Tax Burden} \times \text{Interest Burden}$
- **Altman Z-Score Risk Gauge:**
  - Full 5-component decomposition ($X_1$ to $X_5$) with risk zones:
    - Safe Zone ($Z > 2.99$)
    - Gray Zone ($1.81 \le Z \le 2.99$)
    - Distress Zone ($Z < 1.81$)
- **Working Capital & Cash Conversion Cycle (CCC):**
  - Days Sales Outstanding (DSO), Days Sales of Inventory (DSI), Days Payable Outstanding (DPO), and CCC ($DSO + DSI - DPO$).
- **Interactive Discounted Cash Flow (DCF) & Sensitivity:**
  - Live 5-year discrete projection with Gordon Growth Terminal Value.
  - Interactive sliders for revenue growth, WACC, and terminal growth.
  - 2D Sensitivity Matrix (WACC vs. Terminal Growth Rate table).
- **Report Export:**
  - Instant download of the full Executive Investment Memo in **Markdown** and styled **HTML**.

---

## 🚀 Quickstart & Local Setup

### Prerequisites
- Python 3.10+ installed
- Git

### 1. Clone & Enter Directory
```bash
cd /working_dir/c_6859547c18b432c4
```

### 2. Create and Activate Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

A production-ready `Dockerfile` is included.

### 1. Build the Docker Image
```bash
docker build -t alphasight-equity:latest .
```

### 2. Run the Container
```bash
docker run -d -p 8501:8501 --name alphasight-app alphasight-equity:latest
```
Access the application at `http://localhost:8501`.

### 3. Stop the Container
```bash
docker stop alphasight-app
```

---

## ☁️ Cloud Deployment Instructions

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Under **Advanced settings**, configure environment variables or API keys if desired (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`).
6. Click **Deploy!**

### Hugging Face Spaces (Docker or Streamlit SDK)
1. Create a new Space on Hugging Face.
2. Select **Streamlit** or **Docker** as the SDK.
3. Push the repository files to the Space repository.
4. The Space will automatically build and serve the application.

---

## 📁 Repository Structure

```
.
├── agent_orchestrator.py   # Multi-agent prompt architecture and LLM orchestration
├── app.py                  # Streamlit web interface and interactive dashboard
├── financial_engine.py     # Quantitative financial calculations, DCF, and parsers
├── st_shim.py              # Streamlit stub for headless verification and test environments
├── requirements.txt        # Production dependencies
├── Dockerfile              # Containerization definition
└── README.md               # Architecture documentation and deployment guides
```

---

## 🧪 Testing and Verification

To verify that all quantitative models, financial parsers, multi-agent prompts, and report generation execute without errors:
```bash
python3 verify_system.py
```
All modules run self-contained unit verification ensuring zero runtime syntax errors.
