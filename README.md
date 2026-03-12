# 🏢 AI Company Intelligence Engine

A **production-quality AI analytics platform** that analyses companies and predicts growth potential, hiring expansion, innovation capability, and market risk using machine learning.

---

## 📌 Project Overview

The AI Company Intelligence Engine ingests company profiles (synthetic by default) and applies a suite of Scikit-learn machine learning models to produce four key predictions:

| Score | Description |
|-------|-------------|
| **Growth Potential Score** | How likely is the company to grow revenue and market share? (0–100) |
| **Innovation Capability Index** | How innovation-driven is the company? (0–100) |
| **Market Risk Level** | How exposed is the company to adverse market conditions? (0–100) |
| **Hiring Expansion Probability** | What is the probability the company will expand its headcount? (0–1) |

These four signals are combined into a single **Company Intelligence Index** that surfaces the most promising companies for further analysis.

---

## ✨ Features

- 🤖 **Four independent ML models** — GradientBoosting & RandomForest pipelines with MinMax scaling
- 📊 **Interactive Streamlit dashboard** with live sidebar filters
- 📈 **Plotly visualisations** — scatter plots, bar charts, pie charts, radar charts, histograms
- 🏆 **Company Ranking** table & chart
- 🔬 **Company Deep Analysis** view with score radar
- 🏭 **Industry Comparison** benchmarked against sector averages
- 🎛️ **Real-time filters** by industry, size, market growth, and competition level
- ♻️ **100% synthetic data** — reproducible, no external API required

---

## 🏗️ Architecture

```
AI-Company-Intelligence-Engine/
├── app.py                      # Streamlit entry point
├── config.py                   # Global constants & configuration
├── requirements.txt
│
├── data/
│   ├── company_generator.py    # Synthetic company profile generation
│   └── industry_dataset.py     # Industry benchmark reference data
│
├── models/
│   ├── growth_predictor.py     # GradientBoosting growth model
│   ├── hiring_predictor.py     # RandomForest hiring expansion model
│   ├── innovation_model.py     # GradientBoosting innovation model
│   └── risk_model.py           # GradientBoosting risk model
│
├── analytics/
│   ├── company_scoring.py      # Orchestrates all models → Intelligence Index
│   └── market_analysis.py      # Aggregations: global, industry, ranking
│
├── ui/
│   ├── dashboard.py            # Streamlit section renderers
│   ├── charts.py               # Plotly chart builders
│   └── controls.py             # Sidebar widget definitions
│
└── utils/
    └── helpers.py              # Shared utility functions
```

**Data flow:**

```
company_generator.py
        │
        ▼
company_scoring.py ──► growth_predictor
                   ──► innovation_model
                   ──► hiring_predictor   (uses growth_score)
                   ──► risk_model
                   ──► intelligence_index (weighted combination)
        │
        ▼
market_analysis.py ──► global_summary | industry_summary | top_companies
        │
        ▼
    dashboard.py  ──► charts.py
        │
        ▼
      app.py  (Streamlit + controls.py)
```

---

## 🛠️ Tech Stack

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.10+ | Core language |
| Streamlit | ≥1.32 | Interactive dashboard framework |
| Pandas | ≥2.0 | Tabular data manipulation |
| NumPy | ≥1.26 | Numerical computations |
| Scikit-learn | ≥1.4 | ML model training & prediction |
| Plotly | ≥5.20 | Interactive visualisations |

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ravigohel142996/AI-Company-Intelligence-Engine.git
cd AI-Company-Intelligence-Engine
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

---

## ☁️ Deploy on Streamlit Cloud

1. Push your repository to GitHub (public or private)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repository
4. Set **Main file path** to `app.py`
5. Click **Deploy**

Streamlit Cloud will automatically install packages from `requirements.txt`.

---

## 🧠 ML Model Details

### Growth Predictor
- **Algorithm:** Gradient Boosting Regressor (100 estimators, depth 4)
- **Features:** `revenue`, `market_growth`, `rd_spending`, `employee_count`
- **Output:** Growth Score 0–100

### Hiring Expansion Model
- **Algorithm:** Random Forest Classifier → probability calibration
- **Features:** `revenue`, `growth_score`, `funding_stage`, `employee_count`
- **Output:** Hiring Probability 0–1

### Innovation Model
- **Algorithm:** Gradient Boosting Regressor (100 estimators, depth 4)
- **Features:** `rd_spending`, `product_launch_frequency`, `industry`, `competition_level`
- **Output:** Innovation Score 0–100

### Risk Model
- **Algorithm:** Gradient Boosting Regressor (100 estimators, depth 4)
- **Features:** `competition_level`, `profit_margin`, `market_growth`, `funding_stage`
- **Output:** Risk Score 0–100 (higher = riskier)

### Intelligence Index
Weighted combination of all four scores:

```
Intelligence Index =
  0.35 × growth_score
+ 0.30 × innovation_score
+ 0.20 × hiring_probability × 100
− 0.15 × risk_score
```

---

## 🔮 Future Improvements

- [ ] **Real API integration** — pull live company data from Crunchbase, LinkedIn, or similar
- [ ] **Model persistence** — serialise trained models with `joblib` to avoid re-training on every load
- [ ] **Time-series forecasting** — predict score trajectories over 12/24 month horizons
- [ ] **Explainability layer** — SHAP values for feature importance per company
- [ ] **PDF report export** — generate downloadable analysis reports
- [ ] **Database backend** — PostgreSQL / SQLite for persistent storage
- [ ] **Authentication** — multi-user support with role-based access
- [ ] **CI/CD pipeline** — automated testing and deployment via GitHub Actions

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

*Built with ❤️ using Streamlit · Scikit-learn · Plotly*
