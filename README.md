# Predicting Delayed Hospital Discharges in Scotland Using Explainable Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189BCC)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Master's Dissertation — University of the West of Scotland**  
> MSc Information Technology with Data Analytics | School of Computing | 2026  
> **Author:** Thomas Tafafa Anthonio (B01818359)

---

## 📋 Project Overview

Delayed hospital discharges — where patients are medically fit for release but remain in hospital beds — place significant pressure on NHS Scotland's capacity and resources. This project applies **explainable machine learning** to routinely collected NHS and Social Care data to predict the number of delayed bed days across Scottish Health Boards.

Four regression models are compared:
- Linear Regression
- Random Forest
- XGBoost
- Gradient Boosting

Model predictions are interpreted using **SHAP (SHapley Additive exPlanations)** to ensure clinical explainability and transparency.

---

## 🎯 Research Questions

1. Can routinely collected NHS and social care data predict delayed hospital discharges across Scottish Health Boards?
2. Which patient-level and system-level factors are the strongest predictors of delayed bed days?
3. How do ensemble models (Random Forest, XGBoost, Gradient Boosting) compare to a baseline Linear Regression model?
4. What actionable insights can SHAP explainability analysis provide for NHS Scotland policymakers?

---

## 📂 Repository Structure

```
nhs-delayed-discharge/
│
├── data/
│   ├── raw/                        # Original, unmodified source datasets
│   │   ├── Delayed_Discharge_NHS.csv
│   │   ├── CareInspectorate_CareHomes.xls
│   │   └── MidYear_PopulationEstimates.xlsx
│   └── processed/
│       └── NHS_Discharge_Rate.csv  # Final merged & engineered dataset
│
├── notebooks/
│   └── Dissertation1.ipynb         # Full analysis notebook (EDA → ML → SHAP)
│
├── src/
│   ├── preprocessing.py            # Data cleaning & feature engineering
│   ├── eda.py                      # Exploratory data analysis functions
│   ├── models.py                   # Model training & evaluation
│   └── explainability.py           # SHAP analysis functions
│
├── results/
│   └── figures/                    # All generated plots and charts
│
├── docs/
│   └── Final_Project_Documentation.docx  # Full dissertation document
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Datasets

Three publicly available datasets were integrated for this project:

| Dataset | Source | Description |
|--------|--------|-------------|
| NHS Delayed Discharge by Health Board | [Public Health Scotland](https://www.opendata.nhs.scot/) | Monthly delayed bed days by reason and Health Board (2019–2024) |
| Care Inspectorate Care Home Data | [Care Inspectorate](https://www.careinspectorate.com/) | Active care home services and total bed capacity per Health Board |
| Mid-Year Population Estimates | [NRS Scotland](https://www.nrscotland.gov.uk/) | Annual population by Health Board, age, and sex (1981–2024) |

> **Note:** Raw data files are not included in this repository due to size and licensing constraints. Download instructions are provided in [`data/raw/README.md`](data/raw/README.md).

---

## 🧪 Methodology

### 1. Data Preprocessing
- Removed quality flag columns, duplicates, and irrelevant rows
- Pivoted delay reasons into individual feature columns
- Converted `MonthOfDelay` to datetime; extracted `Year`, `Month`, `Quarter`
- Filtered to 2019–2024 (pre/post-COVID range)
- Engineered delay-reason **rate features** (e.g., `HealthSocialRate`)
- Created a **lag feature** (`PrevMonthDelay`) as a temporal predictor
- Merged all three datasets on Health Board code and year

### 2. Feature Engineering

| Feature | Description |
|---------|-------------|
| `prevmonthdelay` | Delayed bed days in the previous month (lag feature) |
| `healthsocialrate` | Proportion of delays due to health/social care reasons |
| `patientfamilyrate` | Proportion of delays due to patient/family reasons |
| `awirate` | Proportion of delays – Code 9 AWI |
| `nonawirate` | Proportion of delays – Code 9 Non-AWI |
| `elderlyrate` | Population aged 75+ as a proportion of total |
| `carehomecapacityrate` | Care home places per elderly resident |
| `carehomeplaces` | Total active care home beds in Health Board |
| `month`, `year` | Temporal features |
| `hb_encoded` | Label-encoded Health Board identifier |

### 3. Model Training
- **Train/Test Split:** 80/20, `random_state=42`
- **Cross-Validation:** 5-fold CV R² for all models
- **Models:** Linear Regression, Random Forest, XGBoost (200 estimators), Gradient Boosting (200 estimators)

### 4. Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- CV R² (5-Fold Cross-Validated R²)

### 5. Explainability
- SHAP feature importance bar chart
- SHAP summary (beeswarm) plot
- Individual prediction explanations

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/nhs-delayed-discharge.git
cd nhs-delayed-discharge
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch the Notebook
```bash
jupyter notebook notebooks/Dissertation1.ipynb
```

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key libraries:

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `matplotlib` / `seaborn` | Data visualisation |
| `scikit-learn` | ML models, metrics, cross-validation |
| `xgboost` | XGBoost regressor |
| `shap` | Model explainability |
| `jupyter` | Interactive notebook environment |

---

## 📈 Key Results

> Full results and figures are available in the [`results/figures/`](results/figures/) directory and the dissertation document.

| Model | MAE | RMSE | R² | CV R² |
|-------|-----|------|----|-------|
| Linear Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| Gradient Boosting | — | — | — | — |

*Results to be populated after final model runs.*

### SHAP Findings
SHAP analysis identified `prevmonthdelay` and `healthsocialrate` as the most influential predictors of delayed bed days, suggesting that addressing social care bottlenecks and using recent delay trends as early-warning signals could meaningfully reduce discharge delays.

---

## 🏥 Clinical & Policy Relevance

This project demonstrates that routinely collected administrative data — with no additional data collection burden — can power predictive models accurate enough for NHS planning purposes. SHAP explainability ensures results are interpretable by non-technical stakeholders such as Health Board managers and social care planners.

---

## ⚠️ Ethical Considerations

- All data used is **publicly available and fully anonymised** at Health Board level — no individual patient data was accessed.
- The project complies with University of the West of Scotland academic integrity policies.
- Generative AI tools (Grammarly for grammar; QuillBot for paraphrasing) were used only for writing assistance. All analysis and code is original.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Thomas Tafafa Anthonio**  
MSc Information Technology with Data Analytics  
University of the West of Scotland, School of Computing  
Student ID: B01818359

---

## 🙏 Acknowledgements

Supervised by **Md Shakil Amid**, School of Computing, University of the West of Scotland.  
Special thanks to Mrs Rolanda Ayisa Cudjoe for her unwavering support throughout this degree.
