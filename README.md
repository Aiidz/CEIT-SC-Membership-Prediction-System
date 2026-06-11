# CEIT-SC Membership Prediction System

A decision support tool that uses Multiple Linear Regression to predict CEIT Student Council membership fee collections per program per semester. Built with Python, scikit-learn, statsmodels, and Streamlit.

**Live App:** [aiidz-ceit-sc-membership-prediction-system-app-ephhl0.streamlit.app](https://aiidz-ceit-sc-membership-prediction-system-app-ephhl0.streamlit.app/)

## Overview

The student council at Cavite State University (CEIT) collects membership fees from students across different programs. Planning budgets, events, and benefits requires forecasting how many students will pay each semester. This system trains a regression model on historical enrollment and collection data, then lets officers interactively predict outcomes by adjusting inputs.

The model uses 6 input features — program population, payment ratio, semester, benefits claimed, officer count, and events held — to predict paid membership counts. Statistical significance testing is included to identify which variables actually matter.

## Features

- Multiple Linear Regression via scikit-learn, with OLS statistical analysis from statsmodels
- Interactive EDA dashboard with correlation heatmaps, scatter plots, and trend charts
- Data explorer with filters and CSV export
- Predictor playground where you adjust inputs and see real-time predictions
- Dark theme UI

## Project Structure

```
ceitsc-membership-prediction/
├── .streamlit/config.toml        # Streamlit theme config
├── data/
│   ├── ceitsc_raw.csv            # Raw enrollment records (2021–2025)
│   └── ceitsc_cleaned.csv        # Preprocessed dataset
├── models/
│   └── ceitsc_model.pkl          # Trained MLR model
├── src/
│   ├── preprocess.py             # Data cleaning pipeline
│   └── train.py                  # Model training and OLS analysis
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the repo

```bash
git clone https://github.com/Aiidz/CEIT-SC-Membership-Prediction-System.git
cd CEIT-SC-Membership-Prediction-System
```

### 2. Set up virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running Locally

### Preprocess the data

```bash
python src/preprocess.py
```

### Train the model

```bash
python src/train.py
```

### Launch the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Dashboard Sections

| Tab | What it does |
|-----|-------------|
| Overview | Project objectives, research questions, IPO diagram, and current model performance |
| Data Explorer | Filterable table view of the dataset with basic statistics and CSV download |
| EDA Dashboard | Correlation heatmap, program comparisons, population vs. payment scatter, membership trends |
| Model Results | R², MAE, RMSE metrics, coefficient table with p-values, residual plot, regression equation |
| Predictor | Interactive sliders/radios for 6 features, instant prediction output, strategy advisor |

## Model Variables

**Target:** `paid_memberships` — number of students per program who paid the council fee that semester.

**Features:**

| Variable | Description |
|----------|------------|
| X₁ `population` | Total enrolled students in the program |
| X₂ `payment_ratio` | Proportion of online payments (online / total) |
| X₃ `semester_indicator` | 1 = 1st semester, 0 = 2nd semester |
| X₄ `benefits_claimed` | Number of membership benefits redeemed |
| X₅ `officer_count` | Active student council officers for the program |
| X₆ `events_held` | Total council events held that semester |

**Regression equation:**

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_3 + \beta_4X_4 + \beta_5X_5 + \beta_6X_6 + \epsilon$$

## Requirements

- streamlit >= 1.30.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.2.0
- statsmodels >= 0.14.0

## License

Developed by the CPEN70 CPES Group (Cavite State University — Main Campus) for academic purposes.
