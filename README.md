# CEIT-SC Membership Prediction System

A **Machine Learning** dashboard that uses **Multiple Linear Regression (MLR)** to predict CEIT Student Council membership fee collections per program per semester. Built with **Python**, **scikit-learn**, **statsmodels**, and **Streamlit**.

The system trains a regression model on 6 input features (population, payment ratio, semester, benefits claimed, officer count, events held) to forecast paid memberships ($Y$). Student council officers can adjust inputs in the Predictor playground and see real-time predictions backed by statistical significance testing.

## Features

* **ML Regression Model:** scikit-learn `LinearRegression` trained on historical enrollment and collection data.
* **Statistical Analysis:** Statsmodels `OLS` produces p-values, coefficients, and significance metrics for each feature.
* **Interactive EDA:** Correlation heatmaps, scatter plots, trend lines, and bar charts with interactive ECharts.
* **Data Explorer:** Filterable table view of raw and cleaned datasets with CSV export.
* **Predictor Playground:** Adjust 6 input variables via sliders/radios and see instant prediction results.
* **Dark Slate + Orange Theme:** Professional dark-mode UI with glassmorphism cards, hover animations, and Material Symbols Outlined icons.

## Project Structure

```
ceitsc-membership-prediction/
│
├── .streamlit/
│   └── config.toml             # Streamlit theme config (dark slate + orange)
│
├── data/
│   ├── ceitsc_raw.csv          # Raw enrollment and collection records (2021–2025)
│   └── ceitsc_cleaned.csv      # Preprocessed and cleaned dataset
│
├── models/
│   └── ceitsc_model.pkl        # Saved trained MLR model (pickle file)
│
├── src/
│   ├── preprocess.py           # Preprocessing script (raw -> cleaned data)
│   └── train.py                # Model training, stats OLS analysis, & evaluation
│
├── app.py                      # Main Streamlit web application & interface
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation

### Prerequisites

* Python 3.10+
* Git

---

### 1. Clone the repository

```bash
git clone https://github.com/Aiidz/CEIT-SC-Membership-Prediction-System.git
cd ceitsc-membership-prediction
```

### 2. Set up the Python virtual environment

Create a virtual environment and install dependencies using `uv`:

```bash
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

## Usage

### 1. Preprocess the data
Runs the cleaning pipeline to filter missing entries, calculate features like `payment_ratio`, and map categorical indicators.
```bash
python src/preprocess.py
```

### 2. Train the ML model
Fits the scikit-learn regression model, exports the OLS statistical summary from `statsmodels` (including p-values), and saves the trained model binary.
```bash
python src/train.py
```

### 3. Start the Streamlit App
Launches the interactive dashboard locally.
```bash
uv run streamlit run app.py
```
App runs at `http://localhost:8501`.

## Dashboard Sections

The interactive Streamlit application contains the following core views:

| Tab | Description | Key Components |
|-----|-------------|----------------|
| Overview | Project overview, objectives, and metadata | Research questions, system architecture (IPO), active model performance |
| Data Explorer | Interactive tabular data view | Data filters (program, semester), basic statistics, CSV downloader |
| EDA Dashboard | Exploratory data visualizations | Correlation heatmap, program comparisons, population vs. payment scatter, trend line |
| Model Results | Evaluation metrics and OLS statistics | $R^2$, MAE, RMSE, coefficient table with p-values, residual plot, regression equation |
| Predictor | Prediction playground | Interactive sliders/radios for 6 features, predicted membership count, strategy advisor |

## Model Variables

### Dependent Variable (Y)
* `paid_memberships`: Total number of students per program who paid the CEIT-SC membership fee that semester.

### Independent Variables (X)
* $X_1$ (`population`): Total enrolled students in the program that semester.
* $X_2$ (`payment_ratio`): Proportion of online payments (online payments / total payments).
* $X_3$ (`semester_indicator`): Binary indicator (1 = First Semester, 0 = Second Semester).
* $X_4$ (`benefits_claimed`): Number of membership benefits redeemed per program per semester.
* $X_5$ (`officer_count`): Number of active student council officers assigned to that program.
* $X_6$ (`events_held`): Total student council events held that semester.

### MLR Formula
$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_3 + \beta_4X_4 + \beta_5X_5 + \beta_6X_6 + \epsilon$$

*Note: Evaluation goal requires model fit of $R^2 \ge 0.85$.*

## Requirements

* `streamlit>=1.30.0`
* `pandas>=2.0.0`
* `numpy>=1.24.0`
* `scikit-learn>=1.2.0`
* `statsmodels>=0.14.0`

## License

Developed by the CPEN70 CPES Group (Cavite State University — Main Campus) for academic and educational purposes.
