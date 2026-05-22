# CEIT-SC Membership Prediction System

A web-based data mining and visualization system that uses **Multiple Linear Regression (MLR)** to predict the number of paid CEIT Student Council (CEIT-SC) memberships per degree program per semester, based on historical enrollment and collection data. Built with **Python**, **scikit-learn**, **statsmodels**, and **Streamlit**.

This system allows student council officers to forecast student membership payments, optimize budget allocation, and make data-driven decisions before the start of each semester.

## Features

* **Data-driven Forecasting:** Uses Multiple Linear Regression to estimate paid membership counts based on population, payment methods, events, and officer counts.
* **Statistical Insights (statsmodels):** Generates p-values, confidence intervals, and significance metrics to identify which factors (e.g., benefits claimed, events, officers) truly drive membership payments.
* **Interactive EDA Dashboard:** Dynamic visualizations of trends across semesters, correlations between variables (correlation matrix heatmap), and scatter plots of population vs. memberships.
* **Data Explorer:** Tabular explorer for the raw and cleaned datasets, with support for filtering by degree program or semester and exporting/downloading data.
* **Predictive Playground:** Interactive web form where users can adjust variables (e.g., input population, expected events, online payment ratio) to see prediction results instantly.
* **Clean & Modular Structure:** Separated preprocessing, modeling, and dashboard components.

## Project Structure

```
ceitsc-membership-prediction/
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
git clone https://github.com/yourusername/ceitsc-membership-prediction.git
cd ceitsc-membership-prediction
```

### 2. Set up the Python virtual environment

Create and activate a virtual environment to isolate project packages:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
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
streamlit run app.py
```
App runs at `http://localhost:8501`.

## Dashboard Sections

The interactive Streamlit application contains the following core views:

| Page / Tab | Description | Key Components |
|------------|-------------|----------------|
| `🏠 Home` | Project overview, objectives, and metadata | Research questions, Cavite State University guidelines, target goals |
| `📊 Data Explorer` | Interactive tabular data view | Data filters (program, semester), basic statistics, CSV downloader |
| `📈 EDA Dashboard` | Exploratory data visualizations | Correlation heatmap, program comparisons, population vs. payment scatter |
| `🤖 Model Results` | Evaluation metrics and OLS statistics | $R^2$, MAE, RMSE values, Coefficient tables, p-values |
| `🔮 Predict` | Prediction playground | Interactive sliders/inputs, predicted count display, residual alerts |

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
* `matplotlib>=3.7.0`
* `seaborn>=0.12.0`

## License

Developed by the CPEN70 CPES Group (Cavite State University — Main Campus) for academic and educational purposes.
