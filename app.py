import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from src.preprocess import preprocess_data
from src.train import train_model

# Page configuration
st.set_page_config(
    page_title="CEIT-SC Membership Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Look and Google Fonts
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphism style container */
.glass-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}

.gradient-text {
    background: linear-gradient(90deg, #FF4B4B, #FF8F8F, #4B8FFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.metric-box {
    background: rgba(75, 143, 255, 0.08);
    border-left: 5px solid #4B8FFF;
    padding: 15px;
    border-radius: 4px;
}

.metric-box-success {
    background: rgba(46, 204, 113, 0.08);
    border-left: 5px solid #2ECC71;
    padding: 15px;
    border-radius: 4px;
}

.metric-box-warning {
    background: rgba(241, 196, 15, 0.08);
    border-left: 5px solid #F1C40F;
    padding: 15px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# Define paths
RAW_DATA_PATH = "data/ceitsc_raw.csv"
CLEAN_DATA_PATH = "data/ceitsc_cleaned.csv"
MODEL_PATH = "models/ceitsc_model.pkl"

# Check if model exists
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(CLEAN_DATA_PATH)

# ==========================================
# SIDEBAR NAVIGATION & CONTROLS
# ==========================================
st.sidebar.markdown("<h2 class='gradient-text'>CEIT-SC ML System</h2>", unsafe_allow_html=True)
st.sidebar.caption("Cavite State University — Main Campus")

if model_exists:
    st.sidebar.success("● Model Status: Active")
    st.sidebar.info("Model Engine: MLR (sklearn + statsmodels)")
    
    # Reload button
    if st.sidebar.button("🗑️ Reset System & Retrain"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(CLEAN_DATA_PATH):
            os.remove(CLEAN_DATA_PATH)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
else:
    st.sidebar.warning("● Model Status: Not Initialized")
    st.sidebar.info("Upload a dataset to train the model.")

# ==========================================
# ONBOARDING SETUP WIZARD (No Model Found)
# ==========================================
if not model_exists:
    st.markdown("<h1 class='gradient-text'>Welcome to CEIT-SC Membership Prediction System</h1>", unsafe_allow_html=True)
    st.write("A predictive analytics application using Multiple Linear Regression to forecast semester membership collection counts.")
    
    st.markdown("""
    <div class='glass-container'>
        <h3>🔮 Onboarding Setup Wizard</h3>
        <p>To initialize the system, please upload the historical collection and enrollment dataset (e.g. <code>ceitsc_raw.csv</code>). 
        The system will automatically run the cleaning pipeline, train the MLR model, and generate statistical evaluation metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload raw enrollment and collection CSV file:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing data & training ML models..."):
                # Save raw upload
                os.makedirs("data", exist_ok=True)
                raw_df = pd.read_csv(uploaded_file)
                raw_df.to_csv(RAW_DATA_PATH, index=False)
                
                # Preprocess
                clean_df = preprocess_data(raw_df)
                clean_df.to_csv(CLEAN_DATA_PATH, index=False)
                
                # Train
                results = train_model(clean_df)
                
                st.balloons()
                st.success("✅ Model training completed successfully!")
                
                # Quick metrics summary
                col1, col2, col3 = st.columns(3)
                col1.metric("Dataset Rows", f"{len(clean_df)}")
                col2.metric("R² Score", f"{results['r2']:.4f}")
                col3.metric("RMSE", f"{results['rmse']:.2f}")
                
                # Explaining the p-values
                sig_vars = []
                p_values = results['ols_model'].pvalues
                for var, p in p_values.items():
                    if var != 'const' and p < 0.05:
                        sig_vars.append(var)
                        
                st.write(f"**Significance check:** Variables `{', '.join(sig_vars)}` show statistically significant influence (p < 0.05).")
                
                if st.button("Unlock Dashboard 🚀"):
                    st.rerun()
        except Exception as e:
            st.error(f"Error during initialization: {str(e)}")
            st.warning("Please ensure your CSV matches the required columns: semester, academic_year, program, population, paid_memberships, online_payments, facetf_payments, benefits_claimed, officer_count, events_held")

# ==========================================
# ACTIVE DASHBOARD (Model Found)
# ==========================================
else:
    # Load dataset & model
    @st.cache_data
    def load_data():
        return pd.read_csv(CLEAN_DATA_PATH)
        
    @st.cache_resource
    def load_model_data():
        # Load trained sklearn model
        with open(MODEL_PATH, "rb") as f:
            sklearn_model = pickle.load(f)
            
        # Re-run train to get statsmodels result dynamically for the tables
        df = pd.read_csv(CLEAN_DATA_PATH)
        results = train_model(df)
        return sklearn_model, results['ols_model'], results
        
    df = load_data()
    sklearn_model, ols_model, training_results = load_model_data()
    
    # Header
    st.markdown("<h1 class='gradient-text'>CEIT-SC Membership Prediction & Analytics</h1>", unsafe_allow_html=True)
    st.caption("A Decision Support Tool for Cavite State University - College of Engineering and Information Technology Student Council")
    
    # Dynamic tabs
    tab_home, tab_explorer, tab_eda, tab_model, tab_predict = st.tabs([
        "🏠 Home / Overview", 
        "📊 Data Explorer", 
        "📈 EDA Dashboard", 
        "🤖 Model Results", 
        "🔮 Predictor Playground"
    ])
    
    # ------------------------------------------
    # TAB 1: HOME
    # ------------------------------------------
    with tab_home:
        col_main, col_ipo = st.columns([3, 2])
        
        with col_main:
            st.markdown("### 🎓 Project Overview")
            st.write("""
            The **CEIT-SC Membership Prediction System** uses a **Multiple Linear Regression (MLR)** model to forecast the paid student council memberships for different engineering and information technology programs at Cavite State University - Main Campus. 
            By analyzing historical trends, enrollment counts, officer distribution, and student benefits, the tool provides predictive insights to aid council planning.
            """)
            
            st.markdown("### 🔍 Research Questions Addressed")
            st.markdown("""
            1. **RQ1 (Historical Profile):** What is the historical profile of CEIT degree programs over multiple semesters in terms of student population, collection modality (online vs face-to-face), and academic term?
            2. **RQ2 (Feature Significance):** Which factors (population, payment modality, benefits claimed, officer count, events held) significantly influence the collection of paid memberships?
            3. **RQ3 (Predictive Performance):** How accurate is the developed MLR model in forecasting future student council memberships?
            """)
            
            st.markdown("### 👥 CPEN70 Project Group")
            st.text("Aguilar · Bergado · Bituin · Guarin · Reyes · Sarmiento\nCavite State University (CvSU) — Main Campus")
            
        with col_ipo:
            st.markdown("### ⚙️ System Architecture (IPO)")
            st.markdown("""
            ```
            INPUT
            ├── Program population (X1)
            ├── Payment modality ratio (X2)
            ├── Semester indicator (X3)
            ├── Benefits claimed (X4)
            ├── Officer count (X5)
            └── Events held (X6)
            
            PROCESS
            ├── 1. Data preprocessing
            ├── 2. Feature scaling
            ├── 3. Scikit-learn fitting
            └── 4. Statsmodels OLS testing
            
            OUTPUT
            ├── Predicted memberships (Y)
            ├── Model performance metrics
            └── Statistical p-values
            ```
            """)
            
            # Simple Quick Stats in card
            st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
            st.write("#### 📈 Active Model Performance")
            st.metric("R² Score (Accuracy)", f"{training_results['r2']:.4f}")
            st.metric("Average Prediction Error (MAE)", f"{training_results['mae']:.1f} students")
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------
    # TAB 2: DATA EXPLORER
    # ------------------------------------------
    with tab_explorer:
        st.markdown("### 📊 Dataset Explorer")
        st.write("Examine, filter, and export the preprocessed dataset used to fit the MLR model.")
        
        # Sidebar-like filters inside the page
        col_f1, col_f2, col_f3 = st.columns(3)
        
        all_programs = sorted(list(df['program'].unique()))
        all_semesters = sorted(list(df['semester'].unique()))
        all_years = sorted(list(df['academic_year'].unique()))
        
        with col_f1:
            selected_programs = st.multiselect("Filter by Program:", all_programs, default=all_programs)
        with col_f2:
            selected_sems = st.multiselect("Filter by Semester:", all_semesters, default=all_semesters)
        with col_f3:
            selected_years = st.multiselect("Filter by Academic Year:", all_years, default=all_years)
            
        # Apply filters
        filtered_df = df[
            df['program'].isin(selected_programs) & 
            df['semester'].isin(selected_sems) &
            df['academic_year'].isin(selected_years)
        ]
        
        st.markdown(f"**Showing {len(filtered_df)} observations of {len(df)} total rows**")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Summary statistics
        st.markdown("#### 🔢 Summary Statistics")
        st.dataframe(filtered_df.describe().T, use_container_width=True)
        
        # Download button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="ceitsc_filtered_data.csv",
            mime="text/csv"
        )

    # ------------------------------------------
    # TAB 3: EDA DASHBOARD
    # ------------------------------------------
    with tab_eda:
        st.markdown("### 📈 Exploratory Data Analysis")
        st.write("Analyze relationships, correlation matrices, and historical trends within student council collection logs.")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 1. Correlation Matrix Heatmap")
            st.write("Identifies linear associations between factors and paid memberships. Values close to 1 indicate strong correlation.")
            
            # Draw correlation matrix
            fig, ax = plt.subplots(figsize=(6, 5))
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=False)
            plt.title("Correlation matrix of numeric variables")
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_chart2:
            st.markdown("#### 2. Student Population vs Paid Memberships")
            st.write("Visualizes the scaling of paid memberships compared to overall enrolled program populations.")
            
            # Scatter Plot using streamlit built-in (premium theme)
            st.scatter_chart(
                data=df,
                x='population',
                y='paid_memberships',
                color='program',
                use_container_width=True
            )
            
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            st.markdown("#### 3. Paid Memberships per Program (Average)")
            # Group by program and average paid_memberships
            avg_paid = df.groupby('program')['paid_memberships'].mean().reset_index()
            st.bar_chart(
                data=avg_paid,
                x='program',
                y='paid_memberships',
                use_container_width=True
            )
            
        with col_chart4:
            st.markdown("#### 4. Membership Trend Over Semesters")
            # Group by Academic Year + Semester
            df['period'] = df['academic_year'] + " " + df['semester']
            avg_trend = df.groupby('period')['paid_memberships'].sum().reset_index()
            # Sort chronologically (needs standard sorting)
            avg_trend = avg_trend.sort_values(by='period')
            st.line_chart(
                data=avg_trend,
                x='period',
                y='paid_memberships',
                use_container_width=True
            )

    # ------------------------------------------
    # TAB 4: MODEL RESULTS
    # ------------------------------------------
    with tab_model:
        st.markdown("### 🤖 Regression Model Parameters & Metrics")
        st.write("Evaluate the statistical fit and accuracy of the Multiple Linear Regression (MLR) equation.")
        
        # Display model formula
        st.markdown("""
        <div class='glass-container'>
            <h4>📐 Fitted Regression Model Equation</h4>
            <p style='font-size: 1.1em; font-family: monospace;'>
                Y (Predicted Paid Memberships) = {:.2f} 
                + ({:.4f} × Population) 
                + ({:.2f} × Payment Ratio) 
                + ({:.2f} × Semester Indicator) 
                + ({:.4f} × Benefits Claimed) 
                + ({:.2f} × Officer Count) 
                + ({:.2f} × Events Held)
            </p>
        </div>
        """.format(
            training_results['intercept'],
            training_results['coefficients'][0],
            training_results['coefficients'][1],
            training_results['coefficients'][2],
            training_results['coefficients'][3],
            training_results['coefficients'][4],
            training_results['coefficients'][5]
        ), unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        # Color coding R2
        r2_val = training_results['r2']
        if r2_val >= 0.85:
            col_m1.markdown(f"<div class='metric-box-success'><strong>R² Score (Target met):</strong><br><span style='font-size:1.8em; font-weight:bold;'>{r2_val:.4f}</span></div>", unsafe_allow_html=True)
        else:
            col_m1.markdown(f"<div class='metric-box-warning'><strong>R² Score (Target: ≥0.85):</strong><br><span style='font-size:1.8em; font-weight:bold;'>{r2_val:.4f}</span></div>", unsafe_allow_html=True)
            
        col_m2.markdown(f"<div class='metric-box'><strong>Mean Absolute Error (MAE):</strong><br><span style='font-size:1.8em; font-weight:bold;'>{training_results['mae']:.2f}</span></div>", unsafe_allow_html=True)
        col_m3.markdown(f"<div class='metric-box'><strong>Root Mean Squared Error (RMSE):</strong><br><span style='font-size:1.8em; font-weight:bold;'>{training_results['rmse']:.2f}</span></div>", unsafe_allow_html=True)
        
        st.write(" ")
        
        col_stats, col_residual = st.columns([3, 2])
        
        with col_stats:
            st.markdown("#### 🔬 statsmodels OLS Coefficients & Significance (RQ2)")
            st.write("This table outlines the p-values of variables. Features with **p < 0.05** are considered statistically significant predictors of council collections.")
            
            # Extract statsmodels parameters into a clean dataframe
            coef_df = pd.DataFrame({
                "Coefficient (β)": ols_model.params,
                "Standard Error": ols_model.bse,
                "t-Statistic": ols_model.tvalues,
                "p-Value": ols_model.pvalues,
                "[0.025 Conf. Interval]": ols_model.conf_int()[0],
                "[0.975 Conf. Interval]": ols_model.conf_int()[1],
            })
            
            # Highlight significant variables
            def highlight_p_value(val):
                color = 'lightgreen' if val < 0.05 else 'white'
                return f'background-color: {color}'
                
            st.dataframe(coef_df.style.map(highlight_p_value, subset=['p-Value']), use_container_width=True)
            
        with col_residual:
            st.markdown("#### 🔬 Model Residual Plot")
            st.write("A scatter plot showing actual values vs. model predictions. A tightly aligned diagonal line represents a near-perfect fit.")
            
            # Generate test set predictions for validation display
            from sklearn.model_selection import train_test_split
            X = df[training_results['features']]
            y = df['paid_memberships']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = sklearn_model.predict(X_test)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_test, y_pred, color='#4B8FFF', edgecolors='black', alpha=0.8)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Paid Memberships')
            ax.set_ylabel('Predicted Paid Memberships')
            ax.set_title('Actual vs Predicted Values')
            plt.tight_layout()
            st.pyplot(fig)

    # ------------------------------------------
    # TAB 5: PREDICTOR PLAYGROUND
    # ------------------------------------------
    with tab_predict:
        st.markdown("### 🔮 Interactive Membership Prediction Engine")
        st.write("Adjust the slider inputs representing a future semester's environment to estimate paid membership collections per program.")
        
        col_inputs, col_gauge = st.columns([1, 1])
        
        with col_inputs:
            st.markdown("#### 🎛️ Feature Predictor Inputs")
            
            # Dynamic defaults based on dataset averages
            avg_pop = int(df['population'].mean())
            avg_ratio = float(df['payment_ratio'].mean())
            avg_benefits = int(df['benefits_claimed'].mean())
            avg_officers = int(df['officer_count'].mean())
            avg_events = int(df['events_held'].mean())
            
            # Interactive widgets
            population = st.slider("Program Population ($X_1$):", min_value=10, max_value=1000, value=avg_pop, step=10)
            payment_ratio = st.slider("Online Payment Ratio ($X_2$):", min_value=0.0, max_value=1.0, value=avg_ratio, step=0.05, help="Ratio of students paying online versus in person.")
            semester = st.radio("Academic Semester ($X_3$):", options=["1st Semester", "2nd Semester"], index=0)
            benefits_claimed = st.slider("Benefits Claimed ($X_4$):", min_value=0, max_value=int(population * 0.9), value=min(avg_benefits, int(population*0.5)), step=5, help="Number of students in the program claiming benefits.")
            officer_count = st.slider("Active Program Officers ($X_5$):", min_value=0, max_value=10, value=avg_officers, step=1)
            events_held = st.slider("Org Events Held ($X_6$):", min_value=0, max_value=20, value=avg_events, step=1)
            
            # Map semester to binary
            semester_indicator = 1 if semester == "1st Semester" else 0
            
        with col_gauge:
            st.markdown("#### 🎯 Prediction Results")
            st.write(" ")
            
            # Package inputs
            input_data = pd.DataFrame([{
                'population': population,
                'payment_ratio': payment_ratio,
                'semester_indicator': semester_indicator,
                'benefits_claimed': benefits_claimed,
                'officer_count': officer_count,
                'events_held': events_held
            }])
            
            # Run prediction
            predicted_raw = sklearn_model.predict(input_data)[0]
            
            # Sanity capping (Memberships cannot exceed population, or go below 0)
            predicted_capped = min(population, max(0.0, predicted_raw))
            
            # Calculate collection rate percentage
            rate = (predicted_capped / population) * 100
            
            # Beautiful big metric displays
            st.markdown("""
            <div class='glass-container' style='text-align: center; border-left: 8px solid #FF4B4B;'>
                <p style='font-size: 1.1em; color: gray; margin-bottom: 0px;'>Forecasted Paid Memberships</p>
                <h1 style='font-size: 4em; font-weight: 800; margin-top: 5px; margin-bottom: 5px;' class='gradient-text'>{:d}</h1>
                <p style='font-size: 1.2em; font-weight: 600; color: #4B8FFF;'>Estimated Collection Rate: {:.1f}%</p>
            </div>
            """.format(int(np.round(predicted_capped)), rate), unsafe_allow_html=True)
            
            # Warning checks for logical anomalies
            if predicted_raw != predicted_capped:
                if predicted_raw < 0:
                    st.warning("⚠️ Note: The raw regression output was negative ({:.1f}), which represents extreme unpopularity. Capped to 0.".format(predicted_raw))
                elif predicted_raw > population:
                    st.warning("⚠️ Note: The raw regression output exceeded the input population ({:.1f} vs. {:d}). Capped to match enrollment population limit.".format(predicted_raw, population))
            
            # Display warning if benefits claimed exceed prediction significantly
            if benefits_claimed > predicted_capped:
                st.warning("⚠️ Consistency Alert: The number of benefits claimed ({:d}) exceeds the estimated number of paid members ({:d}). Paid membership is typically required to claim benefits.".format(benefits_claimed, int(np.round(predicted_capped))))
                
            # Tips for improving collection
            st.markdown("##### 💡 Council Collection Strategy Tips:")
            if payment_ratio < 0.5:
                st.info("💡 Increasing online payment options (e.g., GCash, PayMaya) could boost the online payment ratio, which holds a coefficient of {:.2f} in predicting paid memberships.".format(training_results['coefficients'][1]))
            if officer_count < 3:
                st.info("💡 Assigning more active officers to represent this program could improve collection outreach. Every additional officer yields an average increase of {:.2f} paid memberships.".format(training_results['coefficients'][4]))
