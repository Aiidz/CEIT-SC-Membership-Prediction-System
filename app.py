import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import base64
from src.preprocess import preprocess_data
from src.train import train_model

# Page configuration
st.set_page_config(
    page_title="CEIT-SC Membership Predictor",
    page_icon="logo.jpg" if os.path.exists("logo.jpg") else "🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib and seaborn dark mode parameters
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#120E0B",
    "axes.facecolor": "#1E1712",
    "savefig.facecolor": "#120E0B",
    "text.color": "#EFEBE9",
    "axes.labelcolor": "#A69B95",
    "xtick.color": "#A69B95",
    "ytick.color": "#A69B95",
    "grid.color": "rgba(255, 143, 0, 0.05)",
    "font.family": "sans-serif"
})

# Custom Styling for Premium Look (Google Material 3 Expressive Guidelines)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* App resets and baseline */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background-color: #120E0B !important;
    color: #EFEBE9 !important;
}

.stAppHeader {
    background-color: rgba(18, 14, 11, 0.85) !important;
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(255, 143, 0, 0.05);
}

/* Custom Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #EFEBE9 !important;
}

.gradient-text {
    background: linear-gradient(135deg, #FF9100 0%, #FF3D00 50%, #FFE082 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Material 3 Expressive Containers */
.m3-hero {
    background: linear-gradient(135deg, #1E1712 0%, #15100C 100%);
    border-radius: 32px;
    border: 1px solid rgba(255, 143, 0, 0.15);
    padding: 32px;
    margin-bottom: 32px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    display: flex;
    align-items: center;
    gap: 24px;
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.m3-hero:hover {
    border-color: rgba(255, 143, 0, 0.3);
    box-shadow: 0 16px 40px rgba(255, 143, 0, 0.08);
}

.m3-card {
    background-color: #1C1612;
    border-radius: 28px;
    border: 1px solid rgba(255, 143, 0, 0.08);
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.m3-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(255, 143, 0, 0.08);
    border-color: rgba(255, 143, 0, 0.2);
}

.m3-card-tonal {
    background-color: #2C2014;
    border-radius: 28px;
    padding: 28px;
    margin-bottom: 24px;
    border: 1px solid rgba(255, 143, 0, 0.05);
}

.m3-card-outlined {
    background-color: transparent;
    border-radius: 28px;
    border: 2px solid rgba(255, 143, 0, 0.15);
    padding: 28px;
    margin-bottom: 24px;
    transition: all 0.3s ease;
}

.m3-card-outlined:hover {
    border-color: rgba(255, 143, 0, 0.3);
    background-color: rgba(255, 143, 0, 0.02);
}

/* Material Symbols Integration styles */
.material-symbols-outlined {
    font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    vertical-align: middle;
    display: inline-block;
}

/* Custom M3 Metrics Panel */
.m3-metric-card {
    background-color: #1C1612;
    border-radius: 24px;
    border: 1px solid rgba(255, 143, 0, 0.08);
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.m3-metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 143, 0, 0.18);
    box-shadow: 0 10px 24px rgba(255, 143, 0, 0.06);
}

/* Custom M3 Segmented Navigation Tabs */
div[role="tablist"] {
    background-color: #1C1612 !important;
    border-radius: 28px !important;
    padding: 8px !important;
    border: 1px solid rgba(255, 143, 0, 0.08) !important;
    gap: 8px !important;
    margin-bottom: 32px !important;
}

button[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95em !important;
    font-weight: 600 !important;
    color: #A69B95 !important;
    padding: 12px 24px !important;
    border-radius: 20px !important;
    border: none !important;
    background-color: transparent !important;
    transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    margin: 0 !important;
}

button[data-baseweb="tab"]:hover {
    color: #FFF8E1 !important;
    background-color: rgba(255, 143, 0, 0.06) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: rgba(255, 143, 0, 0.16) !important;
    color: #FFA000 !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
}

/* Hide default bottom active-bar */
button[data-baseweb="tab"][aria-selected="true"]::after {
    display: none !important;
}

/* Input Fields styling overrides */
div[data-baseweb="input"] > div, 
div[data-baseweb="select"] > div, 
div[data-baseweb="base-input"] {
    border-radius: 16px !important;
    border: 1.5px solid rgba(255, 143, 0, 0.15) !important;
    background-color: #241A12 !important;
    color: #EFEBE9 !important;
    transition: all 0.3s ease !important;
}

div[data-baseweb="input"]:focus-within > div, 
div[data-baseweb="select"]:focus-within > div {
    border-color: #FFA000 !important;
    box-shadow: 0 0 10px rgba(255, 160, 0, 0.2) !important;
}

/* Sliders */
div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #FF5722, #FFA000) !important;
    height: 10px !important;
    border-radius: 5px !important;
}

div[data-baseweb="slider"] [role="slider"] {
    background-color: #FFA000 !important;
    border: 3px solid #1C1612 !important;
    width: 24px !important;
    height: 24px !important;
    border-radius: 50% !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="slider"] [role="slider"]:hover {
    transform: scale(1.2) !important;
    box-shadow: 0 4px 12px rgba(255, 160, 0, 0.4) !important;
}

/* Radio buttons styling */
div[data-testid="stMarkdownContainer"] p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

div[data-baseweb="radio"] label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #EFEBE9 !important;
    font-weight: 500 !important;
}

div[role="radiogroup"] {
    gap: 16px !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #FF5722 0%, #FFA000 100%) !important;
    color: #120E0B !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 12px 36px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.02em !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 6px 18px rgba(255, 87, 34, 0.25) !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}

.stButton>button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 10px 24px rgba(255, 160, 0, 0.4) !important;
    background: linear-gradient(135deg, #FFA000 0%, #FFE082 100%) !important;
}

.stButton>button:active {
    transform: translateY(1px) scale(0.98) !important;
}

/* Reset button in sidebar spec */
[data-testid="stSidebar"] button {
    background: #2C1A1A !important;
    color: #FF5722 !important;
    border: 1px solid rgba(255, 87, 34, 0.3) !important;
    border-radius: 100px !important;
    padding: 8px 20px !important;
    font-size: 0.85em !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] button:hover {
    background: #FF5722 !important;
    color: #120E0B !important;
    border-color: #FF5722 !important;
}

/* Download & File Upload controls */
.stDownloadButton>button {
    background: #2C2014 !important;
    color: #FFA000 !important;
    border: 1.5px solid rgba(255, 143, 0, 0.3) !important;
    border-radius: 100px !important;
    padding: 10px 28px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stDownloadButton>button:hover {
    background-color: rgba(255, 143, 0, 0.08) !important;
    border-color: #FFA000 !important;
    color: #FFF8E1 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(255, 143, 0, 0.15) !important;
}

[data-testid="stFileUploader"] {
    background-color: #1C1612 !important;
    border: 2px dashed rgba(255, 143, 0, 0.25) !important;
    border-radius: 24px !important;
    padding: 24px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #FFA000 !important;
    background-color: rgba(255, 143, 0, 0.04) !important;
}

/* Dataframe customization */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 143, 0, 0.08) !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    background-color: #1C1612 !important;
}

/* Custom beautiful alert blocks */
.m3-alert {
    background-color: rgba(255, 143, 0, 0.08);
    border-left: 4px solid #FFA000;
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.m3-alert-warning {
    background-color: rgba(255, 61, 0, 0.08);
    border-left: 4px solid #FF3D00;
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.m3-alert-info {
    background-color: rgba(41, 121, 255, 0.08);
    border-left: 4px solid #2979FF;
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

/* Custom Table Style */
.m3-table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 0.95em;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.m3-table th {
    background-color: #2C2014;
    color: #FFA000;
    text-align: left;
    padding: 14px 16px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}

.m3-table td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255, 179, 0, 0.06);
    color: #EFEBE9;
    background-color: #1C1612;
}

.m3-table tr:last-child td {
    border-bottom: none;
}

.m3-table tr:hover td {
    background-color: #241C15;
}

[data-testid="stSidebar"] {
    background-color: #120E0B !important;
    border-right: 1px solid rgba(255, 179, 0, 0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# Helper functions for UI
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f"data:image/jpeg;base64,{encoded_string}"
    return ""

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def m3_metric_card(label, value, icon, color="#FFA000", bg_opacity=0.12):
    rgb_tuple = hex_to_rgb(color)
    return f"""
    <div class="m3-metric-card" style="border-left: 5px solid {color}; width: 100%;">
        <div style="background-color: rgba({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]}, {bg_opacity}); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            <span class="material-symbols-outlined" style="color: {color}; font-size: 22px;">{icon}</span>
        </div>
        <div style="overflow: hidden; text-align: left;">
            <div style="font-size: 0.8em; color: #A69B95; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{label}</div>
            <div style="font-size: 1.5em; font-weight: 700; color: #EFEBE9; margin-top: 2px; line-height: 1.1;">{value}</div>
        </div>
    </div>
    """

def make_html_coef_table(coef_df):
    html = '<table class="m3-table">'
    html += '<thead><tr><th>Feature Variable</th><th>Coefficient (β)</th><th>Std. Error</th><th>t-Stat</th><th>p-Value</th><th>Significance (α=0.05)</th></tr></thead>'
    html += '<tbody>'
    for idx, row in coef_df.iterrows():
        p_val = row['p-Value']
        sig_badge = ""
        if p_val < 0.05:
            sig_badge = '<span style="background-color: rgba(0, 200, 83, 0.12); color: #00E676; padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 0.85em; display: inline-block;">★ Significant</span>'
        else:
            sig_badge = '<span style="background-color: rgba(166, 155, 149, 0.12); color: #A69B95; padding: 4px 10px; border-radius: 20px; font-weight: 500; font-size: 0.85em; display: inline-block;">Not Significant</span>'
        
        feat_name = idx
        friendly_names = {
            'const': 'Constant (Intercept)',
            'population': 'Program Population (X₁)',
            'payment_ratio': 'Online Payment Ratio (X₂)',
            'semester_indicator': 'Semester Indicator (X₃)',
            'benefits_claimed': 'Benefits Claimed (X₄)',
            'officer_count': 'Active Program Officers (X₅)',
            'events_held': 'Org Events Held (X₆)'
        }
        name_display = friendly_names.get(feat_name, feat_name)
        if feat_name == 'const':
            name_display = f"<strong>{name_display}</strong>"
            
        html += f"""
        <tr>
            <td>{name_display}</td>
            <td style="font-family: monospace; font-weight: 600;">{row['Coefficient (β)']:.4f}</td>
            <td style="font-family: monospace; color: #A69B95;">{row['Standard Error']:.4f}</td>
            <td style="font-family: monospace; color: #A69B95;">{row['t-Statistic']:.2f}</td>
            <td style="font-family: monospace; font-weight: 600; color: {'#00E676' if p_val < 0.05 else '#EFEBE9'};">{p_val:.4f}</td>
            <td>{sig_badge}</td>
        </tr>
        """
    html += '</tbody></table>'
    return html

# Matplotlib high-fidelity charting helpers
def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=100)
    fig.patch.set_facecolor('#120E0B')
    ax.set_facecolor('#1E1712')
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr_cols = ['population', 'payment_ratio', 'semester_indicator', 'benefits_claimed', 'officer_count', 'events_held', 'paid_memberships']
    corr_cols = [c for c in corr_cols if c in numeric_df.columns]
    corr = numeric_df[corr_cols].corr()
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#2C1F15", "#8D3B0D", "#E65100", "#FFB300", "#FFF8E1"]
    custom_cmap = LinearSegmentedColormap.from_list("ceitsc_cmap", colors)
    
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap=custom_cmap, 
        ax=ax, 
        cbar=False,
        annot_kws={"size": 9, "weight": "bold"}
    )
    
    ax.tick_params(colors='#A69B95', labelsize=8.5)
    ticks_x = [t.get_text().replace('_', ' ').title() for t in ax.get_xticklabels()]
    ticks_y = [t.get_text().replace('_', ' ').title() for t in ax.get_yticklabels()]
    ax.set_xticklabels(ticks_x, rotation=45, ha='right')
    ax.set_yticklabels(ticks_y, rotation=0)
    
    ax.title.set_color('#EFEBE9')
    plt.title("Correlation Matrix of Numeric Features", fontsize=11, fontweight='600', pad=12)
    plt.tight_layout()
    return fig

def plot_scatter_population_memberships(df):
    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=100)
    fig.patch.set_facecolor('#120E0B')
    ax.set_facecolor('#1E1712')
    
    programs = df['program'].unique()
    color_map = {
        'BSCS': '#FF5722',
        'BSIT': '#FFB300',
        'BSCE': '#4CAF50',
        'BSEE': '#00BCD4',
        'BSAE': '#E040FB',
        'BSAB': '#9E9E9E'
    }
    
    for prog in programs:
        prog_data = df[df['program'] == prog]
        ax.scatter(
            prog_data['population'], 
            prog_data['paid_memberships'], 
            color=color_map.get(prog, '#FFA000'), 
            label=prog, 
            s=65, 
            edgecolors='#120E0B', 
            linewidths=0.8,
            alpha=0.85
        )
        
    x_line = np.linspace(df['population'].min(), df['population'].max(), 100)
    coef = np.polyfit(df['population'], df['paid_memberships'], 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x_line, poly1d_fn(x_line), color='#FFF8E1', linestyle='--', linewidth=1.5, alpha=0.6, label='Trendline')
    
    ax.tick_params(colors='#A69B95', labelsize=8.5)
    ax.xaxis.label.set_color('#A69B95')
    ax.yaxis.label.set_color('#A69B95')
    ax.title.set_color('#EFEBE9')
    
    ax.set_xlabel('Total Program Population', fontsize=9.5)
    ax.set_ylabel('Paid Memberships', fontsize=9.5)
    ax.set_title('Program Population vs. Paid Memberships', fontsize=11, fontweight='600')
    
    for spine in ax.spines.values():
        spine.set_color('rgba(255, 143, 0, 0.12)')
        
    ax.grid(True, color='rgba(255, 143, 0, 0.05)', linestyle=':')
    ax.legend(facecolor='#1E1712', edgecolor='rgba(255, 143, 0, 0.1)', labelcolor='#EFEBE9', fontsize=8.5)
    plt.tight_layout()
    return fig

def plot_avg_memberships_program(df):
    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=100)
    fig.patch.set_facecolor('#120E0B')
    ax.set_facecolor('#1E1712')
    
    avg_paid = df.groupby('program')['paid_memberships'].mean().reset_index()
    avg_paid = avg_paid.sort_values(by='paid_memberships', ascending=False)
    
    bar_colors = [
        '#FF5722', '#FF8F00', '#FFB300', '#FFC107', '#FFE082', '#FFF8E1'
    ][:len(avg_paid)]
    
    bars = ax.bar(avg_paid['program'], avg_paid['paid_memberships'], color=bar_colors, edgecolor='none', width=0.55, alpha=0.9)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='#EFEBE9', weight='bold')
                    
    ax.tick_params(colors='#A69B95', labelsize=8.5)
    ax.yaxis.label.set_color('#A69B95')
    ax.title.set_color('#EFEBE9')
    
    ax.set_ylabel('Average Paid Memberships', fontsize=9.5)
    ax.set_title('Average Paid Memberships by Degree Program', fontsize=11, fontweight='600')
    
    for spine in ax.spines.values():
        spine.set_color('rgba(255, 143, 0, 0.12)')
        
    ax.grid(True, color='rgba(255, 143, 0, 0.05)', linestyle=':', axis='y')
    plt.tight_layout()
    return fig

def plot_membership_trend(df):
    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=100)
    fig.patch.set_facecolor('#120E0B')
    ax.set_facecolor('#1E1712')
    
    df_copy = df.copy()
    df_copy['period'] = df_copy['academic_year'] + " \n" + df_copy['semester']
    avg_trend = df_copy.groupby('period')['paid_memberships'].sum().reset_index()
    avg_trend = avg_trend.sort_values(by='period')
    
    ax.plot(
        avg_trend['period'], 
        avg_trend['paid_memberships'], 
        color='#FF8F00', 
        marker='o', 
        markersize=6, 
        linewidth=2, 
        markerfacecolor='#FFE082', 
        markeredgecolor='#FF5722', 
        label='Total Memberships'
    )
    
    ax.fill_between(
        avg_trend['period'], 
        avg_trend['paid_memberships'], 
        color='#FF5722', 
        alpha=0.12
    )
    
    for x, y in zip(avg_trend['period'], avg_trend['paid_memberships']):
        ax.annotate(f'{int(y)}',
                    xy=(x, y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='#EFEBE9', weight='bold')
                    
    ax.tick_params(colors='#A69B95', labelsize=8)
    ax.yaxis.label.set_color('#A69B95')
    ax.title.set_color('#EFEBE9')
    
    ax.set_ylabel('Total Paid Memberships', fontsize=9.5)
    ax.set_title('Council Membership Trend Over Semesters', fontsize=11, fontweight='600')
    
    for spine in ax.spines.values():
        spine.set_color('rgba(255, 143, 0, 0.12)')
        
    ax.grid(True, color='rgba(255, 143, 0, 0.05)', linestyle=':')
    plt.tight_layout()
    return fig

def plot_residual_chart(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=100)
    fig.patch.set_facecolor('#120E0B')
    ax.set_facecolor('#1E1712')
    
    ax.scatter(y_test, y_pred, color='#FFC107', edgecolors='#FF5722', s=55, alpha=0.85, label='Actual Data')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#FFF8E1', linestyle='--', linewidth=2, label='Perfect Fit')
    
    ax.tick_params(colors='#A69B95', labelsize=9)
    ax.xaxis.label.set_color('#A69B95')
    ax.yaxis.label.set_color('#A69B95')
    ax.title.set_color('#EFEBE9')
    
    ax.set_xlabel('Actual Paid Memberships', fontsize=10, fontweight='500')
    ax.set_ylabel('Predicted Paid Memberships', fontsize=10, fontweight='500')
    ax.set_title('Model Residual Plot (Actual vs. Predicted)', fontsize=11, fontweight='600')
    
    for spine in ax.spines.values():
        spine.set_color('rgba(255, 143, 0, 0.15)')
        
    ax.grid(True, color='rgba(255, 143, 0, 0.06)', linestyle=':')
    ax.legend(facecolor='#1E1712', edgecolor='rgba(255, 143, 0, 0.1)', labelcolor='#EFEBE9', fontsize=9)
    plt.tight_layout()
    return fig

# Define paths
RAW_DATA_PATH = "data/ceitsc_raw.csv"
CLEAN_DATA_PATH = "data/ceitsc_cleaned.csv"
MODEL_PATH = "models/ceitsc_model.pkl"

# Check if model exists
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(CLEAN_DATA_PATH)

# ==========================================
# SIDEBAR NAVIGATION & CONTROLS
# ==========================================
if os.path.exists("logo.jpg"):
    st.sidebar.image("logo.jpg", use_container_width=True)
st.sidebar.markdown("<h2 class='gradient-text' style='font-family: Outfit; font-size: 1.55em; margin-bottom: 0px;'>CEIT-SC ML System</h2>", unsafe_allow_html=True)
st.sidebar.caption("Cavite State University — Main Campus")

if model_exists:
    st.sidebar.markdown("""
    <div style="background-color: rgba(0, 200, 83, 0.08); border-left: 4px solid #00C853; border-radius: 12px; padding: 12px; margin-bottom: 16px; margin-top: 16px;">
        <div style="font-size: 0.8em; color: #A69B95; font-weight: 600; text-transform: uppercase;">System Status</div>
        <div style="font-size: 0.95em; font-weight: 700; color: #00E676; margin-top: 2px; display: flex; align-items: center; gap: 6px;">
            <span class="material-symbols-outlined" style="font-size: 16px;">check_circle</span> Active Predictor
        </div>
    </div>
    <div style="background-color: rgba(255, 143, 0, 0.08); border-left: 4px solid #FFA000; border-radius: 12px; padding: 12px; margin-bottom: 24px;">
        <div style="font-size: 0.8em; color: #A69B95; font-weight: 600; text-transform: uppercase;">Model Engine</div>
        <div style="font-size: 0.9em; color: #FFF8E1; margin-top: 2px; font-weight: 500;">Multiple Linear Regression</div>
        <div style="font-size: 0.8em; color: #A69B95; margin-top: 2px;">sklearn + statsmodels OLS</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset button in sidebar
    if st.sidebar.button("🗑️ Reset System & Retrain"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(CLEAN_DATA_PATH):
            os.remove(CLEAN_DATA_PATH)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
else:
    st.sidebar.markdown("""
    <div style="background-color: rgba(255, 61, 0, 0.08); border-left: 4px solid #FF3D00; border-radius: 12px; padding: 12px; margin-bottom: 24px; margin-top: 16px;">
        <div style="font-size: 0.8em; color: #A69B95; font-weight: 600; text-transform: uppercase;">System Status</div>
        <div style="font-size: 0.95em; font-weight: 700; color: #FF3D00; margin-top: 2px; display: flex; align-items: center; gap: 6px;">
            <span class="material-symbols-outlined" style="font-size: 16px;">cancel</span> Uninitialized
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# ONBOARDING SETUP WIZARD (No Model Found)
# ==========================================
if not model_exists:
    logo_base64 = get_base64_image("logo.jpg")
    
    st.markdown(f"""
    <div class="m3-hero" style="margin-top: 20px;">
        <img src="{logo_base64}" style="width: 80px; height: 80px; border-radius: 20px; border: 2px solid #FFB300;" />
        <div>
            <h1 class="gradient-text" style="margin: 0; font-size: 2.2em; font-family: 'Outfit';">CEIT-SC Membership Prediction System</h1>
            <p style="margin: 6px 0 0 0; color: #A69B95; font-size: 1.05em; font-weight: 500;">
                Cavite State University College of Engineering and Information Technology Student Council
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="m3-card">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 8px; font-family: 'Outfit';">
            <span class="material-symbols-outlined" style="color: #FFA000; font-size: 28px;">auto_awesome</span>
            System Onboarding Wizard
        </h3>
        <p style="color: #EFEBE9; line-height: 1.6; margin-bottom: 0;">
            Welcome! To initialize the predictive dashboard, the system needs to process historical enrollment and collection logs. 
            Please upload a raw CSV dataset containing your council history. The backend pipeline will clean the data, calculate relative features, and fit the regression equations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select raw enrollment and collection CSV file to begin onboarding:", type=["csv"])
    
    with st.expander("📋 Review Expected CSV Dataset Structure", expanded=False):
        st.markdown("""
        <div style="font-size: 0.9em;">
        Your CSV file should contain the following headers:
        <table class="m3-table" style="margin-top: 8px;">
            <thead>
                <tr>
                    <th>Column Name</th>
                    <th>Type</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>semester</code></td>
                    <td>Text</td>
                    <td>e.g., '1st' or '2nd'</td>
                </tr>
                <tr>
                    <td><code>academic_year</code></td>
                    <td>Text</td>
                    <td>e.g., '2023-2024'</td>
                </tr>
                <tr>
                    <td><code>program</code></td>
                    <td>Text</td>
                    <td>e.g., 'BSCS', 'BSIT', 'BSCE'</td>
                </tr>
                <tr>
                    <td><code>population</code></td>
                    <td>Integer</td>
                    <td>Total enrolled students in the program</td>
                </tr>
                <tr>
                    <td><code>paid_memberships</code></td>
                    <td>Integer</td>
                    <td>Total students who paid council fee (Target Variable)</td>
                </tr>
                <tr>
                    <td><code>online_payments</code></td>
                    <td>Integer</td>
                    <td>Number of fees collected via GCash/online channels</td>
                </tr>
                <tr>
                    <td><code>facetf_payments</code></td>
                    <td>Integer</td>
                    <td>Number of fees collected face-to-face (cash)</td>
                </tr>
                <tr>
                    <td><code>benefits_claimed</code></td>
                    <td>Integer</td>
                    <td>Number of students who claimed their member shirts/benefits</td>
                </tr>
                <tr>
                    <td><code>officer_count</code></td>
                    <td>Integer</td>
                    <td>Number of active program officers in the council</td>
                </tr>
                <tr>
                    <td><code>events_held</code></td>
                    <td>Integer</td>
                    <td>Number of student events organized during the semester</td>
                </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)
        
    if uploaded_file is not None:
        try:
            with st.spinner("Processing data & training model equations..."):
                os.makedirs("data", exist_ok=True)
                raw_df = pd.read_csv(uploaded_file)
                raw_df.to_csv(RAW_DATA_PATH, index=False)
                
                clean_df = preprocess_data(raw_df)
                clean_df.to_csv(CLEAN_DATA_PATH, index=False)
                
                results = train_model(clean_df)
                
                st.balloons()
                
                st.markdown(f"""
                <div class="m3-alert" style="border-left-color: #00C853; background-color: rgba(0, 200, 83, 0.08); margin-top: 20px;">
                    <span class="material-symbols-outlined" style="color: #00C853;">check_circle</span>
                    <div>
                        <strong style="color: #00C853; font-family: 'Outfit';">Model Initialized Successfully</strong><br>
                        <span style="font-size: 0.95em; color: #EFEBE9;">The Multiple Linear Regression model has been trained on {len(clean_df)} historical records.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show results in custom metric cards
                col_on1, col_on2, col_on3 = st.columns(3)
                with col_on1:
                    st.markdown(m3_metric_card("Dataset Rows", f"{len(clean_df)}", "database", "#FFA000"), unsafe_allow_html=True)
                with col_on2:
                    st.markdown(m3_metric_card("R² Accuracy", f"{results['r2']:.4f}", "emoji_events", "#00C853"), unsafe_allow_html=True)
                with col_on3:
                    st.markdown(m3_metric_card("RMSE Loss", f"{results['rmse']:.2f}", "trending_down", "#FF5722"), unsafe_allow_html=True)
                
                sig_vars = []
                p_values = results['ols_model'].pvalues
                for var, p in p_values.items():
                    if var != 'const' and p < 0.05:
                        sig_vars.append(var)
                        
                friendly_feats = {
                    'population': 'Program Population',
                    'payment_ratio': 'Online Payment Ratio',
                    'semester_indicator': 'Semester',
                    'benefits_claimed': 'Benefits Claimed',
                    'officer_count': 'Active Officers',
                    'events_held': 'Org Events Held'
                }
                sig_display = [friendly_feats.get(v, v) for v in sig_vars]
                
                st.markdown(f"""
                <div class="m3-card" style="margin-top: 20px;">
                    <h4 style="margin-top: 0; font-family: 'Outfit'; color: #FFA000;">Statistical Significance Check</h4>
                    <p style="margin-bottom: 0;">Variables with statistically significant influence (p < 0.05):</p>
                    <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
                        {" ".join([f'<span style="background-color: rgba(0, 200, 83, 0.12); color: #00E676; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9em;">{name}</span>' for name in sig_display]) if sig_display else '<span style="color: #A69B95;">None (check sample size)</span>'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Proceed to Dashboard 🚀"):
                    st.rerun()
        except Exception as e:
            st.error(f"Error during initialization: {str(e)}")
            st.warning("Please ensure your CSV matches the required columns listed above.")

# ==========================================
# ACTIVE DASHBOARD (Model Found)
# ==========================================
else:
    @st.cache_data
    def load_data():
        return pd.read_csv(CLEAN_DATA_PATH)
        
    @st.cache_resource
    def load_model_data():
        with open(MODEL_PATH, "rb") as f:
            sklearn_model = pickle.load(f)
        df = pd.read_csv(CLEAN_DATA_PATH)
        results = train_model(df)
        return sklearn_model, results['ols_model'], results
        
    df = load_data()
    sklearn_model, ols_model, training_results = load_model_data()
    
    # Header
    logo_base64 = get_base64_image("logo.jpg")
    st.markdown(f"""
    <div class="m3-hero">
        <img src="{logo_base64}" style="width: 80px; height: 80px; border-radius: 20px; border: 2px solid #FFB300;" />
        <div>
            <h1 class="gradient-text" style="margin: 0; font-size: 2.2em; font-family: 'Outfit';">CEIT-SC Membership Analytics Portal</h1>
            <p style="margin: 4px 0 0 0; color: #A69B95; font-size: 1.02em; font-weight: 500;">
                Cavite State University College of Engineering and Information Technology Student Council
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # M3 Pill Navigation Tabs
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
        col_main, col_ipo = st.columns([13, 10])
        
        with col_main:
            st.markdown(f"""
            <div class="m3-card">
                <h3 style="margin-top: 0; color: #FFA000; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                    <span class="material-symbols-outlined">school</span>
                    Project Overview
                </h3>
                <p style="line-height: 1.6; color: #EFEBE9;">
                    The <strong>CEIT-SC Membership Prediction System</strong> is a decision support application engineered for the 
                    <strong>College of Engineering and Information Technology Student Council (CEIT-SC)</strong> at Cavite State University (CvSU) — Main Campus.
                </p>
                <p style="line-height: 1.6; color: #EFEBE9;">
                    By leveraging a Multiple Linear Regression (MLR) framework, the tool analyzes historical collection figures, student program sizes, 
                    semester variables, and officer profiles to forecast future membership fee collections. This enables the student council 
                    to plan organization budgets, event resources, and student benefits with data-backed accuracy.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="m3-card">
                <h3 style="margin-top: 0; color: #FFA000; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                    <span class="material-symbols-outlined">help_center</span>
                    Research Questions Addressed
                </h3>
                <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 16px;">
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <span style="background-color: rgba(255, 143, 0, 0.12); color: #FFA000; padding: 4px 10px; border-radius: 8px; font-weight: 700; font-size: 0.85em;">RQ1</span>
                        <div><strong>Historical Profile:</strong> What are the historical enrollment patterns, fee collection volumes, and modality preferences (cash vs online) across different CEIT courses?</div>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <span style="background-color: rgba(255, 143, 0, 0.12); color: #FFA000; padding: 4px 10px; border-radius: 8px; font-weight: 700; font-size: 0.85em;">RQ2</span>
                        <div><strong>Feature Significance:</strong> Which organizational and external factors (population, payment modality, benefits claimed, officer counts, event frequencies) significantly influence membership collections?</div>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <span style="background-color: rgba(255, 143, 0, 0.12); color: #FFA000; padding: 4px 10px; border-radius: 8px; font-weight: 700; font-size: 0.85em;">RQ3</span>
                        <div><strong>Predictive Accuracy:</strong> How accurately can the fitted multiple regression model predict collection outcomes for upcoming semesters?</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="m3-card-tonal">
                <h4 style="margin-top: 0; color: #FFF8E1; font-family: 'Outfit';">CPEN70 Course Project Group</h4>
                <p style="margin-bottom: 0; color: #A69B95; font-size: 0.95em;">
                    Aguilar · Bergado · Bituin · Guarin · Reyes · Sarmiento<br>
                    <strong>Cavite State University (CvSU) — Main Campus</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_ipo:
            st.markdown("""
            <div class="m3-card">
                <h3 style="margin-top: 0; color: #FFA000; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                    <span class="material-symbols-outlined">account_tree</span>
                    System Architecture (IPO)
                </h3>
                <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 20px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="background-color: rgba(41, 121, 255, 0.12); color: #2979FF; font-weight: 700; width: 85px; padding: 6px; border-radius: 12px; text-align: center; font-size: 0.85em;">INPUT</div>
                        <div style="font-size: 0.9em; color: #EFEBE9;">Program population (X₁), online payment ratio (X₂), semester index (X₃), benefits claimed (X₄), officer count (X₅), events held (X₆)</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="background-color: rgba(255, 143, 0, 0.12); color: #FFA000; font-weight: 700; width: 85px; padding: 6px; border-radius: 12px; text-align: center; font-size: 0.85em;">PROCESS</div>
                        <div style="font-size: 0.9em; color: #EFEBE9;">Data Cleaning & Ratio Calculation → Scikit-learn LinearRegression & Statsmodels OLS Fit</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="background-color: rgba(0, 200, 83, 0.12); color: #00E676; font-weight: 700; width: 85px; padding: 6px; border-radius: 12px; text-align: center; font-size: 0.85em;">OUTPUT</div>
                        <div style="font-size: 0.9em; color: #EFEBE9;">Predicted student memberships (Y), OLS statistical coefficients, strategy advisory insights</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="m3-card" style="border-left: 6px solid #00C853;">
                <h4 style="margin-top: 0; color: #EFEBE9; font-family: 'Outfit';">Active Model Performance</h4>
                <div style="margin-top: 16px; display: flex; flex-direction: column; gap: 12px;">
            """, unsafe_allow_html=True)
            st.markdown(m3_metric_card("Model Accuracy (R²)", f"{training_results['r2']:.4f}", "verified", "#00C853", 0.08), unsafe_allow_html=True)
            st.markdown(m3_metric_card("Average Error (MAE)", f"{training_results['mae']:.1f} students", "bar_chart", "#FFA000", 0.08), unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    # ------------------------------------------
    # TAB 2: DATA EXPLORER
    # ------------------------------------------
    with tab_explorer:
        st.markdown("""
        <div class="m3-card" style="margin-bottom: 24px;">
            <h3 style="margin-top: 0; color: #FFA000; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                <span class="material-symbols-outlined">database</span>
                Dataset Explorer & Filters
            </h3>
            <p style="color: #A69B95; margin-bottom: 16px;">Filter and inspect historical records of student population, modalities, and membership collections.</p>
        </div>
        """, unsafe_allow_html=True)
        
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
            
        filtered_df = df[
            df['program'].isin(selected_programs) & 
            df['semester'].isin(selected_sems) &
            df['academic_year'].isin(selected_years)
        ]
        
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.markdown(f"**Showing {len(filtered_df)} observations of {len(df)} total rows**")
        with col_t2:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="ceitsc_filtered_data.csv",
                mime="text/csv"
            )
            
        st.dataframe(filtered_df, use_container_width=True)
        
        st.markdown("""
        <div class="m3-card" style="margin-top: 24px;">
            <h4 style="margin-top: 0; color: #FFA000; font-family: 'Outfit';">Descriptive Statistics Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(filtered_df.describe().T, use_container_width=True)

    # ------------------------------------------
    # TAB 3: EDA DASHBOARD
    # ------------------------------------------
    with tab_eda:
        st.markdown("""
        <div class="m3-card">
            <h3 style="margin-top: 0; color: #FFA000; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                <span class="material-symbols-outlined">insights</span>
                Exploratory Data Analysis
            </h3>
            <p style="color: #A69B95; margin-bottom: 0;">Explore historical trends, correlations, and relationships within the student council membership records.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig1 = plot_correlation_heatmap(df)
            st.pyplot(fig1)
            st.caption("Correlation Heatmap: Identifies linear association strengths (close to 1.00 is high correlation) between student statistics and collections.")
            
        with col_chart2:
            fig2 = plot_scatter_population_memberships(df)
            st.pyplot(fig2)
            st.caption("Scatter Plot: Displays the relationship between student enrollment counts and paid memberships by program, along with overall linear fit.")
            
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            fig3 = plot_avg_memberships_program(df)
            st.pyplot(fig3)
            st.caption("Bar Chart: Illustrates the average number of paid student memberships generated by each department program over historical semesters.")
            
        with col_chart4:
            fig4 = plot_membership_trend(df)
            st.pyplot(fig4)
            st.caption("Line Chart: Highlights the aggregate sum of student memberships collected per academic period across the main campus.")

    # ------------------------------------------
    # TAB 4: MODEL RESULTS
    # ------------------------------------------
    with tab_model:
        st.markdown("""
        <div class="m3-card">
            <h3 style="margin-top: 0; color: #FFA000; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                <span class="material-symbols-outlined">query_stats</span>
                Regression Equation & Statistical Significance
            </h3>
            <p style="color: #A69B95; margin-bottom: 0;">Evaluate the Multiple Linear Regression (MLR) model parameters, residuals, and variable significance coefficients.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model formula
        st.markdown("""
        <div class="m3-card" style="border-left: 6px solid #FF5722;">
            <h4 style="margin-top: 0; font-family: 'Outfit'; color: #FFA000; display: flex; align-items: center; gap: 6px;">
                <span class="material-symbols-outlined">calculate</span>
                Fitted Regression Model Equation (Y)
            </h4>
            <div style="font-size: 1.15em; font-family: monospace; color: #EFEBE9; padding: 12px; background-color: #241A12; border-radius: 12px; border: 1px solid rgba(255, 143, 0, 0.1); overflow-x: auto; white-space: nowrap; margin-top: 12px;">
                Y_pred = {:.2f} 
                + ({:.4f} × Pop) 
                + ({:.2f} × PayRatio) 
                + ({:.2f} × SemInd) 
                + ({:.4f} × Benefits) 
                + ({:.2f} × Officers) 
                + ({:.2f} × Events)
            </div>
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
        
        # M3 Metric row
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            r2_val = training_results['r2']
            r2_color = "#00C853" if r2_val >= 0.85 else "#FFA000"
            st.markdown(m3_metric_card("R² Score (Model Fit)", f"{r2_val:.4f}", "verified", r2_color), unsafe_allow_html=True)
        with col_m2:
            st.markdown(m3_metric_card("Mean Absolute Error (MAE)", f"{training_results['mae']:.2f}", "error", "#FF5722"), unsafe_allow_html=True)
        with col_m3:
            st.markdown(m3_metric_card("Root Mean Squared Error (RMSE)", f"{training_results['rmse']:.2f}", "analytics", "#FFE082"), unsafe_allow_html=True)
            
        col_stats, col_residual = st.columns([13, 10])
        
        with col_stats:
            st.markdown("""
            <div style="margin-top: 16px; margin-bottom: 8px;">
                <h4 style="margin: 0; font-family: 'Outfit'; display: flex; align-items: center; gap: 6px;">
                    <span class="material-symbols-outlined" style="color: #FFA000;">table_chart</span>
                    Coefficients & Statistical Significance (RQ2)
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            coef_df = pd.DataFrame({
                "Coefficient (β)": ols_model.params,
                "Standard Error": ols_model.bse,
                "t-Statistic": ols_model.tvalues,
                "p-Value": ols_model.pvalues,
            })
            
            html_table = make_html_coef_table(coef_df)
            st.markdown(html_table, unsafe_allow_html=True)
            
        with col_residual:
            st.markdown("""
            <div style="margin-top: 16px; margin-bottom: 8px;">
                <h4 style="margin: 0; font-family: 'Outfit'; display: flex; align-items: center; gap: 6px;">
                    <span class="material-symbols-outlined" style="color: #FFA000;">scatter_plot</span>
                    Residual Analysis
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            from sklearn.model_selection import train_test_split
            X = df[training_results['features']]
            y = df['paid_memberships']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = sklearn_model.predict(X_test)
            
            fig_resid = plot_residual_chart(y_test, y_pred)
            st.pyplot(fig_resid)

    # ------------------------------------------
    # TAB 5: PREDICTOR PLAYGROUND
    # ------------------------------------------
    with tab_predict:
        col_inputs, col_gauge = st.columns([11, 10])
        
        with col_inputs:
            st.markdown("""
            <div style='margin-bottom: 16px; display: flex; align-items: center; gap: 8px;'>
                <span class="material-symbols-outlined" style="color: #FFA000; font-size: 26px;">tune</span>
                <h4 style='margin: 0; font-family: Outfit; color: #FFA000;'>Adjust Simulation Inputs</h4>
            </div>
            """, unsafe_allow_html=True)
            
            avg_pop = int(df['population'].mean())
            avg_ratio = float(df['payment_ratio'].mean())
            avg_benefits = int(df['benefits_claimed'].mean())
            avg_officers = int(df['officer_count'].mean())
            avg_events = int(df['events_held'].mean())
            
            st.markdown("""
            <div style="background-color: rgba(255,255,255,0.02); padding: 18px; border-radius: 20px; border: 1px solid rgba(255,143,0,0.05); margin-bottom: 20px;">
                <h5 style="margin-top: 0; margin-bottom: 12px; color: #FFF8E1; font-family: 'Outfit';">👥 1. Program Demographics</h5>
            """, unsafe_allow_html=True)
            population = st.slider("Enrolled Program Population ($X_1$):", min_value=10, max_value=1200, value=avg_pop, step=10)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: rgba(255,255,255,0.02); padding: 18px; border-radius: 20px; border: 1px solid rgba(255,143,0,0.05); margin-bottom: 20px;">
                <h5 style="margin-top: 0; margin-bottom: 12px; color: #FFF8E1; font-family: 'Outfit';">💳 2. Collection Modality & Calendar</h5>
            """, unsafe_allow_html=True)
            payment_ratio = st.slider("Online Payment Ratio ($X_2$):", min_value=0.0, max_value=1.0, value=avg_ratio, step=0.05, help="Proportion of payments collected via GCash/PayMaya versus Cash.")
            semester = st.radio("Academic Term ($X_3$):", options=["1st Semester (High Activity)", "2nd Semester (Low Activity)"], index=0, horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: rgba(255,255,255,0.02); padding: 18px; border-radius: 20px; border: 1px solid rgba(255,143,0,0.05);">
                <h5 style="margin-top: 0; margin-bottom: 12px; color: #FFF8E1; font-family: 'Outfit';">🎗️ 3. Engagement & Outreach Features</h5>
            """, unsafe_allow_html=True)
            benefits_claimed = st.slider("Benefits Claimed ($X_4$):", min_value=0, max_value=int(population), value=min(avg_benefits, int(population*0.6)), step=5)
            
            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                officer_count = st.slider("Active Officers ($X_5$):", min_value=0, max_value=15, value=avg_officers, step=1)
            with col_sub2:
                events_held = st.slider("Council Events ($X_6$):", min_value=0, max_value=25, value=avg_events, step=1)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_gauge:
            st.markdown("""
            <div style='margin-bottom: 16px; display: flex; align-items: center; gap: 8px;'>
                <span class="material-symbols-outlined" style="color: #FFA000; font-size: 26px;">analytics</span>
                <h4 style='margin: 0; font-family: Outfit; color: #FFA000;'>Membership Predictions</h4>
            </div>
            """, unsafe_allow_html=True)
            
            semester_indicator = 1 if "1st" in semester else 0
            
            input_data = pd.DataFrame([{
                'population': population,
                'payment_ratio': payment_ratio,
                'semester_indicator': semester_indicator,
                'benefits_claimed': benefits_claimed,
                'officer_count': officer_count,
                'events_held': events_held
            }])
            
            pred_val = sklearn_model.predict(input_data)[0]
            pred_capped = min(population, max(0.0, pred_val))
            collection_rate = (pred_capped / population) * 100 if population > 0 else 0
            
            if collection_rate >= 80:
                glow_color = "#00C853"
                status_text = "Optimal Performance"
            elif collection_rate >= 50:
                glow_color = "#FFA000"
                status_text = "Moderate Performance"
            else:
                glow_color = "#FF5722"
                status_text = "Underperforming"
                
            st.markdown(f"""
            <div class="m3-card" style="border-left: 8px solid {glow_color}; text-align: center; padding: 36px 20px;">
                <span class="material-symbols-outlined" style="color: {glow_color}; font-size: 36px;">insights</span>
                <div style="font-size: 1.1em; color: #A69B95; margin-top: 8px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">Predicted Paid Memberships (Y)</div>
                <h1 style="font-size: 4.8em; font-weight: 800; margin: 12px 0; font-family: 'Outfit', sans-serif;" class="gradient-text">{int(np.round(pred_capped))}</h1>
                <div style="font-size: 1.2em; font-weight: 700; color: #FFF8E1; display: flex; align-items: center; justify-content: center; gap: 8px;">
                    <span>{collection_rate:.1f}% Collection Rate</span>
                    <span style="background-color: rgba(255,255,255,0.06); padding: 4px 10px; border-radius: 12px; font-size: 0.75em; color: {glow_color}; border: 1px solid {glow_color}33;">{status_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if pred_val != pred_capped:
                if pred_val < 0:
                    st.markdown(f"""
                    <div class="m3-alert-warning">
                        <span class="material-symbols-outlined" style="color: #FF3D00; margin-top: 2px;">warning</span>
                        <div>
                            <strong style="color: #FF3D00; font-family: 'Outfit';">Negative Output Capped</strong><br>
                            <span style="font-size: 0.9em; color: #EFEBE9;">The raw regression model predicted a negative count ({pred_val:.1f}). This usually indicates highly unfavorable conditions. Capped to 0.</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_val > population:
                    st.markdown(f"""
                    <div class="m3-alert-warning">
                        <span class="material-symbols-outlined" style="color: #FF3D00; margin-top: 2px;">warning</span>
                        <div>
                            <strong style="color: #FF3D00; font-family: 'Outfit';">Exceeded Enrollment Limit</strong><br>
                            <span style="font-size: 0.9em; color: #EFEBE9;">The raw model predicted {pred_val:.1f} payments, exceeding total enrollment ({population}). Capped to population limit.</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            if benefits_claimed > pred_capped:
                st.markdown(f"""
                <div class="m3-alert-warning">
                    <span class="material-symbols-outlined" style="color: #FF3D00; margin-top: 2px;">error</span>
                    <div>
                        <strong style="color: #FF3D00; font-family: 'Outfit';">Consistency Conflict</strong><br>
                        <span style="font-size: 0.9em; color: #EFEBE9;">Benefits claimed ({benefits_claimed}) exceed predicted paying members ({int(np.round(pred_capped))}). Paid membership is typically required to claim council benefits.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"""
            <div class="m3-card-tonal">
                <h4 style="margin-top: 0; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; color: #FFA000; font-family: 'Outfit';">
                    <span class="material-symbols-outlined">lightbulb</span>
                    Council Strategy Advisor
                </h4>
                <div style="font-size: 0.95em; line-height: 1.55; color: #EFEBE9;">
                    <ul style="margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 8px;">
                        <li><strong>Modality:</strong> Increasing online payments by 10% improves forecast collections by average of <strong>{training_results['coefficients'][1] * 0.1:.1f}</strong> students.</li>
                        <li><strong>Officer Staffing:</strong> Adding 1 active program officer yields an average increase of <strong>{training_results['coefficients'][4]:.1f}</strong> paid memberships.</li>
                        <li><strong>Events Promotion:</strong> Running 1 additional student event corresponds to an average collection increase of <strong>{training_results['coefficients'][5]:.1f}</strong> memberships.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
