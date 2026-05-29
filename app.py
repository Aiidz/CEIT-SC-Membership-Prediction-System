import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import streamlit.components.v1 as components
import statsmodels.api as sm
import base64
from src.preprocess import preprocess_data
from src.train import train_model


def st_echarts(options, height=520):
    import json
    import uuid
    uid = "ech" + uuid.uuid4().hex[:8]
    options_json = json.dumps(options)
    html = f"""<div id="{uid}" style="position:relative;width:100%;height:{height}px" onmouseenter="document.getElementById('{uid}_btn').style.opacity='1'" onmouseleave="document.getElementById('{uid}_btn').style.opacity='0'">
<div id="{uid}_chart" style="width:100%;height:100%"></div>
<button id="{uid}_btn" onclick="(function(){{var el=document.getElementById('{uid}');if(!document.fullscreenElement){{el.requestFullscreen?.()}}else{{document.exitFullscreen?.()}}}})()"
style="position:absolute;top:4px;right:4px;z-index:10;background:rgba(30,41,59,0.8);border:none;border-radius:6px;cursor:pointer;padding:4px;line-height:1;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .15s" title="Fullscreen">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg>
</button></div>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script>
var c=echarts.init(document.getElementById('{uid}_chart'));
c.setOption({options_json});
window.addEventListener('resize',function(){{c.resize();}});
</script>"""
    components.html(html, height=height)


# Page configuration
st.set_page_config(
    page_title="CEIT-SC Membership Predictor",
    page_icon="logo.jpg" if os.path.exists("logo.jpg") else "🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Configuration and Styling (Material 3 Dynamic Theme System)
THEMES = {
    "dark": {
        "bg_app": "#0F172A",
        "bg_app_header": "rgba(15, 23, 42, 0.85)",
        "bg_sidebar": "#0F172A",
        "text_primary": "#F1F5F9",
        "text_muted": "#94A3B8",
        "text_glow": "#FED7AA",
        "border_color": "#334155",
        "border_color_hero": "#334155",
        "border_color_input": "#334155",
        "bg_card": "#1E293B",
        "bg_hero": "linear-gradient(135deg, #1E293B 0%, #0F172A 100%)",
        "bg_card_tonal": "#334155",
        "bg_input": "#1E293B",
        "bg_tablist": "#1E293B",
        "bg_tab_hover": "rgba(249, 115, 22, 0.06)",
        "bg_tab_active": "rgba(249, 115, 22, 0.16)",
        "text_tab_active": "#F97316",
        "text_tab_hover": "#FED7AA",
        "bg_download": "#334155",
        "text_download": "#F97316",
        "border_download": "rgba(249, 115, 22, 0.3)",
        "chart_bg": "#0F172A",
        "chart_axes_bg": "#1E293B",
        "chart_text": "#F1F5F9",
        "chart_label": "#94A3B8",
        "chart_grid": "#1E293B",
        "chart_spine": "#334155",
        "chart_legend_bg": "#1E293B",
        "chart_legend_border": "#334155",
        "heatmap_colors": ["#1E293B", "#7C2D12", "#EA580C", "#F97316", "#FED7AA"]
    },
}

t = THEMES["dark"]

# Custom Styling — Minimal Clean Design
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

:root {{
    --bg-app: {t['bg_app']};
    --bg-card: {t['bg_card']};
    --bg-card-tonal: {t['bg_card_tonal']};
    --text-primary: {t['text_primary']};
    --text-muted: {t['text_muted']};
    --text-glow: {t['text_glow']};
    --border-color: {t['border_color']};
}}

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}

.stApp {{
    background-color: var(--bg-app) !important;
    color: var(--text-primary);
}}

[data-testid="stSidebar"] {{
    background-color: {t['bg_sidebar']} !important;
    border-right: 1px solid var(--border-color) !important;
    color: var(--text-primary);
}}

.stCaption, [data-testid="stCaption"] {{
    color: var(--text-muted) !important;
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}}

.gradient-text {{
    background: linear-gradient(135deg, #F97316 0%, #EA580C 50%, #FACC15 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}}

.material-symbols-outlined {{
    font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 20;
    vertical-align: middle;
}}

.m3-hero {{
    background: {t['bg_hero']};
    border-radius: 0.5rem;
    border: 1px solid {t['border_color_hero']};
    padding: 32px;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    gap: 24px;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}}

.m3-hero:hover {{
    box-shadow: 0 20px 25px -5px rgba(249, 115, 22, 0.05);
    border-color: rgba(249, 115, 22, 0.3) !important;
}}

.m3-card {{
    background-color: var(--bg-card);
    background-color: color-mix(in srgb, var(--bg-card) 85%, transparent);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color);
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}}

.m3-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 20px 25px -5px rgba(249, 115, 22, 0.05);
    border-color: rgba(249, 115, 22, 0.3) !important;
}}

.m3-card-tonal {{
    background-color: var(--bg-card-tonal);
    background-color: color-mix(in srgb, var(--bg-card-tonal) 85%, transparent);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 0.5rem;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid var(--border-color);
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}}

.m3-card-tonal:hover {{
    transform: translateY(-4px);
    box-shadow: 0 20px 25px -5px rgba(249, 115, 22, 0.05);
    border-color: rgba(249, 115, 22, 0.3) !important;
}}

.m3-alert {{
    background-color: rgba(249, 115, 22, 0.08);
    border-left: 4px solid #F97316;
    border-radius: 0.5rem;
    padding: 14px 18px;
    margin-bottom: 16px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}}

.m3-alert-warning {{
    background-color: rgba(239, 68, 68, 0.08);
    border-left: 4px solid #EF4444;
    border-radius: 0.5rem;
    padding: 14px 18px;
    margin-bottom: 16px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}}

.m3-table {{
    width: 100%;
    border-collapse: collapse;
    border-radius: 0.5rem;
    overflow: hidden;
}}

.m3-table th {{
    background-color: var(--bg-card-tonal);
    color: {t['text_tab_active']};
    text-align: left;
    padding: 12px 14px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}}

.m3-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-primary);
    background-color: var(--bg-card);
}}

.m3-table tr:last-child td {{
    border-bottom: none;
}}

/* Animation Keyframes */
@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(20px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes float {{
    0%, 100% {{
        transform: translateY(0px);
    }}
    50% {{
        transform: translateY(-8px);
    }}
}}

@keyframes pulse-slow {{
    0%, 100% {{
        opacity: 1;
    }}
    50% {{
        opacity: 0.6;
    }}
}}

@keyframes bounce-slow {{
    0%, 100% {{
        transform: translateY(0);
    }}
    50% {{
        transform: translateY(-3px);
    }}
}}

@keyframes shimmer {{
    0% {{
        background-position: -1000px 0;
    }}
    100% {{
        background-position: 1000px 0;
    }}
}}

@keyframes loading-slide {{
    0% {{
        transform: translateX(-100%);
    }}
    100% {{
        transform: translateX(100%);
    }}
}}

.animate-item {{
    animation: fadeInUp 0.7s ease-out forwards;
}}

.animate-float {{
    animation: float 3s ease-in-out infinite;
}}

.animate-pulse-slow {{
    animation: pulse-slow 3s ease-in-out infinite;
}}

.animate-bounce-slow {{
    animation: bounce-slow 2s ease-in-out infinite;
}}

.animate-loading-slide {{
    animation: loading-slide 1.5s ease-in-out infinite;
}}

.animate-shimmer {{
    animation: shimmer 2s linear infinite;
    background: linear-gradient(to right, transparent 0%, rgba(249, 115, 22, 0.1) 50%, transparent 100%);
    background-size: 1000px 100%;
}}

/* Background effects */
.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image: radial-gradient(circle at 1px 1px, rgba(148, 163, 184, 0.05) 1px, transparent 0);
    background-size: 32px 32px;
}}

.stApp::after {{
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(circle at 90% 10%, rgba(249, 115, 22, 0.12) 0%, transparent 50%),
        radial-gradient(circle at 10% 90%, rgba(59, 130, 246, 0.06) 0%, transparent 50%);
    background-size: 500px 500px, 400px 400px;
    background-repeat: no-repeat, no-repeat;
    background-position: top right, bottom left;
    animation: pulse-slow 8s ease-in-out infinite;
}}

/* Tab navigation icons */
[data-testid="stTabs"] button[data-testid="stTab"] {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.05em !important;
    padding: 10px 18px !important;
}}

[data-testid="stTabs"] button[data-testid="stTab"]::before {{
    font-family: 'Material Symbols Outlined';
    font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24;
    font-size: 22px;
}}

[data-testid="stTabs"] button[data-testid="stTab"]:nth-child(1)::before {{ content: "dashboard"; }}
[data-testid="stTabs"] button[data-testid="stTab"]:nth-child(2)::before {{ content: "database"; }}
[data-testid="stTabs"] button[data-testid="stTab"]:nth-child(3)::before {{ content: "query_stats"; }}
[data-testid="stTabs"] button[data-testid="stTab"]:nth-child(4)::before {{ content: "model_training"; }}
[data-testid="stTabs"] button[data-testid="stTab"]:nth-child(5)::before {{ content: "tune"; }}
</style>
""", unsafe_allow_html=True)

# Helper functions for UI
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            mime_type = "image/png" if image_path.endswith(".png") else "image/jpeg"
            return f"data:{mime_type};base64,{encoded_string}"
    return ""

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def m3_metric_card(label, value, icon, color="#F97316", bg_opacity=0.12):
    rgb_tuple = hex_to_rgb(color)
    return f"""
    <div style="border-left: 5px solid {color}; width: 100%; display: flex; align-items: center; gap: 14px;">
        <div style="background-color: rgba({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]}, {bg_opacity}); width: 44px; height: 44px; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            <span class="material-symbols-outlined" style="color: {color}; font-size: 22px;">{icon}</span>
        </div>
        <div style="overflow: hidden; text-align: left;">
            <div style="font-size: 0.8em; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{label}</div>
            <div style="font-size: 1.5em; font-weight: 700; color: var(--text-primary); margin-top: 2px; line-height: 1.1;">{value}</div>
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
            sig_badge = '<span style="background-color: rgba(34, 197, 94, 0.12); color: #22C55E; padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 0.85em; display: inline-block;">★ Significant</span>'
        else:
            sig_badge = '<span style="background-color: rgba(148, 163, 184, 0.12); color: var(--text-muted); padding: 4px 10px; border-radius: 20px; font-weight: 500; font-size: 0.85em; display: inline-block;">Not Significant</span>'
        
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
            <td style="font-family: monospace; color: var(--text-muted);">{row['Standard Error']:.4f}</td>
            <td style="font-family: monospace; color: var(--text-muted);">{row['t-Statistic']:.2f}</td>
            <td style="font-family: monospace; font-weight: 600; color: {'#22C55E' if p_val < 0.05 else 'var(--text-primary)'};">{p_val:.4f}</td>
            <td>{sig_badge}</td>
        </tr>
        """
    html += '</tbody></table>'
    return html

# ECharts interactive chart helpers
def _echarts_base(title, x_name=None, y_name=None, height=520):
    return {
        "backgroundColor": t['chart_bg'],
        "title": {
            "text": title, "left": "center", "top": 0,
            "textStyle": {"color": t['chart_text'], "fontSize": 13, "fontWeight": "bold"},
        },
        "tooltip": {
            "backgroundColor": t['chart_legend_bg'],
            "borderColor": "transparent",
            "textStyle": {"color": t['chart_text'], "fontSize": 13},
            "extraCssText": "border-radius: 10px; box-shadow: none;",
        },
        "legend": {
            "textStyle": {"color": t['chart_text']},
            "top": 30,
        },
        "grid": {"left": 50, "right": 16, "top": 60, "bottom": 36, "containLabel": True},
        "xAxis": {
            "name": x_name,
            "nameTextStyle": {"color": t['chart_label'], "fontSize": 10},
            "axisLabel": {"color": t['chart_label'], "fontSize": 9},
            "axisLine": {"show": False},
            "splitLine": {"lineStyle": {"color": t['chart_grid'], "width": 0.5}},
        },
        "yAxis": {
            "name": y_name,
            "nameTextStyle": {"color": t['chart_label'], "fontSize": 10},
            "axisLabel": {"color": t['chart_label'], "fontSize": 9},
            "axisLine": {"show": False},
            "splitLine": {"lineStyle": {"color": t['chart_grid'], "width": 0.5}},
        },
    }

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_cols = ['population', 'payment_ratio', 'semester_indicator', 'benefits_claimed', 'officer_count', 'events_held', 'paid_memberships']
    corr_cols = [c for c in corr_cols if c in numeric_df.columns]
    corr = numeric_df[corr_cols].corr()
    labels = [c.replace('_', ' ').title() for c in corr.columns]
    n = len(labels)

    opts = _echarts_base('Correlation Matrix of Numeric Features')
    data = [[j, i, round(corr.values[i][j], 2)] for i in range(n) for j in range(n)]
    opts["xAxis"] = {"type": "category", "data": labels, "axisLabel": {"rotate": 45, "color": t['chart_label'], "fontSize": 9}, "splitLine": {"show": False}}
    opts["yAxis"] = {"type": "category", "data": labels, "axisLabel": {"color": t['chart_label'], "fontSize": 9}, "splitLine": {"show": False}}
    opts["series"] = [{
        "type": "heatmap", "data": data,
        "label": {"show": True, "color": t['chart_text'], "fontSize": 9, "fontWeight": "bold", "formatter": "{c}"},
        "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}},
    }]
    opts["visualMap"] = {"show": False, "min": -1, "max": 1, "inRange": {"color": t['heatmap_colors']}}
    opts["tooltip"]["trigger"] = "item"
    return opts

def plot_scatter_population_memberships(df):
    color_map = {
        'BSCS': '#F97316', 'BSIT': '#3B82F6', 'BSCE': '#10B981',
        'BSEE': '#8B5CF6', 'BSAE': '#EC4899', 'BSAB': '#94A3B8'
    }

    x_line = np.linspace(df['population'].min(), df['population'].max(), 100)
    coef = np.polyfit(df['population'], df['paid_memberships'], 1)
    poly1d_fn = np.poly1d(coef)
    trend_data = [[round(x, 1), round(poly1d_fn(x), 2)] for x in x_line]

    series = []
    for prog in df['program'].unique():
        prog_data = df[df['program'] == prog][['population', 'paid_memberships']].values.tolist()
        series.append({
            "type": "scatter", "name": prog, "data": prog_data,
            "symbolSize": 7,
            "itemStyle": {"color": color_map.get(prog, t['text_tab_active'])},
        })

    series.append({
        "type": "line", "name": "Trendline", "data": trend_data,
        "symbol": "none", "smooth": True,
        "lineStyle": {"color": t['text_tab_active'], "width": 1.5, "type": "dashed"},
    })

    opts = _echarts_base('Program Population vs. Paid Memberships', x_name='Total Program Population', y_name='Paid Memberships')
    opts["xAxis"]["type"] = "value"
    opts["yAxis"]["type"] = "value"
    opts["series"] = series
    opts["tooltip"]["trigger"] = "item"
    opts["tooltip"]["formatter"] = "{b}: ({c})"
    return opts

def plot_avg_memberships_program(df):
    avg_paid = df.groupby('program')['paid_memberships'].mean().reset_index()
    avg_paid = avg_paid.sort_values(by='paid_memberships', ascending=False)
    bar_colors = ['#F97316', '#3B82F6', '#10B981', '#8B5CF6', '#EC4899', '#94A3B8'][:len(avg_paid)]

    data = [{"value": round(v, 1), "itemStyle": {"color": c}} for v, c in zip(avg_paid['paid_memberships'], bar_colors)]

    opts = _echarts_base('Average Paid Memberships by Degree Program', y_name='Average Paid Memberships')
    opts["xAxis"] = {"type": "category", "data": avg_paid['program'].tolist(), "axisLabel": {"color": t['chart_label'], "fontSize": 9}, "axisLine": {"show": False}}
    opts["series"] = [{
        "type": "bar", "data": data, "barWidth": "55%",
        "label": {"show": True, "position": "top", "color": t['chart_text'], "fontSize": 9, "fontWeight": "bold", "formatter": "{c}"},
    }]
    return opts

def plot_membership_trend(df):
    df_copy = df.copy()
    df_copy['period'] = df_copy['academic_year'] + ' ' + df_copy['semester']
    avg_trend = df_copy.groupby('period')['paid_memberships'].sum().reset_index()
    avg_trend = avg_trend.sort_values(by='period')

    opts = _echarts_base('Council Membership Trend Over Semesters', y_name='Total Paid Memberships')
    opts["xAxis"] = {"type": "category", "data": avg_trend['period'].tolist(), "axisLabel": {"color": t['chart_label'], "fontSize": 8, "rotate": 15}, "axisLine": {"show": False}}
    opts["series"] = [{
        "type": "line", "data": [int(v) for v in avg_trend['paid_memberships'].tolist()],
        "symbol": "circle", "symbolSize": 7,
        "lineStyle": {"color": "#F97316", "width": 2},
        "itemStyle": {"color": "#FED7AA", "borderColor": "#EA580C", "borderWidth": 1.5},
        "areaStyle": {"color": "rgba(249,115,22,0.12)"},
        "label": {"show": True, "position": "top", "color": t['chart_text'], "fontSize": 9, "fontWeight": "bold"},
    }]
    return opts

def plot_residual_chart(y_test, y_pred):
    data = [[float(x), float(y)] for x, y in zip(y_test, y_pred)]
    min_val = float(min(y_test.min(), y_pred.min()))
    max_val = float(max(y_test.max(), y_pred.max()))

    opts = _echarts_base('Model Residual Plot (Actual vs. Predicted)', x_name='Actual Paid Memberships', y_name='Predicted Paid Memberships')
    opts["xAxis"]["type"] = "value"
    opts["yAxis"]["type"] = "value"
    opts["tooltip"]["trigger"] = "item"
    opts["series"] = [
        {
            "type": "scatter", "name": "Actual Data", "data": data,
            "symbolSize": 7,
            "itemStyle": {"color": "#F97316", "borderColor": "#EA580C", "borderWidth": 1},
        },
        {
            "type": "line", "name": "Perfect Fit", "data": [[min_val, min_val], [max_val, max_val]],
            "symbol": "none",
            "lineStyle": {"color": t['text_glow'], "width": 2, "type": "dashed"},
        },
    ]
    return opts

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
    <div class="m3-alert" style="border-left-color: #22C55E; background-color: rgba(34, 197, 94, 0.08); margin-bottom: 16px; margin-top: 16px;">
        <div style="font-size: 0.8em; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">System Status</div>
        <div style="font-size: 0.95em; font-weight: 700; color: #22C55E; margin-top: 2px; display: flex; align-items: center; gap: 6px;">
            <span class="material-symbols-outlined" style="font-size: 16px;">check_circle</span> Active Predictor
        </div>
    </div>
    <div class="m3-card" style="padding: 14px; margin-bottom: 24px;">
        <div style="font-size: 0.8em; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">Model Engine</div>
        <div style="font-size: 0.9em; color: var(--text-primary); margin-top: 2px; font-weight: 500;">Multiple Linear Regression</div>
        <div style="font-size: 0.8em; color: var(--text-muted); margin-top: 2px;">sklearn + statsmodels OLS</div>
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
    <div class="m3-alert-warning" style="margin-bottom: 24px; margin-top: 16px;">
        <div style="font-size: 0.8em; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">System Status</div>
        <div style="font-size: 0.95em; font-weight: 700; color: #EF4444; margin-top: 2px; display: flex; align-items: center; gap: 6px;">
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
        <img src="{logo_base64}" style="width: 80px; height: 80px; border-radius: 0.5rem; border: 2px solid {t['text_tab_active']};" />
        <div>
            <h1 class="gradient-text" style="margin: 0; font-size: 2.2em; font-family: 'Outfit';">CEIT-SC Membership Prediction System</h1>
            <p style="margin: 6px 0 0 0; color: var(--text-muted); font-size: 1.05em; font-weight: 500;">
                Cavite State University College of Engineering and Information Technology Student Council
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="m3-card">
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 8px; font-family: 'Outfit';">
            <span class="material-symbols-outlined" style="color: {t['text_tab_active']}; font-size: 28px;">auto_awesome</span>
            System Onboarding Wizard
        </h3>
        <p style="color: var(--text-primary); line-height: 1.6; margin-bottom: 0;">
            Welcome! To initialize the predictive dashboard, the system needs to process historical enrollment and collection logs. 
            Please upload a raw CSV dataset containing your council history. The backend pipeline will clean the data, calculate relative features, and fit the regression equations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select raw enrollment and collection CSV file to begin onboarding:", type=["csv"])
    
    with st.expander("📋 Review Expected CSV Dataset Structure", expanded=False):
        st.html("""
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
        """)
        
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
                <div class="m3-alert" style="border-left-color: #22C55E; background-color: rgba(34, 197, 94, 0.08); margin-top: 20px;">
                    <span class="material-symbols-outlined" style="color: #22C55E;">check_circle</span>
                    <div>
                        <strong style="color: #22C55E; font-family: 'Outfit';">Model Initialized Successfully</strong><br>
                        <span style="font-size: 0.95em; color: var(--text-primary);">The Multiple Linear Regression model has been trained on {len(clean_df)} historical records.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show results in custom metric cards
                col_on1, col_on2, col_on3 = st.columns(3)
                with col_on1:
                    st.markdown(m3_metric_card("Dataset Rows", f"{len(clean_df)}", "database"), unsafe_allow_html=True)
                with col_on2:
                    st.markdown(m3_metric_card("R² Accuracy", f"{results['r2']:.4f}", "emoji_events", "#22C55E"), unsafe_allow_html=True)
                with col_on3:
                    st.markdown(m3_metric_card("RMSE Loss", f"{results['rmse']:.2f}", "trending_down"), unsafe_allow_html=True)
                
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
                    <h4 style="margin-top: 0; font-family: 'Outfit'; color: {t['text_tab_active']};">Statistical Significance Check</h4>
                    <p style="margin-bottom: 0;">Variables with statistically significant influence (p < 0.05):</p>
                    <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
                        {" ".join([f'<span style="background-color: rgba(34, 197, 94, 0.12); color: #22C55E; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9em;">{name}</span>' for name in sig_display]) if sig_display else '<span style="color: var(--text-muted);">None (check sample size)</span>'}
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
        <img src="{logo_base64}" style="width: 80px; height: 80px; border-radius: 0.5rem; border: 2px solid {t['text_tab_active']};" />
        <div>
            <h1 class="gradient-text" style="margin: 0; font-size: 2.2em; font-family: 'Outfit';">CEIT-SC Membership Analytics Portal</h1>
            <p style="margin: 4px 0 0 0; color: var(--text-muted); font-size: 1.02em; font-weight: 500;">
                Cavite State University College of Engineering and Information Technology Student Council
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # M3 Pill Navigation Tabs
    tab_home, tab_explorer, tab_eda, tab_model, tab_predict = st.tabs([
        "Overview", 
        "Data Explorer", 
        "EDA Dashboard", 
        "Model Results", 
        "Predictor"
    ])
    
    # ------------------------------------------
    # TAB 1: HOME
    # ------------------------------------------
    with tab_home:
        col_main, col_ipo = st.columns([13, 10])
        
        with col_main:
            st.markdown(f"""
            <div class="m3-card">
                <h3 style="margin-top: 0; color: {t['text_tab_active']}; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                    <span class="material-symbols-outlined">school</span>
                    Project Overview
                </h3>
                <p style="line-height: 1.6; color: var(--text-primary);">
                    The <strong>CEIT-SC Membership Prediction System</strong> is a decision support application engineered for the 
                    <strong>College of Engineering and Information Technology Student Council (CEIT-SC)</strong> at Cavite State University (CvSU) — Main Campus.
                </p>
                <p style="line-height: 1.6; color: var(--text-primary);">
                    By leveraging a Multiple Linear Regression (MLR) framework, the tool analyzes historical collection figures, student program sizes, 
                    semester variables, and officer profiles to forecast future membership fee collections. This enables the student council 
                    to plan organization budgets, event resources, and student benefits with data-backed accuracy.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="m3-card">
                <h3 style="margin-top: 0; color: {t['text_tab_active']}; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                    <span class="material-symbols-outlined">help_center</span>
                    Research Questions Addressed
                </h3>
                <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 16px;">
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <span style="background-color: {t['bg_tab_active']}; color: {t['text_tab_active']}; padding: 4px 10px; border-radius: 8px; font-weight: 700; font-size: 0.85em;">RQ1</span>
                        <div><strong>Historical Profile:</strong> What are the historical enrollment patterns, fee collection volumes, and modality preferences (cash vs online) across different CEIT courses?</div>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <span style="background-color: {t['bg_tab_active']}; color: {t['text_tab_active']}; padding: 4px 10px; border-radius: 8px; font-weight: 700; font-size: 0.85em;">RQ2</span>
                        <div><strong>Feature Significance:</strong> Which organizational and external factors (population, payment modality, benefits claimed, officer counts, event frequencies) significantly influence membership collections?</div>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: flex-start;">
                        <span style="background-color: {t['bg_tab_active']}; color: {t['text_tab_active']}; padding: 4px 10px; border-radius: 8px; font-weight: 700; font-size: 0.85em;">RQ3</span>
                        <div><strong>Predictive Accuracy:</strong> How accurately can the fitted multiple regression model predict collection outcomes for upcoming semesters?</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="m3-card-tonal">
                <h4 style="margin-top: 0; color: var(--text-glow); font-family: 'Outfit';">CPEN70 Course Project Group</h4>
                <p style="margin-bottom: 0; color: var(--text-muted); font-size: 0.95em;">
                    Aguilar · Bergado · Bituin · Guarin · Reyes · Sarmiento<br>
                    <strong>Cavite State University (CvSU) — Main Campus</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_ipo:
            st.markdown("""
            <div class="m3-card">
                <h3 style="margin-top: 0; color: {t['text_tab_active']}; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                    <span class="material-symbols-outlined">account_tree</span>
                    System Architecture (IPO)
                </h3>
                <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 20px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="background-color: rgba(59, 130, 246, 0.12); color: #3B82F6; font-weight: 700; width: 85px; padding: 6px; border-radius: 0.5rem; text-align: center; font-size: 0.85em;">INPUT</div>
                        <div style="font-size: 0.9em; color: var(--text-primary);">Program population (X₁), online payment ratio (X₂), semester index (X₃), benefits claimed (X₄), officer count (X₅), events held (X₆)</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="background-color: {t['bg_tab_active']}; color: {t['text_tab_active']}; font-weight: 700; width: 85px; padding: 6px; border-radius: 0.5rem; text-align: center; font-size: 0.85em;">PROCESS</div>
                        <div style="font-size: 0.9em; color: var(--text-primary);">Data Cleaning & Ratio Calculation → Scikit-learn LinearRegression & Statsmodels OLS Fit</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="background-color: rgba(34, 197, 94, 0.12); color: #22C55E; font-weight: 700; width: 85px; padding: 6px; border-radius: 0.5rem; text-align: center; font-size: 0.85em;">OUTPUT</div>
                        <div style="font-size: 0.9em; color: var(--text-primary);">Predicted student memberships (Y), OLS statistical coefficients, strategy advisory insights</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="m3-card" style="border-left: 6px solid #22C55E;">
                <h4 style="margin-top: 0; color: var(--text-primary); font-family: 'Outfit';">Active Model Performance</h4>
                <div style="margin-top: 16px; display: flex; flex-direction: column; gap: 12px;">
            """, unsafe_allow_html=True)
            st.markdown(m3_metric_card("Model Accuracy (R²)", f"{training_results['r2']:.4f}", "verified", "#10B981", 0.08), unsafe_allow_html=True)
            st.markdown(m3_metric_card("Average Error (MAE)", f"{training_results['mae']:.1f} students", "bar_chart"), unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    # ------------------------------------------
    # TAB 2: DATA EXPLORER
    # ------------------------------------------
    with tab_explorer:
        st.markdown("""
        <div class="m3-card" style="margin-bottom: 24px;">
            <h3 style="margin-top: 0; color: {t['text_tab_active']}; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                <span class="material-symbols-outlined">database</span>
                Dataset Explorer & Filters
            </h3>
            <p style="color: var(--text-muted); margin-bottom: 16px;">Filter and inspect historical records of student population, modalities, and membership collections.</p>
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
            <h4 style="margin-top: 0; color: {t['text_tab_active']}; font-family: 'Outfit';">Descriptive Statistics Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(filtered_df.describe().T, use_container_width=True)

    # ------------------------------------------
    # TAB 3: EDA DASHBOARD
    # ------------------------------------------
    with tab_eda:
        st.markdown("""
        <div class="m3-card">
            <h3 style="margin-top: 0; color: {t['text_tab_active']}; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                <span class="material-symbols-outlined">insights</span>
                Exploratory Data Analysis
            </h3>
            <p style="color: var(--text-muted); margin-bottom: 0;">Explore historical trends, correlations, and relationships within the student council membership records.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig1 = plot_correlation_heatmap(df)
            st_echarts(fig1)
            st.caption("Correlation Heatmap: Identifies linear association strengths (close to 1.00 is high correlation) between student statistics and collections.")
            
        with col_chart2:
            fig2 = plot_scatter_population_memberships(df)
            st_echarts(fig2)
            st.caption("Scatter Plot: Displays the relationship between student enrollment counts and paid memberships by program, along with overall linear fit.")
            
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            fig3 = plot_avg_memberships_program(df)
            st_echarts(fig3)
            st.caption("Bar Chart: Illustrates the average number of paid student memberships generated by each department program over historical semesters.")
            
        with col_chart4:
            fig4 = plot_membership_trend(df)
            st_echarts(fig4)
            st.caption("Line Chart: Highlights the aggregate sum of student memberships collected per academic period across the main campus.")

    # ------------------------------------------
    # TAB 4: MODEL RESULTS
    # ------------------------------------------
    with tab_model:
        st.markdown("""
        <div class="m3-card">
            <h3 style="margin-top: 0; color: {t['text_tab_active']}; font-family: 'Outfit'; display: flex; align-items: center; gap: 8px;">
                <span class="material-symbols-outlined">query_stats</span>
                Regression Equation & Statistical Significance
            </h3>
            <p style="color: var(--text-muted); margin-bottom: 0;">Evaluate the Multiple Linear Regression (MLR) model parameters, residuals, and variable significance coefficients.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model formula
        _intercept = training_results['intercept']
        _c = training_results['coefficients']
        st.markdown(f"""
        <div class="m3-card" style="border-left: 6px solid {t['text_tab_active']};">
            <h4 style="margin-top: 0; font-family: 'Outfit'; color: {t['text_tab_active']}; display: flex; align-items: center; gap: 6px;">
                <span class="material-symbols-outlined">calculate</span>
                Fitted Regression Model Equation (Y)
            </h4>
            <div style="font-size: 1.15em; font-family: monospace; color: var(--text-primary); padding: 12px; background-color: var(--bg-card-tonal); border-radius: 0.5rem; border: 1px solid var(--border-color); overflow-x: auto; white-space: nowrap; margin-top: 12px;">
                Y_pred = {_intercept:.2f} 
                + ({_c[0]:.4f} × Pop) 
                + ({_c[1]:.2f} × PayRatio) 
                + ({_c[2]:.2f} × SemInd) 
                + ({_c[3]:.4f} × Benefits) 
                + ({_c[4]:.2f} × Officers) 
                + ({_c[5]:.2f} × Events)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # M3 Metric row
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            r2_val = training_results['r2']
            r2_color = "#22C55E" if r2_val >= 0.85 else t['text_tab_active']
            st.markdown(m3_metric_card("R² Score (Model Fit)", f"{r2_val:.4f}", "verified", r2_color), unsafe_allow_html=True)
        with col_m2:
            st.markdown(m3_metric_card("Mean Absolute Error (MAE)", f"{training_results['mae']:.2f}", "error"), unsafe_allow_html=True)
        with col_m3:
            st.markdown(m3_metric_card("Root Mean Squared Error (RMSE)", f"{training_results['rmse']:.2f}", "analytics"), unsafe_allow_html=True)
            
        col_stats, col_residual = st.columns([13, 10])
        
        with col_stats:
            st.html("""
            <div style="margin-top: 16px; margin-bottom: 8px;">
                <h4 style="margin: 0; font-family: 'Outfit'; display: flex; align-items: center; gap: 6px;">
                    <span class="material-symbols-outlined" style="color: {t['text_tab_active']};">table_chart</span>
                    Coefficients & Statistical Significance (RQ2)
                </h4>
            </div>
            """)
            
            coef_df = pd.DataFrame({
                "Coefficient (β)": ols_model.params,
                "Standard Error": ols_model.bse,
                "t-Statistic": ols_model.tvalues,
                "p-Value": ols_model.pvalues,
            })
            
            html_table = make_html_coef_table(coef_df)
            st.html(html_table)
            
        with col_residual:
            st.html("""
            <div style="margin-top: 16px; margin-bottom: 8px;">
                <h4 style="margin: 0; font-family: 'Outfit'; display: flex; align-items: center; gap: 6px;">
                    <span class="material-symbols-outlined" style="color: {t['text_tab_active']};">scatter_plot</span>
                    Residual Analysis
                </h4>
            </div>
            """)
            
            from sklearn.model_selection import train_test_split
            X = df[training_results['features']]
            y = df['paid_memberships']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = sklearn_model.predict(X_test)
            
            fig_resid = plot_residual_chart(y_test, y_pred)
            st_echarts(fig_resid)

    # ------------------------------------------
    # TAB 5: PREDICTOR PLAYGROUND
    # ------------------------------------------
    with tab_predict:
        col_inputs, col_gauge = st.columns([11, 10])
        
        with col_inputs:
            st.markdown("""
            <div style='margin-bottom: 16px; display: flex; align-items: center; gap: 8px;'>
                <span class="material-symbols-outlined" style="color: {t['text_tab_active']}; font-size: 26px;">tune</span>
                <h4 style='margin: 0; font-family: Outfit; color: {t['text_tab_active']}'>Adjust Simulation Inputs</h4>
            </div>
            """, unsafe_allow_html=True)
            
            avg_pop = int(df['population'].mean())
            avg_ratio = float(df['payment_ratio'].mean())
            avg_benefits = int(df['benefits_claimed'].mean())
            avg_officers = int(df['officer_count'].mean())
            avg_events = int(df['events_held'].mean())
            
            st.markdown("""
            <div style="background-color: var(--bg-card-tonal); padding: 18px; border-radius: 0.5rem; border: 1px solid var(--border-color); margin-bottom: 20px;">
                <h5 style="margin-top: 0; margin-bottom: 12px; color: var(--text-glow); font-family: 'Outfit'; display: flex; align-items: center; gap: 6px;"><span class="material-symbols-outlined" style="font-size: 22px;">group</span>1. Program Demographics</h5>
            """, unsafe_allow_html=True)
            population = st.slider("Enrolled Program Population ($X_1$):", min_value=10, max_value=1200, value=avg_pop, step=10)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: var(--bg-card-tonal); padding: 18px; border-radius: 0.5rem; border: 1px solid var(--border-color); margin-bottom: 20px;">
                <h5 style="margin-top: 0; margin-bottom: 12px; color: var(--text-glow); font-family: 'Outfit'; display: flex; align-items: center; gap: 6px;"><span class="material-symbols-outlined" style="font-size: 22px;">credit_card</span>2. Collection Modality & Calendar</h5>
            """, unsafe_allow_html=True)
            payment_ratio = st.slider("Online Payment Ratio ($X_2$):", min_value=0.0, max_value=1.0, value=avg_ratio, step=0.05, help="Proportion of payments collected via GCash/PayMaya versus Cash.")
            semester = st.radio("Academic Term ($X_3$):", options=["1st Semester (High Activity)", "2nd Semester (Low Activity)"], index=0, horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: var(--bg-card-tonal); padding: 18px; border-radius: 0.5rem; border: 1px solid var(--border-color);">
                <h5 style="margin-top: 0; margin-bottom: 12px; color: var(--text-glow); font-family: 'Outfit'; display: flex; align-items: center; gap: 6px;"><span class="material-symbols-outlined" style="font-size: 22px;">celebration</span>3. Engagement & Outreach Features</h5>
            """, unsafe_allow_html=True)
            benefits_claimed = st.slider("Benefits Claimed ($X_4$):", min_value=0, max_value=1200, value=min(avg_benefits, 720), step=5)
            
            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                officer_count = st.slider("Active Officers ($X_5$):", min_value=0, max_value=15, value=avg_officers, step=1)
            with col_sub2:
                events_held = st.slider("Council Events ($X_6$):", min_value=0, max_value=25, value=avg_events, step=1)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_gauge:
            st.markdown("""
            <div style='margin-bottom: 16px; display: flex; align-items: center; gap: 8px;'>
                <span class="material-symbols-outlined" style="color: {t['text_tab_active']}; font-size: 26px;">analytics</span>
                <h4 style='margin: 0; font-family: Outfit; color: {t['text_tab_active']}'>Membership Predictions</h4>
            </div>
            """, unsafe_allow_html=True)
            
            semester_indicator = 1 if "1st" in semester else 0
            
            benefits_claimed_model = min(benefits_claimed, population)
            
            input_data = pd.DataFrame([{
                'population': population,
                'payment_ratio': payment_ratio,
                'semester_indicator': semester_indicator,
                'benefits_claimed': benefits_claimed_model,
                'officer_count': officer_count,
                'events_held': events_held
            }])
            
            pred_val = sklearn_model.predict(input_data)[0]
            pred_capped = min(population, max(0.0, pred_val))
            collection_rate = (pred_capped / population) * 100 if population > 0 else 0
            
            if collection_rate >= 80:
                glow_color = "#22C55E"
                status_text = "Optimal Performance"
            elif collection_rate >= 50:
                glow_color = t['text_tab_active']
                status_text = "Moderate Performance"
            else:
                glow_color = "#EF4444"
                status_text = "Underperforming"
                
            st.markdown(f"""
            <div class="m3-card" style="border-left: 8px solid {glow_color}; text-align: center; padding: 36px 20px;">
                <span class="material-symbols-outlined" style="color: {glow_color}; font-size: 36px;">insights</span>
                <div style="font-size: 1.1em; color: var(--text-muted); margin-top: 8px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">Predicted Paid Memberships (Y)</div>
                <h1 style="font-size: 4.8em; font-weight: 800; margin: 12px 0; font-family: 'Outfit', sans-serif;" class="gradient-text">{int(np.round(pred_capped))}</h1>
                <div style="font-size: 1.2em; font-weight: 700; color: var(--text-glow); display: flex; align-items: center; justify-content: center; gap: 8px;">
                    <span>{collection_rate:.1f}% Collection Rate</span>
                    <span style="background-color: rgba(255,255,255,0.06); padding: 4px 10px; border-radius: 0.5rem; font-size: 0.75em; color: {glow_color}; border: 1px solid {glow_color}33;">{status_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if benefits_claimed > population:
                st.markdown(f"""
                <div class="m3-alert-warning">
                    <span class="material-symbols-outlined" style="color: #F97316; margin-top: 2px;">warning</span>
                    <div>
                        <strong style="color: #F97316; font-family: 'Outfit';">Benefits Exceed Population</strong><br>
                        <span style="font-size: 0.9em; color: var(--text-primary);">Benefits claimed ({benefits_claimed}) cannot exceed total enrolled population ({population}). Capping benefits to {population} for the simulation model.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if pred_val != pred_capped:
                if pred_val < 0:
                    st.markdown(f"""
                    <div class="m3-alert-warning">
                        <span class="material-symbols-outlined" style="color: #EF4444; margin-top: 2px;">warning</span>
                        <div>
                            <strong style="color: #EF4444; font-family: 'Outfit';">Negative Output Capped</strong><br>
                            <span style="font-size: 0.9em; color: var(--text-primary);">The raw regression model predicted a negative count ({pred_val:.1f}). This usually indicates highly unfavorable conditions. Capped to 0.</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_val > population:
                    st.markdown(f"""
                    <div class="m3-alert-warning">
                        <span class="material-symbols-outlined" style="color: #EF4444; margin-top: 2px;">warning</span>
                        <div>
                            <strong style="color: #EF4444; font-family: 'Outfit';">Exceeded Enrollment Limit</strong><br>
                            <span style="font-size: 0.9em; color: var(--text-primary);">The raw model predicted {pred_val:.1f} payments, exceeding total enrollment ({population}). Capped to population limit.</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            if benefits_claimed_model > pred_capped:
                st.markdown(f"""
                <div class="m3-alert-warning">
                    <span class="material-symbols-outlined" style="color: #EF4444; margin-top: 2px;">error</span>
                    <div>
                        <strong style="color: #EF4444; font-family: 'Outfit';">Consistency Conflict</strong><br>
                        <span style="font-size: 0.9em; color: var(--text-primary);">Benefits claimed ({benefits_claimed_model}) exceed predicted paying members ({int(np.round(pred_capped))}). Paid membership is typically required to claim council benefits.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"""
            <div class="m3-card-tonal">
                <h4 style="margin-top: 0; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; color: {t['text_tab_active']}; font-family: 'Outfit';">
                    <span class="material-symbols-outlined">lightbulb</span>
                    Council Strategy Advisor
                </h4>
                <div style="font-size: 0.95em; line-height: 1.55; color: var(--text-primary);">
                    <ul style="margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 8px;">
                        <li><strong>Modality:</strong> Increasing online payments by 10% improves forecast collections by average of <strong>{training_results['coefficients'][1] * 0.1:.1f}</strong> students.</li>
                        <li><strong>Officer Staffing:</strong> Adding 1 active program officer yields an average increase of <strong>{training_results['coefficients'][4]:.1f}</strong> paid memberships.</li>
                        <li><strong>Events Promotion:</strong> Running 1 additional student event corresponds to an average collection increase of <strong>{training_results['coefficients'][5]:.1f}</strong> memberships.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
