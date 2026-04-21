import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DARK_CSS = """
:root {
  --bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--accent:#58a6ff;--accent2:#3fb950;
  --accent3:#f78166;--accent4:#d2a8ff;--text:#e6edf3;--text2:#8b949e;
  --border:#30363d;--card-bg:#161b22;--card-hover:#1c2128;
  --shadow:0 8px 32px rgba(0,0,0,0.5);--radius:12px;
}"""
LIGHT_CSS = """
:root {
  --bg:#f0f4f8;--bg2:#ffffff;--bg3:#e8edf2;--accent:#1a73e8;--accent2:#0d904f;
  --accent3:#d93025;--accent4:#7c4dff;--text:#1a1a2e;--text2:#5f6368;
  --border:#dadce0;--card-bg:#ffffff;--card-hover:#f8f9fa;
  --shadow:0 4px 20px rgba(0,0,0,0.1);--radius:12px;
}"""
BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.stButton>button{background:transparent;border:1px solid var(--border);color:var(--text)!important;border-radius:8px;padding:8px 16px;font-family:'DM Sans',sans-serif;font-size:14px;transition:all 0.2s ease;width:100%;}
.stButton>button:hover{background:var(--accent);border-color:var(--accent);color:#fff!important;transform:translateY(-1px);box-shadow:0 4px 12px rgba(88,166,255,0.3);}
.stSelectbox>div,.stSlider>div{color:var(--text)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg3);border-radius:var(--radius);padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:var(--text2)!important;border-radius:8px;font-family:'DM Sans',sans-serif;font-size:13px;font-weight:500;padding:8px 16px;}
.stTabs [aria-selected="true"]{background:var(--accent)!important;color:#fff!important;}
.streamlit-expanderHeader{background:var(--bg3)!important;border-radius:8px!important;color:var(--text)!important;}
.streamlit-expanderContent{background:var(--bg2)!important;border:1px solid var(--border);border-radius:0 0 8px 8px;}
[data-testid="stMetric"]{background:var(--card-bg);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;box-shadow:var(--shadow);}
[data-testid="stMetricLabel"]{color:var(--text2)!important;font-size:12px!important;}
[data-testid="stMetricValue"]{color:var(--accent)!important;font-family:'Space Mono',monospace!important;}
.stDataFrame{border-radius:var(--radius);overflow:hidden;}
::-webkit-scrollbar{width:6px;}::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent);}
#MainMenu,footer,header{visibility:hidden;}[data-testid="stToolbar"]{display:none;}
"""
COMPONENT_CSS = """
.dal-hero{background:linear-gradient(135deg,var(--accent) 0%,var(--accent4) 100%);border-radius:var(--radius);padding:40px;margin-bottom:28px;position:relative;overflow:hidden;}
.dal-hero::before{content:'';position:absolute;top:-50%;right:-20%;width:300px;height:300px;background:rgba(255,255,255,0.05);border-radius:50%;}
.dal-hero::after{content:'';position:absolute;bottom:-30%;right:10%;width:200px;height:200px;background:rgba(255,255,255,0.05);border-radius:50%;}
.dal-hero h1{font-family:'Space Mono',monospace;font-size:2.4rem;color:#fff;font-weight:700;letter-spacing:-1px;margin-bottom:8px;}
.dal-hero p{color:rgba(255,255,255,0.85);font-size:1rem;font-weight:400;}
.exp-card{background:var(--card-bg);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:12px;transition:all 0.25s ease;cursor:pointer;position:relative;overflow:hidden;}
.exp-card::before{content:'';position:absolute;top:0;left:0;width:4px;height:100%;background:var(--accent);border-radius:4px 0 0 4px;}
.exp-card:hover{background:var(--card-hover);border-color:var(--accent);transform:translateX(4px);box-shadow:var(--shadow);}
.exp-card .exp-num{font-family:'Space Mono',monospace;font-size:11px;color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing:1px;}
.exp-card .exp-title{font-size:15px;font-weight:600;color:var(--text);margin:4px 0 6px;}
.exp-card .exp-desc{font-size:12px;color:var(--text2);line-height:1.5;}
.exp-card .exp-icon{font-size:28px;margin-bottom:8px;}
.section-header{font-family:'Space Mono',monospace;font-size:13px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:2px;margin:24px 0 12px;display:flex;align-items:center;gap:8px;}
.section-header::after{content:'';flex:1;height:1px;background:var(--border);}
.exp-banner{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:24px 28px;margin-bottom:20px;display:flex;align-items:center;gap:16px;}
.exp-banner .banner-icon{font-size:3rem;}
.exp-banner .banner-num{font-family:'Space Mono',monospace;font-size:11px;color:var(--accent);font-weight:700;letter-spacing:2px;text-transform:uppercase;}
.exp-banner .banner-title{font-size:1.6rem;font-weight:700;color:var(--text);font-family:'Space Mono',monospace;}
.insight-box{background:var(--bg3);border-left:3px solid var(--accent2);border-radius:0 var(--radius) var(--radius) 0;padding:16px 20px;margin:8px 0;}
.insight-box p{color:var(--text);font-size:14px;line-height:1.6;}
.tag{display:inline-block;background:var(--bg3);border:1px solid var(--border);color:var(--accent);font-size:11px;font-family:'Fira Code',monospace;padding:3px 10px;border-radius:20px;margin:3px;}
.step-item{display:flex;gap:12px;padding:10px 0;border-bottom:1px solid var(--border);align-items:flex-start;}
.step-num{min-width:28px;height:28px;background:var(--accent);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;font-family:'Space Mono',monospace;}
.step-text{font-size:14px;color:var(--text);line-height:1.6;padding-top:4px;}
.sidebar-brand{padding:16px;border-bottom:1px solid var(--border);margin-bottom:8px;}
.sidebar-brand h2{font-family:'Space Mono',monospace;font-size:14px;color:var(--accent);font-weight:700;}
.sidebar-brand p{font-size:11px;color:var(--text2);margin-top:4px;}
.about-box{background:var(--bg3);border-radius:var(--radius);padding:16px;margin-top:16px;border:1px solid var(--border);}
.about-box h4{font-family:'Space Mono',monospace;font-size:12px;color:var(--accent);margin-bottom:8px;}
.about-box p{font-size:12px;color:var(--text2);line-height:1.6;}
.conclusion-card{background:linear-gradient(135deg,var(--bg2),var(--bg3));border:1px solid var(--border);border-radius:var(--radius);padding:20px 24px;margin-top:16px;}
.conclusion-card h4{font-family:'Space Mono',monospace;font-size:13px;color:var(--accent4);margin-bottom:10px;}
.conclusion-card p{font-size:14px;color:var(--text);line-height:1.7;}
"""

AQI_COLORS = {
    'Good':'#3fb950','Moderate':'#ffa657',
    'Unhealthy for Sensitive Groups':'#d2a8ff',
    'Unhealthy':'#f78166','Very Unhealthy':'#ff4444','Hazardous':'#cc0000',
}
AQI_ORDER = ['Good','Moderate','Unhealthy for Sensitive Groups',
             'Unhealthy','Very Unhealthy','Hazardous']
PALETTE = ['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657','#ff8c00']


def inject_css(dark_mode):
    theme = DARK_CSS if dark_mode else LIGHT_CSS
    st.markdown(f"<style>{theme}{BASE_CSS}{COMPONENT_CSS}</style>", unsafe_allow_html=True)


def plotly_theme(dark):
    bg   = "#161b22" if dark else "#ffffff"
    bg2  = "#0d1117" if dark else "#f0f4f8"
    text = "#e6edf3" if dark else "#1a1a2e"
    grid = "#30363d" if dark else "#dadce0"
    return dict(paper_bgcolor=bg, plot_bgcolor=bg2,
                font=dict(family="DM Sans", color=text, size=13),
                xaxis=dict(gridcolor=grid, zerolinecolor=grid),
                yaxis=dict(gridcolor=grid, zerolinecolor=grid),
                margin=dict(l=40, r=20, t=50, b=40))


# ── chart builders ────────────────────────────────────────────────────────────

def build_boxplot(df, col, dark, group_col=None):
    theme = plotly_theme(dark)
    fig = go.Figure()
    if group_col and group_col in df.columns:
        groups = [g for g in AQI_ORDER if g in df[group_col].unique()]
        for i, grp in enumerate(groups):
            vals = df[df[group_col] == grp][col].dropna()
            fig.add_trace(go.Box(y=vals, name=grp,
                                 marker_color=AQI_COLORS.get(grp, PALETTE[i % 6]),
                                 boxmean='sd', notched=True, hoverinfo='y+name'))
    else:
        fig.add_trace(go.Box(y=df[col], name=col,
                             marker_color='#58a6ff', boxmean='sd', notched=True))
    fig.update_layout(**theme,
                      title=dict(text=f"Box Plot — {col}",
                                 font=dict(size=16, family="Space Mono")))
    return fig


def build_regression(df, x_col, y_col, dark):
    theme = plotly_theme(dark)
    slope, intercept, r, p, se = stats.linregress(df[x_col], df[y_col])
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    y_line  = slope * x_line + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers',
                             marker=dict(color='#58a6ff', size=5, opacity=0.4,
                                         line=dict(color='#fff', width=0.3)),
                             name='Observations'))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                             line=dict(color='#f78166', width=2.5),
                             name=f'Fit (R²={r**2:.4f})'))
    fig.update_layout(**theme,
                      title=dict(text="Linear Regression",
                                 font=dict(size=16, family="Space Mono")),
                      xaxis_title=x_col, yaxis_title=y_col)
    return fig


def build_sampling_comparison(df, dark, class_col='Is_Polluted'):
    theme = plotly_theme(dark)
    n = len(df)
    sample_n = min(200, n // 5)
    k = max(1, n // sample_n)
    stratified = pd.concat([
        grp.sample(frac=0.2, random_state=42)
        for _, grp in df.groupby(class_col, group_keys=False)
    ])
    methods = {
        'Population':    df[class_col].value_counts(normalize=True),
        'Simple Random': df.sample(n=sample_n, random_state=42)[class_col].value_counts(normalize=True),
        'Systematic':    df.iloc[::k][class_col].value_counts(normalize=True),
        'Stratified':    stratified[class_col].value_counts(normalize=True),
    }
    labels = {0: 'Clean (Good)', 1: 'Polluted'}
    colors = {0: '#3fb950', 1: '#f78166'}
    fig = go.Figure()
    for cls_val in [0, 1]:
        vals = [m.get(cls_val, 0) for m in methods.values()]
        fig.add_trace(go.Bar(name=labels[cls_val], x=list(methods.keys()),
                             y=vals, marker_color=colors[cls_val], opacity=0.85))
    fig.update_layout(**theme, barmode='group',
                      title=dict(text="Sampling Comparison — Clean vs Polluted Cities",
                                 font=dict(size=15, family="Space Mono")),
                      yaxis_title="Proportion", xaxis_title="Sampling Method")
    return fig


def build_clustering(df, dark):
    theme = plotly_theme(dark)
    features = ['AQI Value','CO AQI Value','Ozone AQI Value',
                'NO2 AQI Value','PM2.5 AQI Value']
    X = df[features].fillna(0).values
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    fig_elbow = go.Figure(go.Scatter(
        x=list(range(1, 11)), y=wcss, mode='lines+markers',
        marker=dict(color='#58a6ff', size=8),
        line=dict(color='#58a6ff', width=2)))
    fig_elbow.update_layout(**theme,
                            title=dict(text="Elbow Method (WCSS vs K)",
                                       font=dict(size=15, family="Space Mono")),
                            xaxis_title="K", yaxis_title="WCSS")

    kmeans   = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    clabels = {0:'Cluster 0 — Clean', 1:'Cluster 1 — Moderate',
               2:'Cluster 2 — Polluted'}
    ccolors = {0:'#3fb950', 1:'#ffa657', 2:'#f78166'}

    fig_scatter = go.Figure()
    for c in [0, 1, 2]:
        mask = clusters == c
        fig_scatter.add_trace(go.Scatter(
            x=X_pca[mask, 0], y=X_pca[mask, 1], mode='markers',
            name=clabels[c],
            marker=dict(color=ccolors[c], size=5, opacity=0.6,
                        line=dict(color='#fff', width=0.2))))
    centers_pca = pca.transform(kmeans.cluster_centers_)
    fig_scatter.add_trace(go.Scatter(
        x=centers_pca[:, 0], y=centers_pca[:, 1], mode='markers',
        marker=dict(symbol='x', size=16, color='white', line=dict(width=2.5)),
        name='Centroids'))
    fig_scatter.update_layout(**theme,
                              title=dict(text="K-Means Clusters (PCA Space)",
                                         font=dict(size=15, family="Space Mono")),
                              xaxis_title="PCA Component 1",
                              yaxis_title="PCA Component 2")
    return fig_elbow, fig_scatter


def build_probability_distributions(df, dark):
    theme = plotly_theme(dark)
    p_good = (df['AQI Category'] == 'Good').mean()
    mu, sigma = df['AQI Value'].mean(), df['AQI Value'].std()
    n_b, p_b = 20, p_good
    lam = df['NO2 AQI Value'].mean()

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Bernoulli (Good AQI)', 'Normal (AQI Value)',
                                        'Binomial (Good Cities in 20)', 'Poisson (NO2 AQI)'),
                        horizontal_spacing=0.12, vertical_spacing=0.18)

    # Bernoulli
    fig.add_trace(go.Bar(x=['Clean/Good (0)', 'Polluted (1)'],
                          y=[1 - p_good, p_good],
                          marker_color=['#3fb950', '#f78166'],
                          showlegend=False), row=1, col=1)
    # Normal
    x_norm = np.linspace(max(0, mu - 4*sigma), mu + 4*sigma, 300)
    fig.add_trace(go.Scatter(x=x_norm, y=stats.norm.pdf(x_norm, mu, sigma),
                              fill='tozeroy', fillcolor='rgba(88,166,255,0.2)',
                              line=dict(color='#58a6ff', width=2),
                              showlegend=False), row=1, col=2)
    # Binomial
    x_b = np.arange(0, n_b + 1)
    fig.add_trace(go.Bar(x=x_b, y=stats.binom.pmf(x_b, n_b, p_b),
                          marker_color='#3fb950', showlegend=False), row=2, col=1)
    # Poisson
    x_p = np.arange(0, 20)
    fig.add_trace(go.Bar(x=x_p, y=stats.poisson.pmf(x_p, lam),
                          marker_color='#d2a8ff', showlegend=False), row=2, col=2)

    fig.update_layout(**theme, height=500,
                      title=dict(text="Probability Distributions — Air Quality Dataset",
                                 font=dict(size=15, family="Space Mono")))
    return fig


def build_stat_props(df, col, dark):
    theme = plotly_theme(dark)
    vals = df[col].dropna()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'Histogram of {col}', 'Q-Q Plot'),
                        horizontal_spacing=0.12)
    fig.add_trace(go.Histogram(x=vals, nbinsx=40, marker_color='#58a6ff',
                                opacity=0.85, showlegend=False), row=1, col=1)
    qq = stats.probplot(vals)
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                              marker=dict(color='#3fb950', size=4, opacity=0.6),
                              showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=qq[0][0],
                              y=qq[1][1] + qq[1][0] * qq[0][0],
                              mode='lines', line=dict(color='#f78166', width=2),
                              showlegend=False), row=1, col=2)
    fig.update_layout(**theme,
                      title=dict(text=f"Distribution Analysis — {col}",
                                 font=dict(size=15, family="Space Mono")))
    return fig


def build_inference(df, dark):
    theme = plotly_theme(dark)
    feat   = 'AQI Value'
    sample = df[feat].dropna()
    n, mean, std_ = len(sample), sample.mean(), sample.std()
    se = std_ / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)

    clean    = df[df['Is_Polluted'] == 0][feat].dropna()
    polluted = df[df['Is_Polluted'] == 1][feat].dropna()

    fig_dist = go.Figure()
    for grp, color, label in [(clean, '#3fb950', 'Clean (Good)'),
                               (polluted, '#f78166', 'Polluted')]:
        fig_dist.add_trace(go.Histogram(x=grp, nbinsx=40, name=label,
                                         marker_color=color, opacity=0.6))
    fig_dist.update_layout(**theme, barmode='overlay',
                            title=dict(text="AQI Distribution — Clean vs Polluted",
                                       font=dict(size=15, family="Space Mono")),
                            xaxis_title="AQI Value", yaxis_title="Frequency")

    x_range = np.linspace(mean - 5*se, mean + 5*se, 300)
    y_pdf   = stats.norm.pdf(x_range, mean, se)
    ci_mask = (x_range >= ci[0]) & (x_range <= ci[1])
    fig_ci  = go.Figure()
    fig_ci.add_trace(go.Scatter(x=x_range, y=y_pdf, mode='lines',
                                 line=dict(color='#58a6ff', width=2),
                                 name='Sampling Distribution'))
    fig_ci.add_trace(go.Scatter(x=x_range[ci_mask], y=y_pdf[ci_mask],
                                 fill='tozeroy', fillcolor='rgba(88,166,255,0.2)',
                                 line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
    fig_ci.add_vline(x=mean, line_dash='dash', line_color='#f78166',
                     annotation_text=f'μ={mean:.2f}')
    fig_ci.update_layout(**theme,
                          title=dict(text=f"95% CI for Mean AQI: [{ci[0]:.2f}, {ci[1]:.2f}]",
                                     font=dict(size=15, family="Space Mono")),
                          xaxis_title="Mean AQI Value", yaxis_title="Density")
    return fig_dist, fig_ci, ci, mean, std_


def build_timeseries(df, dark):
    """df: Country, Mean_AQI — 175 countries sorted by Mean_AQI."""
    theme = plotly_theme(dark)
    ts = df.reset_index(drop=True).copy()
    ts['Rolling_Mean'] = ts['Mean_AQI'].rolling(window=10, center=True).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=ts['Country'], y=ts['Mean_AQI'],
                                    mode='lines', name='Mean AQI',
                                    line=dict(color='#58a6ff', width=1.5), opacity=0.8))
    fig_trend.add_trace(go.Scatter(x=ts['Country'], y=ts['Rolling_Mean'],
                                    mode='lines', name='10-Country Rolling Avg',
                                    line=dict(color='#f78166', width=2.5)))
    fig_trend.update_layout(**theme,
                             title=dict(text="Country Mean AQI — Trend (Sorted by Pollution)",
                                         font=dict(size=15, family="Space Mono")),
                             xaxis_title="Country", yaxis_title="Mean AQI Value")
    fig_trend.update_xaxes(tickangle=-45, tickfont=dict(size=7))

    ts['Band'] = pd.cut(ts['Mean_AQI'],
                        bins=[0, 50, 100, 150, 600],
                        labels=['Good (0–50)', 'Moderate (51–100)',
                                'Unhealthy (101–150)', 'Hazardous (>150)'])
    band_counts = ts['Band'].value_counts().reindex(
        ['Good (0–50)', 'Moderate (51–100)', 'Unhealthy (101–150)', 'Hazardous (>150)'],
        fill_value=0)
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Bar(
        x=band_counts.index, y=band_counts.values,
        marker_color=['#3fb950', '#ffa657', '#f78166', '#cc0000'],
        opacity=0.9))
    fig_seasonal.add_hline(y=band_counts.mean(), line_dash='dash',
                            line_color='#d2a8ff',
                            annotation_text=f'Mean: {band_counts.mean():.0f} countries/band')
    fig_seasonal.update_layout(**theme,
                                title=dict(text="Countries per AQI Pollution Band",
                                           font=dict(size=15, family="Space Mono")),
                                xaxis_title="Pollution Band",
                                yaxis_title="Number of Countries")

    from statsmodels_fallback import simple_forecast
    forecast_vals, _ = simple_forecast(ts['Mean_AQI'].values, 24)
    forecast_vals = np.clip(forecast_vals, 0, 600)
    future_x = [f"Forecast+{i+1}" for i in range(24)]

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=ts['Country'], y=ts['Mean_AQI'],
                                       mode='lines', name='Historical',
                                       line=dict(color='#58a6ff', width=1.5)))
    fig_forecast.add_trace(go.Scatter(x=future_x, y=forecast_vals,
                                       mode='lines', name='Forecast',
                                       line=dict(color='#f78166', width=2, dash='dash')))
    upper = forecast_vals * 1.1
    lower = forecast_vals * 0.9
    fig_forecast.add_trace(go.Scatter(
        x=future_x + future_x[::-1],
        y=list(upper) + list(lower[::-1]),
        fill='toself', fillcolor='rgba(247,129,102,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='±10% Confidence Band'))
    fig_forecast.update_layout(**theme,
                                title=dict(text="24-Step AQI Forecast",
                                           font=dict(size=15, family="Space Mono")),
                                xaxis_title="Country / Forecast Step",
                                yaxis_title="Mean AQI")
    fig_forecast.update_xaxes(tickangle=-45, tickfont=dict(size=7))
    return fig_trend, fig_seasonal, fig_forecast


# ── UI helpers ────────────────────────────────────────────────────────────────

def banner(icon, num, title):
    st.markdown(f"""
    <div class="exp-banner">
      <div class="banner-icon">{icon}</div>
      <div>
        <div class="banner-num">EXPERIMENT {num}</div>
        <div class="banner-title">{title}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box"><p>💡 {text}</p></div>', unsafe_allow_html=True)

def steps(items):
    html = ""
    for i, item in enumerate(items, 1):
        html += (f'<div class="step-item"><div class="step-num">{i}</div>'
                 f'<div class="step-text">{item}</div></div>')
    st.markdown(html, unsafe_allow_html=True)

def conclusion(text):
    st.markdown(f"""
    <div class="conclusion-card">
      <h4>✅ Conclusion</h4>
      <p>{text}</p>
    </div>""", unsafe_allow_html=True)
