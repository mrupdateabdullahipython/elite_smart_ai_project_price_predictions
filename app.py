import base64
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------- PAGE CONFIG ---------------- #
st.markdown("<h1 style='text-align: center;'>EASY BUSINESS TECHNOLOGY NIGERIA LTD</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 14px;'>Home of Project,Research, and Mentorship</p>", 
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 14px;'>Developed An</p>", 
    unsafe_allow_html=True)
st.set_page_config(
    page_title="Elite AI Project Price Estimator", page_icon="💼", layout="wide"
)


# ---------------- LOAD BACKGROUND IMAGE ---------------- #


def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()

    return base64.b64encode(data).decode()

bg_image = get_base64("background.jpg")

# ---------------- ADVANCED RESPONSIVE CSS ---------------- #

page_bg = f"""
<style>

[data-testid="stAppViewContainer"]{{
background-image:url("data:image/jpg;base64,{bg_image}");
background-size:cover;
background-position:center;
background-attachment:fixed;
}}

[data-testid="stHeader"]{{
background:rgba(0,0,0,0);
}}

[data-testid="stSidebar"]{{
background:rgba(0,0,0,0.65);
}}

/* Babban Title: Yana daidaita girman kansa dangane da girman allo */
.main-title{{
font-size: 3.5rem;
font-weight:bold;
text-align:center;
color:white;
margin-top:20px;
text-shadow:2px 2px 20px black;
line-height: 1.2;
}}

.sub-title{{
font-size: 1.4rem;
text-align:center;
color:#f1f1f1;
margin-bottom:30px;
}}

/* Gidan da ke riƙe da Metrics (Flexbox Grid) */
.metrics-container {{
display: flex;
flex-wrap: wrap;
gap: 20px;
justify-content: center;
width: 100%;
margin-bottom: 30px;
}}

/* Tsarin katin da gane girman allo da kansa */
.glass-card{{
flex: 1 1 calc(25% - 20px); /* A kwamfuta zai ɗauki kashi 25% (guda 4 a jere) */
min-width: 220px;
background:rgba(255,255,255,0.12);
padding:25px;
border-radius:20px;
backdrop-filter:blur(12px);
-webkit-backdrop-filter:blur(12px);
box-shadow:0px 8px 32px rgba(0,0,0,0.4);
transition: 0.3s ease-in-out;
display: flex;
flex-direction: column;
justify-content: center;
align-items: center;
text-align: center;
box-sizing: border-box;
}}

.glass-card:hover{{
transform: translateY(-5px);
background:rgba(255,255,255,0.18);
}}

.metric-number{{
font-size: 2.2rem;
font-weight:bold;
color:#00ffd5;
line-height: 1.1;
}}

.metric-text{{
font-size: 1rem;
color:white;
margin-top: 8px;
font-weight: 500;
}}

.prediction-box{{
background:linear-gradient(135deg, #00c6ff, #0072ff);
padding:30px;
border-radius:25px;
text-align:center;
color:white;
font-size: 2.5rem;
font-weight:bold;
box-shadow:0px 8px 25px rgba(0,0,0,0.5);
margin-top:20px;
line-height: 1.3;
}}

.insight-box{{
background:rgba(0,0,0,0.55);
padding:20px;
border-radius:20px;
color:white;
font-size:1.1rem;
margin-top:20px;
line-height: 1.6;
}}

/* ---------------- MEDIA QUERIES (Wurin da ke gyara Waya da Tablet) ---------------- */

/* Don Wayoyin Hannu (Mobile Phones) */
@media screen and (max-width: 600px) {{
    .main-title {{
        font-size: 2rem !important; /* Rage girman rubutun babban title a waya */
    }}
    .sub-title {{
        font-size: 1rem !important;
    }}
    .glass-card {{
        flex: 1 1 100% !important; /* Kati guda ɗaya kacal a jere domin ya faɗaɗa da kyau */
        padding: 20px;
    }}
    .metric-number {{
        font-size: 1.8rem !important;
    }}
    .prediction-box {{
        font-size: 1.8rem !important;
        padding: 20px;
    }}
}}

/* Don Allon Tsaka-tsaki (Tablets) */
@media screen and (min-width: 601px) and (max-width: 1024px) {{
    .glass-card {{
        flex: 1 1 calc(50% - 20px) !important; /* Kati guda biyu a jere */
    }}
    .main-title {{
        font-size: 2.8rem !important;
    }}
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("project_price_model.pkl")
training_columns = joblib.load("training_columns.pkl")

# ---------------- TITLE ---------------- #

st.markdown(
    "<div class='main-title'>💼 ELITE AI PROJECT PRICE ESTIMATOR</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-title'>AI-Powered Software Project Cost Prediction Dashboard</div>",
    unsafe_allow_html=True,
)
# ---------------- SIDEBAR ---------------- #
st.sidebar.title('Welcome to EBT Select Your Desired Project Topic')
st.sidebar.title("⚙ Project Details")

project_type = st.sidebar.selectbox(
    "Project Type",
    ["Website", "ML Project", "Dashboard", "Mobile App", "Research Project"],
)
complexity = st.sidebar.selectbox("Complexity", ["Low", "Medium", "High"])
estimated_days = st.sidebar.slider("Estimated Days", 5, 60, 20)
ui_quality = st.sidebar.selectbox("UI Quality", ["Basic", "Premium", "CEO-Level"])
database = st.sidebar.selectbox("Database", ["Yes", "No"])
authentication = st.sidebar.selectbox("Authentication", ["Yes", "No"])
api_integration = st.sidebar.selectbox("API Integration", ["Yes", "No"])
deployment = st.sidebar.selectbox("Deployment", ["Yes", "No"])
st.sidebar.caption("💼 Elite AI Project Price Estimator is a premium Machine Learning web application designed to predict software project prices using Linear Regression. The system analyzes important project features such as project type, complexity, estimated development days, UI quality, database integration, authentication systems, API integration, and deployment requirements to generate intelligent project cost estimations.")
st.sidebar.markdown(
    "<p style='text-align: center; color: green; font-size: 14px;'>Created by updateabdullahi</p>", 
    unsafe_allow_html=True)
# ---------------- NEW DATA ---------------- #

new_data = pd.DataFrame(
    {
        "Project_Type": [project_type],
        "Complexity": [complexity],
        "Estimated_Days": [estimated_days],
        "UI_Quality": [ui_quality],
        "Database": [database],
        "Authentication": [authentication],
        "API_Integration": [api_integration],
        "Deployment": [deployment],
    }
)

# ---------------- ENCODING ---------------- #

new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=training_columns, fill_value=0)

# ---------------- PREDICTION BUTTON ---------------- #

if st.button("🚀 Generate Project Price Estimation"):
    with st.spinner("Analyzing Project Complexity..."):
        time.sleep(3)

    prediction = model.predict(new_data)[0]
    prediction = round(prediction)

    if prediction <= 20000:
        category = "💡 Basic Project"
    elif prediction <= 35000:
        category = "🔥 Premium Project"
    else:
        category = "👑 CEO-Level Project"

    st.markdown(
        f"""
    <div class='prediction-box'>
    Estimated Project Price <br>
    ₦{prediction:,}
    <br><br>
    {category}
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class='insight-box'>
    🤖 AI BUSINESS INSIGHTS <br><br>
    This project requires <b>{complexity}</b> development complexity with <b>{ui_quality}</b> interface quality.<br><br>
    Estimated development duration: <b>{estimated_days} days</b><br><br>
    AI analysis indicates this project has strong professional implementation requirements.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---------------- METRICS (Responsive Cards) ---------------- #

st.markdown("## 📊 LIVE BUSINESS METRICS")

st.markdown(
    """
<div class="metrics-container">
    <div class="glass-card">
        <div class="metric-number">500+</div>
        <div class="metric-text">Projects Completed</div>
    </div>
    <div class="glass-card">
        <div class="metric-number">96%</div>
        <div class="metric-text">AI Accuracy</div>
    </div>
    <div class="glass-card">
        <div class="metric-number">98%</div>
        <div class="metric-text">Client Satisfaction</div>
    </div>
    <div class="glass-card">
        <div class="metric-number">₦5M+</div>
        <div class="metric-text">Revenue Insights</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- CHART 1: SMOOTH CURVE AREA CHART ---------------- #

st.markdown("## 📈 PROJECT ANALYTICS")

categories = ["Estimated Days", "Complexity", "UI Level", "Features"]
values = [
    estimated_days,
    10 if complexity == "High" else 6 if complexity == "Medium" else 3,
    10 if ui_quality == "CEO-Level" else 7 if ui_quality == "Premium" else 4,
    sum(
        [
            database == "Yes",
            authentication == "Yes",
            api_integration == "Yes",
            deployment == "Yes",
        ]
    )
    * 3,
]

fig_curve = go.Figure()

fig_curve.add_trace(
    go.Scatter(
        x=categories,
        y=values,
        mode="lines+markers",
        line=dict(shape="spline", width=4, color="#00ffd5"),
        marker=dict(size=8, color="#0072ff"),
        fill="tozeroy",
        fillcolor="rgba(0, 255, 213, 0.15)",
    )
)

fig_curve.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
)

# use_container_width=True yana sa jadawalin ya zama responsive a kowace na'ura
st.plotly_chart(fig_curve, use_container_width=True)


# ---------------- CHART 2: PLOTLY DONUT CHART ---------------- #

st.markdown("## 🍩 FEATURE DISTRIBUTION DYNAMICS")

features = [
    database == "Yes",
    authentication == "Yes",
    api_integration == "Yes",
    deployment == "Yes",
]
labels = ["Database", "Authentication", "API", "Deployment"]
intensity_values = [1 if x else 0.2 for x in features]

feature_data = pd.DataFrame({"Labels": labels, "Intensity": intensity_values})

fig_donut = px.pie(
    feature_data,
    values="Intensity",
    names="Labels",
    hole=0.6,
    template="plotly_dark",
)

fig_donut.update_traces(
    marker=dict(colors=["#00ffd5", "#00c6ff", "#0072ff", "#3a86ff"]),
    hoverinfo="label+percent",
    textinfo="none",
)

fig_donut.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=True,
    legend=dict(font=dict(color="white"), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
)

st.plotly_chart(fig_donut, use_container_width=True)


# ---------------- FOOTER ---------------- #
st.markdown(
    """
<center style='color:white;font-size:16px;'>
🚀 Developed with Python, Machine Learning & Streamlit
<br><br>
💼 Elite AI Project Price Estimations Dashboard
</center>
""",
    unsafe_allow_html=True,
)
st.caption("The dashboard includes futuristic business UI design, interactive analytics, AI-powered insights, modern charts, animated business metrics, glassmorphism effects, and responsive layouts optimized for freelancers, software agencies, and portfolio presentations.")
st.caption("Built with Python, Streamlit, Scikit-learn, Pandas, and Machine Learning technologies, this project demonstrates real-world AI business applications and professional dashboard engineering.")

st.caption('Follow for more Projects')
st.caption('Tiktok: @updatecodesml')
st.caption('Facebook: Update Codes ML')
st.caption('Youtube: UpdateCodesML')
st.caption('Telegram: UpdateCodesML')
st.markdown(
    "<p style='text-align: center; color: green; font-size: 14px;'>All right Reserved 2026 @updateabdullahi</p>", 
    unsafe_allow_html=True)