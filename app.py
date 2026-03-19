
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Telecom Customer Intelligence Platform",
    layout="wide",
    page_icon="📡"
)

# -------------------------------------------------
# HEADER STYLE
# -------------------------------------------------

st.markdown("""
<style>

.main-header {
    background: linear-gradient(90deg, #1f4e79, #3a7bd5);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.main-header h1 {
    color: white;
    margin: 0;
}

.kpi-card {
    background-color: #f7f9fc;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e1e5ee;
    text-align: center;
}

.kpi-title {
    color: #555;
    font-size: 14px;
}

.kpi-value {
    font-size: 30px;
    font-weight: bold;
    color: #111;   /* FIX */
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/Telco-Customer-Churn.csv")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna()

    return df

df = load_data()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

def load_model():

    model = joblib.load("models/churn_pipeline.pkl")

    return model

model = load_model()

# -------------------------------------------------
# API FUNCTION
# -------------------------------------------------

def predict_churn(data):
    
    # Convert user input to dataframe
    input_df = pd.DataFrame([data])

    # Predict churn probability
    probability = model.predict_proba(input_df)[0][1]

    return probability

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Churn Prediction",
        "Customer Segmentation",
        "Customer Risk Ranking",
        "Dataset Explorer",
        "Project Info"
    ]
)

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

if page == "Dashboard":

    st.markdown("""
    <div class="main-header">
        <h1>Telecom Customer Intelligence Platform</h1>
        <p>AI-powered churn prediction and retention analytics</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    total_customers = len(df)
    churn_rate = (df["Churn"].value_counts(normalize=True)["Yes"]) * 100
    avg_charge = df["MonthlyCharges"].mean()

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-title">Total Customers</div>
        <div class="kpi-value">{total_customers}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-title">Churn Rate</div>
        <div class="kpi-value">{churn_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-title">Average Monthly Charges</div>
        <div class="kpi-value">${avg_charge:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.header("Customer Churn Insights")

    st.divider()

    col1, col2 = st.columns(2)

# -----------------------------
# Chart 1: Churn Distribution
# -----------------------------

    with col1:

        fig, ax = plt.subplots()

        sns.countplot(data=df, x="Churn", palette="Set2", ax=ax)

        ax.set_title("Customer Churn Distribution")
        ax.set_xlabel("Churn Status")
        ax.set_ylabel("Number of Customers")

        st.pyplot(fig)

# -----------------------------
# Chart 2: Contract Churn
# -----------------------------

    with col2:

        contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100

        fig, ax = plt.subplots()

        contract.plot(kind="bar", stacked=True, colormap="coolwarm", ax=ax)

        ax.set_title("Churn Rate by Contract Type")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Contract Type")

        st.pyplot(fig)


# SECOND ROW
    col3, col4 = st.columns(2)

# -----------------------------
# Chart 3: Internet Service
# -----------------------------

    with col3:

        internet = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100

        fig, ax = plt.subplots()

        internet.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)

        ax.set_title("Churn Rate by Internet Service")
        ax.set_ylabel("Percentage (%)")

        st.pyplot(fig)

# -----------------------------
# Chart 4: Payment Method
# -----------------------------

    with col4:

        payment = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100

        fig, ax = plt.subplots()

        payment.plot(kind="bar", stacked=True, colormap="magma", ax=ax)

        ax.set_title("Churn Rate by Payment Method")
        ax.set_ylabel("Percentage (%)")

        st.pyplot(fig)

    st.divider()

    st.subheader("Key Business Insights")

    churn_rate = (df["Churn"] == "Yes").mean() * 100

    month_contract_churn = (
        df[df["Contract"] == "Month-to-month"]["Churn"]
        .value_counts(normalize=True)["Yes"] * 100
    )

    fiber_churn = (
        df[df["InternetService"] == "Fiber optic"]["Churn"]
        .value_counts(normalize=True)["Yes"] * 100
 )

    new_customer_churn = (
        df[df["tenure"] <= 12]["Churn"]
        .value_counts(normalize=True)["Yes"] * 100
    )

    st.info(f"""
Key findings from the dataset:

• Overall churn rate is **{churn_rate:.1f}%**

• Customers with **month-to-month contracts churn at {month_contract_churn:.1f}%**

• **Fiber optic users churn at {fiber_churn:.1f}%**

• Customers in their **first year churn at {new_customer_churn:.1f}%**
""")

# -------------------------------------------------
# CHURN PREDICTION
# -------------------------------------------------

elif page == "Churn Prediction":

    st.title("AI Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.slider("Tenure", 0, 72, 12)

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month","One year","Two year"]
        )

        gender = st.selectbox("Gender", ["Male","Female"])

        device_protection = st.selectbox("Device Protection", ["Yes","No","No internet service"])

        streaming_tv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])

        streaming_movies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

        paperless_billing = st.selectbox("Paperless Billing", ["Yes","No"])

        total_charges = st.number_input("Total Charges", min_value=0.0)

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL","Fiber optic","No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes","No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes","No"]
        )
        senior_citizen = st.selectbox("Senior Citizen", [0,1])

        partner = st.selectbox("Partner", ["Yes","No"])

        dependents = st.selectbox("Dependents", ["Yes","No"])

        phone_service = st.selectbox("Phone Service", ["Yes","No"])

        multiple_lines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])

        online_backup = st.selectbox("Online Backup", ["Yes","No","No internet service"])

    

    if st.button("Predict Churn Risk"):
        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

    # -------- Feature engineering (same as notebook) --------
        input_df["AvgMonthlySpend"] = input_df["MonthlyCharges"]
        input_df["CLV"] = input_df["MonthlyCharges"] * input_df["tenure"]
        input_df["HasInternet"] = (input_df["InternetService"] != "No").astype(int)

        services = [
            "OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies"
        ]

        input_df["ServiceCount"] = input_df[services].apply(lambda x: (x=="Yes").sum(), axis=1)

    # Tenure groups
        bins = [0,12,24,48,100]
        labels = ["0-12 Months","12-24 Months","24-48 Months","Over 48 Months"]
        input_df["TenureGroup"] = pd.cut(input_df["tenure"], bins=bins, labels=labels)

    # -------- Encoding --------
        input_encoded = pd.get_dummies(input_df)

    # Align with training columns
        input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # -------- Prediction --------
        probability = model.predict_proba(input_encoded)[0][1]

        st.subheader("AI Risk Score")

        st.progress(min(float(probability), 1.0))
        st.metric("Churn Probability", f"{probability*100:.2f}%")


        if probability < 0.25:

            risk = "Low Risk"

            st.success("🟢 Low Risk")

        elif probability < 0.47:

            risk = "Medium Risk"

            st.warning("🟡 Medium Risk")

        else:

            risk = "High Risk"

            st.error("🔴 High Risk")

        st.divider()

        st.subheader("Customer Risk Profile")

        c1, c2, c3 = st.columns(3)

        c1.metric("Tenure", f"{tenure} months")
        c2.metric("Monthly Charges", f"${monthly_charges:,.2f}")
        c3.metric("Contract", contract)

        st.divider()

        st.subheader("Recommended Retention Strategy")

        strategies = []

        if contract == "Month-to-month":
            strategies.append("Offer discount for yearly contract")

        if payment_method == "Electronic check":
            strategies.append("Encourage automatic payment")

        if tech_support == "No":
            strategies.append("Offer free tech support trial")

        if monthly_charges > 80:
            strategies.append("Provide loyalty discount")

        if strategies:
            for s in strategies:
                st.write(f"• {s}")
        else:
            st.write("• Maintain current customer engagement strategy")

        st.divider()

        st.subheader("Business Impact Estimation")

        annual_revenue = monthly_charges * 12

        saved_revenue = annual_revenue * 0.6

        col1, col2, col3 = st.columns(3)

        col1.metric("Monthly Revenue", f"${monthly_charges:,.2f}")
        col2.metric("Annual Value", f"${annual_revenue:,.2f}")
        col3.metric("Potential Revenue Saved", f"${saved_revenue:,.2f}")

# -------------------------------------------------
# CUSTOMER SEGMENTATION (IMPROVED)
# -------------------------------------------------

elif page == "Customer Segmentation":

    st.title("Customer Segmentation Map")

    features = df[["tenure","MonthlyCharges","TotalCharges"]]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)

    df["Segment"] = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled)

    df["PCA1"] = components[:,0]
    df["PCA2"] = components[:,1]

    fig, ax = plt.subplots(figsize=(8,6))

    scatter = ax.scatter(
        df["PCA1"],
        df["PCA2"],
        c=df["Segment"],
        cmap="viridis",
        alpha=0.7
    )


    legend = ax.legend(
        *scatter.legend_elements(),
        title="Customer Segments"
    )

    ax.add_artist(legend)

    ax.set_title("Customer Segmentation (KMeans)")
    ax.set_title("Customer Segmentation Map")
    ax.set_xlabel("Behavior Component 1")
    ax.set_ylabel("Behavior Component 2")

    st.pyplot(fig)

    st.subheader("Segment Statistics")

    segment_summary = df.groupby("Segment").agg({
        "tenure":"mean",
        "MonthlyCharges":"mean"
    })

    st.dataframe(segment_summary)

    st.info("""
Segment Interpretation

Segment 0 → Long tenure + high spending customers (loyal users)

Segment 1 → New customers with low monthly charges

Segment 2 → Stable moderate value customers

Segment 3 → High spending customers with shorter tenure (higher churn risk)
""")

# -------------------------------------------------
# RISK RANKING
# -------------------------------------------------

elif page == "Customer Risk Ranking":

    st.title("Top Customers Likely To Churn")

    data = df.copy()

    # Remove target and ID for prediction
    features_df = data.drop(columns=["customerID","Churn"])

    # Align with training features
    
    try:
        features_df = features_df.reindex(
        columns=model.named_steps["preprocess"].feature_names_in_,
        fill_value=0
    )
    except:
        pass

    # Predict churn probability
    risk_scores = model.predict_proba(features_df)[:,1]

    data["Risk Score"] = risk_scores

    # Risk categories
    data["Risk Level"] = data["Risk Score"].apply(
        lambda x: "High Risk" if x > 0.45 else
                  "Medium Risk" if x > 0.25 else
                  "Low Risk"
    )

    high_risk_count = (data["Risk Level"] == "High Risk").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(data))
    col2.metric("High Risk Customers", high_risk_count)
    col3.metric("Avg Risk Score", f"{data['Risk Score'].mean():.2f}")

    # Sort by highest risk
    top_risk = data.sort_values("Risk Score", ascending=False).head(20)
    top_risk["Risk Score"] = top_risk["Risk Score"].round(3)

    st.dataframe(
        top_risk[[
            "customerID",
            "tenure",
            "MonthlyCharges",
            "Contract",
            "InternetService",
            "Risk Score",
            "Risk Level"
        ]]
    )

    st.subheader("Top Churn Drivers")

    try:

        lr_model = model.named_steps["model"]

        importance = lr_model.coef_[0]

        feature_names = model.named_steps["preprocess"].get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        importance_df = importance_df.sort_values(
            "Importance", ascending=False
        ).head(10)

        st.bar_chart(importance_df.set_index("Feature"))

    except:
        st.info("Feature importance not available for this model.")

    csv = top_risk.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download High Risk Customers",
        csv,
        "high_risk_customers.csv",
        "text/csv"
    )

# -------------------------------------------------
# DATASET EXPLORER
# -------------------------------------------------

elif page == "Dataset Explorer":

    st.title("Dataset Explorer")

    st.dataframe(df)

    st.subheader("Summary")

    st.write(df.describe())

# -------------------------------------------------
# PROJECT INFO
# -------------------------------------------------

else:

    st.title("Project Information")

    st.write("""
    End-to-End Machine Learning System for Telecom Customer Churn.

    Includes:
    - Data Cleaning & EDA
    - Feature Engineering
    - Customer Segmentation
    - Churn Prediction Model
    - Explainable AI
    - Retention Strategy Engine
    - Business Impact Estimation
    - Flask Prediction API
    - Streamlit SaaS Dashboard
    """)

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "80%")
    col2.metric("ROC-AUC", "0.84")
    col3.metric("F1 Score", "0.63")

st.divider()
st.caption(
"Telecom Customer Intelligence Platform | Built with Python, Scikit-Learn, and Streamlit"
)