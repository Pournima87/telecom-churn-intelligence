from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load model and features
model = joblib.load("models/churn_model.pkl")
features = joblib.load("models/model_features.pkl")

app = Flask(__name__)


@app.route("/")
def home():
    return {"message": "Telecom Churn Prediction API Running"}


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_df = pd.DataFrame(columns=features)
    input_df.loc[0] = 0

    # Basic inputs
    input_df["tenure"] = data["tenure"]
    input_df["MonthlyCharges"] = data["MonthlyCharges"]

    # Contract
    if data["Contract"] == "One year":
        input_df["Contract_One year"] = 1

    if data["Contract"] == "Two year":
        input_df["Contract_Two year"] = 1

    # Internet service
    if data["InternetService"] == "Fiber optic":
        input_df["InternetService_Fiber optic"] = 1

    # Payment method
    if data["PaymentMethod"] == "Electronic check":
        input_df["PaymentMethod_Electronic check"] = 1

    # Tech support
    if data["TechSupport"] == "Yes":
        input_df["TechSupport_Yes"] = 1

    # Online security
    if data["OnlineSecurity"] == "Yes":
        input_df["OnlineSecurity_Yes"] = 1

    # Prediction
    probability = model.predict_proba(input_df)[0][1]

    if probability < 0.3:
        risk = "Low Risk"
        action = "No retention action needed"

    elif probability < 0.6:
        risk = "Medium Risk"
        action = "Offer engagement incentives"

    else:
        risk = "High Risk"
        action = "Immediate retention campaign"

    prediction = 1 if probability > 0.5 else 0

    return jsonify({
    "prediction": int(prediction),
    "churn_probability": round(float(probability),3),
    "churn_probability_percent": f"{probability*100:.2f}%",
    "risk_level": risk,
    "recommended_action": action
})


if __name__ == "__main__":
    app.run(debug=True)