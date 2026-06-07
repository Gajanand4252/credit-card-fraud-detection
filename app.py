from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline model
model = joblib.load("fraud_detection_pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    result = None
    probability = None
    error = None

    # Default values
    form_data = {
        "type": "PAYMENT",
        "amount": 1000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

    if request.method == "POST":
        try:
            form_data = {
                "type": request.form["type"],
                "amount": float(request.form["amount"]),
                "oldbalanceOrg": float(request.form["oldbalanceOrg"]),
                "newbalanceOrig": float(request.form["newbalanceOrig"]),
                "oldbalanceDest": float(request.form["oldbalanceDest"]),
                "newbalanceDest": float(request.form["newbalanceDest"]),
            }

            input_df = pd.DataFrame([form_data])

            prediction = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_df)[0][1]
                probability = round(probability * 100, 2)

            if prediction == 1:
                result = "FRAUDULENT TRANSACTION"
            else:
                result = "NON-FRAUDULENT TRANSACTION"

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        result=result,
        probability=probability,
        form_data=form_data,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)