from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("aqi_xgboost.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    sample = pd.DataFrame([{
        "pm10": data["pm10"],
        "pm2_5": data["pm2_5"]
    }])

    prediction = model.predict(sample)

    return jsonify({
        "predicted_aqi": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )