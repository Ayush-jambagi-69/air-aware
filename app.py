from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)
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



if __name__ == '__main__':
    # Render automatically injects the correct port into the environment variables
    port = int(os.environ.get("PORT", 5000))
    
    # Run the application with production configurations
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=False  # Crucial: Always disable debug mode on live cloud servers
    )