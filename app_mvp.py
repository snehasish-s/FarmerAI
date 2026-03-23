import random
from flask import Flask, render_template, request, jsonify
import io
import requests
import os

app = Flask(__name__)

# Free Weather API
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', 'demo')
WEATHER_BASE_URL = 'https://wttr.in/{location}?format=j1'

# Global flags for MVP (set False on prod without TF)
MODELS_AVAILABLE = False  # Change to True when TF installed

# Mock data
class_labels = {0: "Healthy", 1: "Nutrient Deficient", 2: "Leaf Blight", 3: "Rust", 4: "Other Disease"}
irrigation_preproc_mock = {'crop_map': {'wheat': 0, 'rice': 1}, 'soil_map': {'loamy': 1}, 'X_mean': 0, 'X_std': 1}
fertilizer_preproc_mock = {'crop_map': {'wheat': 0}, 'soil_map': {'loamy': 1}, 'X_mean': 0, 'X_std': 1}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/crophealth")
def crophealth_page():
    return render_template("crophealth.html")

@app.route("/yieldprediction")
def yieldprediction_page():
    return render_template("yieldprediction.html")

@app.route("/irrigation")
def irrigation_page():
    lang = request.args.get('lang', 'en')
    return render_template("irrigation.html", lang=lang)

@app.route("/fertilizer")
def fertilizer_page():
    return render_template("fertilizer.html")

@app.route("/pestrisk")
def pestrisk_page():
    return render_template("pestrisk.html")

@app.route("/soil")
def soil_page():
    return render_template("soil.html")

@app.route("/get_weather", methods=["GET"])
def get_weather():
    try:
        location = request.args.get('location', 'Bhubaneswar')
        url = WEATHER_BASE_URL.format(location=location.replace(' ', '%20'))
        response = requests.get(url, timeout=10)
        data = response.json()
        current = data.get('currentCondition', {})
        weather = {
            'temperature': float(current.get('temp_C', 25)),
            'humidity': float(current.get('humidity', 70)),
            'rainfall': float(data['weather'][0].get('precipMM', 0)),
            'description': current.get('weatherDesc', [{}])[0].get('value', 'Clear'),
            'location': location
        }
        return jsonify(weather)
    except:
        return jsonify({'temperature': 25, 'humidity': 70, 'rainfall': 0})

@app.route("/get_crop_prices", methods=["GET"])
def get_crop_prices():
    prices = [{"state": "Punjab", "crops": [{"name": "Rice", "price": 2450, "unit": "Rs/Qtl"}]}]  # Shortened
    return jsonify(prices)

@app.route("/predict_model1", methods=["POST"])
def predict_model1():
    if not MODELS_AVAILABLE:
        return jsonify({"label": "Healthy", "status": "Healthy", "accuracy": "85.00%", "mode": "mock"})
    # Original TF code...
    return jsonify({"error": "ML disabled (lite deploy)"})

@app.route("/predict_yield", methods=["POST"])
def predict_yield():
    if not MODELS_AVAILABLE:
        yield_val = 3500 + random.randint(-500, 500)
        return jsonify({"predicted_yield": f"{yield_val:.2f} kg/ha", "mode": "mock"})
    # Original
    return jsonify({"error": "ML disabled"})

@app.route("/predict_model2", methods=["POST"])
def predict_model2():
    data = request.get_json() or {'rainfall': 0, 'temperature': 25, 'humidity': 70}
    if not MODELS_AVAILABLE:
        yield_val = 3500 + data['temperature'] * 20 - data['rainfall'] * 5 + random.randint(-200, 200)
        return jsonify({"yield": f"{max(0, yield_val):.2f} kg/ha", "used_weather": True, "mode": "mock"})
    # Original
    return jsonify({"error": "ML disabled"})

@app.route("/predict_irrigation", methods=["POST"])
def predict_irrigation():
    data = request.form
    area = float(data.get('area', 10))
    if not MODELS_AVAILABLE:
        water = 4.5 * area
        freq = 3
        return jsonify({"water_required": int(water*1000), "irrigation_frequency": freq, "schedule": [{"day": "Mon", "amount": "1500 liters"}], "mode": "mock"})
    # Original
    return jsonify({"error": "ML disabled"})

@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    if not MODELS_AVAILABLE:
        return jsonify({"n_kg_ha": 120, "p_kg_ha": 60, "k_kg_ha": 40, "mode": "mock"})
    # Original
    return jsonify({"error": "ML disabled"})

@app.route("/predict_pest", methods=["POST"])
def predict_pest():
    data = request.form
    temp = float(data.get('temperature', 28))
    risk = min(90, 40 + abs(temp - 25) * 2)
    return jsonify({"aphids_risk": risk, "fungal_risk": 30})

@app.route("/predict_soil", methods=["POST"])
def predict_soil():
    data = request.form
    ph = float(data.get('ph', 6.5))
    total_score = 75 - abs(ph - 6.5)*10
    rating = "Good"
    recs = ["Balanced soil"] if 6 < ph < 7.5 else ["Adjust pH"]
    return jsonify({"score": round(total_score), "rating": rating, "recommendations": recs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)

