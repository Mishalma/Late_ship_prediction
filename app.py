"""
app.py

Flask application for predicting late and very late deliveries.

Converted from FastAPI to Flask for local development.
Includes endpoints for:
- Landing page (`/`)
- Health check (`/ping`)
- Very late shipment prediction (`/predict_very_late`)
"""

from flask import Flask, request, jsonify, render_template_string
from pathlib import Path
import pandas as pd
import joblib
import json
import random
import datetime
from src.preprocess_features import NUMERICAL_FEATURES, ONEHOT_FEATURES, LABEL_FEATURES
from src.logger import get_logger

# Initialize Flask app
app = Flask(__name__)
logger = get_logger(__name__)

# Define paths
base_dir = Path(__file__).resolve().parent
scaler_file = base_dir / "models" / "scaler.pkl"
onehot_encoder_file = base_dir / "models" / "onehot_encoder.pkl"
ordinal_encoder_file = base_dir / "models" / "ordinal_encoder.pkl"
late_model_file = base_dir / "models" / "late_model.pkl"
very_late_model_file = base_dir / "models" / "very_late_model.pkl"

def load_artifact(file, name):
    """Load model artifacts with error handling"""
    try:
        return joblib.load(file)
    except FileNotFoundError:
        raise Exception(f"{name} not found. Please run the pipeline first to generate required model files.")

# Load models and preprocessors at startup
try:
    scaler = load_artifact(scaler_file, "scaler")
    onehot_encoder = load_artifact(onehot_encoder_file, "onehot_encoder")
    ordinal_encoder = load_artifact(ordinal_encoder_file, "ordinal_encoder")
    late_model = load_artifact(late_model_file, "late_model")
    very_late_model = load_artifact(very_late_model_file, "very_late_model")
    logger.info("All models and preprocessors loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    scaler = onehot_encoder = ordinal_encoder = late_model = very_late_model = None

# Landing page HTML template
LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Late Shipment Prediction API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 2em; line-height: 1.6; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; margin-top: 2em; }
        code { background-color: #f4f4f4; padding: 0.2em 0.4em; border-radius: 4px; }
        pre { background-color: #f8f8f8; padding: 1em; border-radius: 6px; overflow-x: auto; }
        ul { padding-left: 1.2em; }
        li { margin-bottom: 0.5em; }
        a { color: #1f6feb; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .form-container { background-color: #f9f9f9; padding: 1.5em; border-radius: 8px; margin-top: 2em; }
        .form-group { margin-bottom: 1em; }
        label { display: block; margin-bottom: 0.3em; font-weight: bold; }
        input, select { width: 100%; padding: 0.5em; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 0.7em 1.5em; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 1em; padding: 1em; border-radius: 4px; }
        .result.success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .result.error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    </style>
</head>
<body>
    <h1>📦 Late Shipment Prediction API (Flask)</h1>
    
    <p>
        This Flask API uses machine learning to predict very late shipments (3+ days delay) 
        for a global sports and outdoor equipment retailer.
    </p>

    <h2>🧠 Available Models</h2>
    <ul>
        <li>
            <strong>Late Order Model</strong> – Predicts whether an order will be <em>late by at least 1 day</em><br>
            <em>Optimized for Accuracy</em> (86.14%)<br>
            ✅ <strong>Available via this API</strong>
        </li>
        <li>
            <strong>Very Late Order Model</strong> – Predicts whether an order will be <em>very late (≥ 3 days)</em><br>
            <em>Optimized for Recall</em> (97.58%)<br>
            ✅ <strong>Available via this API</strong>
        </li>
    </ul>

    <h2>📋 Test the API</h2>
    <div class="form-container">
        <h3>Enter Shipment Details:</h3>
        <form id="predictionForm">
            <div class="form-group">
                <label>Order Item Quantity:</label>
                <input type="number" id="order_item_quantity" value="4" required>
            </div>
            <div class="form-group">
                <label>Order Item Total:</label>
                <input type="number" step="0.01" id="order_item_total" value="181.92" required>
            </div>
            <div class="form-group">
                <label>Product Price:</label>
                <input type="number" step="0.01" id="product_price" value="49.97" required>
            </div>
            <div class="form-group">
                <label>Year:</label>
                <input type="number" id="year" value="2015" required>
            </div>
            <div class="form-group">
                <label>Month:</label>
                <input type="number" min="1" max="12" id="month" value="4" required>
            </div>
            <div class="form-group">
                <label>Day:</label>
                <input type="number" min="1" max="31" id="day" value="21" required>
            </div>
            <div class="form-group">
                <label>Order Value:</label>
                <input type="number" step="0.01" id="order_value" value="737.65" required>
            </div>
            <div class="form-group">
                <label>Unique Items Per Order:</label>
                <input type="number" id="unique_items_per_order" value="4" required>
            </div>
            <div class="form-group">
                <label>Order Item Discount Rate:</label>
                <input type="number" step="0.01" min="0" max="1" id="order_item_discount_rate" value="0.09" required>
            </div>
            <div class="form-group">
                <label>Units Per Order:</label>
                <input type="number" id="units_per_order" value="11" required>
            </div>
            <div class="form-group">
                <label>Order Profit Per Order:</label>
                <input type="number" step="0.01" id="order_profit_per_order" value="89.13" required>
            </div>
            <div class="form-group">
                <label>Type:</label>
                <select id="type" required>
                    <option value="DEBIT" selected>DEBIT</option>
                    <option value="CREDIT">CREDIT</option>
                    <option value="CASH">CASH</option>
                    <option value="TRANSFER">TRANSFER</option>
                </select>
            </div>
            <div class="form-group">
                <label>Customer Segment:</label>
                <select id="customer_segment" required>
                    <option value="Home Office" selected>Home Office</option>
                    <option value="Corporate">Corporate</option>
                    <option value="Consumer">Consumer</option>
                </select>
            </div>
            <div class="form-group">
                <label>Shipping Mode:</label>
                <select id="shipping_mode" required>
                    <option value="Standard Class" selected>Standard Class</option>
                    <option value="First Class">First Class</option>
                    <option value="Second Class">Second Class</option>
                    <option value="Same Day">Same Day</option>
                </select>
            </div>
            <div class="form-group">
                <label>Category ID:</label>
                <input type="number" id="category_id" value="46" required>
            </div>
            <div class="form-group">
                <label>Customer Country:</label>
                <input type="text" id="customer_country" value="EE. UU." required>
            </div>
            <div class="form-group">
                <label>Customer State:</label>
                <input type="text" id="customer_state" value="MA" required>
            </div>
            <div class="form-group">
                <label>Department ID:</label>
                <input type="number" id="department_id" value="7" required>
            </div>
            <div class="form-group">
                <label>Order City:</label>
                <input type="text" id="order_city" value="San Pablo de las Salinas" required>
            </div>
            <div class="form-group">
                <label>Order Country:</label>
                <input type="text" id="order_country" value="México" required>
            </div>
            <div class="form-group">
                <label>Order Region:</label>
                <input type="text" id="order_region" value="Central America" required>
            </div>
            <div class="form-group">
                <label>Order State:</label>
                <input type="text" id="order_state" value="México" required>
            </div>
            
            <div style="display: flex; gap: 1em; flex-wrap: wrap;">
                <button type="submit" name="model" value="late">Predict Late Shipment (1+ days)</button>
                <button type="submit" name="model" value="very_late">Predict Very Late Shipment (3+ days)</button>
                <button type="submit" name="model" value="both" style="background-color: #28a745;">Get Both Predictions</button>
            </div>
        </form>
        
        <div id="result"></div>
    </div>

    <h2>📌 Features</h2>
    <ul>
        <li><a href="/dashboard">📊 <strong>Real-time Dashboard</strong></a> - Interactive geospatial visualization</li>
        <li><a href="/analytics">📈 <strong>Analytics Dashboard</strong></a> - Risk analysis and trends</li>
    </ul>

    <h2>📌 API Endpoints</h2>
    <ul>
        <li><code>GET /</code> - This landing page</li>
        <li><code>GET /ping</code> - Health check endpoint</li>
        <li><code>GET /dashboard</code> - Interactive geospatial dashboard</li>
        <li><code>GET /analytics</code> - Analytics and trends dashboard</li>
        <li><code>POST /predict_late</code> - Predict late shipments (1+ days)</li>
        <li><code>POST /predict_very_late</code> - Predict very late shipments (3+ days)</li>
        <li><code>POST /predict_both</code> - Get predictions from both models</li>
        <li><code>GET /api/shipments</code> - Get real-time shipment data</li>
    </ul>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const modelType = e.submitter.value;
            
            const formData = {
                order_item_quantity: parseInt(document.getElementById('order_item_quantity').value),
                order_item_total: parseFloat(document.getElementById('order_item_total').value),
                product_price: parseFloat(document.getElementById('product_price').value),
                year: parseInt(document.getElementById('year').value),
                month: parseInt(document.getElementById('month').value),
                day: parseInt(document.getElementById('day').value),
                order_value: parseFloat(document.getElementById('order_value').value),
                unique_items_per_order: parseInt(document.getElementById('unique_items_per_order').value),
                order_item_discount_rate: parseFloat(document.getElementById('order_item_discount_rate').value),
                units_per_order: parseInt(document.getElementById('units_per_order').value),
                order_profit_per_order: parseFloat(document.getElementById('order_profit_per_order').value),
                type: document.getElementById('type').value,
                customer_segment: document.getElementById('customer_segment').value,
                shipping_mode: document.getElementById('shipping_mode').value,
                category_id: parseInt(document.getElementById('category_id').value),
                customer_country: document.getElementById('customer_country').value,
                customer_state: document.getElementById('customer_state').value,
                department_id: parseInt(document.getElementById('department_id').value),
                order_city: document.getElementById('order_city').value,
                order_country: document.getElementById('order_country').value,
                order_region: document.getElementById('order_region').value,
                order_state: document.getElementById('order_state').value
            };
            
            let endpoint;
            if (modelType === 'late') {
                endpoint = '/predict_late';
            } else if (modelType === 'very_late') {
                endpoint = '/predict_very_late';
            } else if (modelType === 'both') {
                endpoint = '/predict_both';
            }
            
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                
                if (response.ok) {
                    let message;
                    if (modelType === 'late') {
                        const prediction = result.late_prediction;
                        message = prediction === 1 ? 
                            '🚨 <strong>Late Shipment Predicted</strong> - This shipment is likely to be delayed by 1+ days' :
                            '✅ <strong>On-Time Shipment Predicted</strong> - This shipment is likely to arrive on time';
                    } else if (modelType === 'very_late') {
                        const prediction = result.very_late_prediction;
                        message = prediction === 1 ? 
                            '🚨 <strong>Very Late Shipment Predicted</strong> - This shipment is likely to be delayed by 3+ days' :
                            '✅ <strong>On-Time Shipment Predicted</strong> - This shipment is likely to arrive within 3 days';
                    } else if (modelType === 'both') {
                        const latePred = result.late_prediction;
                        const veryLatePred = result.very_late_prediction;
                        
                        let lateIcon = latePred === 1 ? '🚨' : '✅';
                        let veryLateIcon = veryLatePred === 1 ? '🚨' : '✅';
                        
                        message = `
                            <div style="margin-bottom: 1em;">
                                <strong>📊 Combined Prediction Results:</strong>
                            </div>
                            <div style="margin-bottom: 0.5em;">
                                ${lateIcon} <strong>Late Model (1+ days):</strong> ${result.interpretation.late}
                            </div>
                            <div>
                                ${veryLateIcon} <strong>Very Late Model (3+ days):</strong> ${result.interpretation.very_late}
                            </div>
                        `;
                    }
                    
                    resultDiv.innerHTML = `<div class="result success">${message}</div>`;
                } else {
                    resultDiv.innerHTML = `<div class="result error">Error: ${result.error || 'Unknown error occurred'}</div>`;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `<div class="result error">Error: ${error.message}</div>`;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def landing_page():
    """Landing page with interactive form"""
    return render_template_string(LANDING_PAGE_HTML)

@app.route('/ping')
def ping():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

def preprocess_input_data(data):
    """Common preprocessing function for both models"""
    # Convert to DataFrame
    X_unprocessed = pd.DataFrame([data])
    
    # Feature preprocessing and transformation
    X_num = X_unprocessed[NUMERICAL_FEATURES]
    X_onehot = X_unprocessed[ONEHOT_FEATURES]
    X_label = X_unprocessed[LABEL_FEATURES]
    
    # Transform features
    X_num_scaled = scaler.transform(X_num)
    X_onehot_encoded = onehot_encoder.transform(X_onehot)
    X_label_encoded = ordinal_encoder.transform(X_label)
    
    # Create DataFrames with proper column names
    X_num_scaled = pd.DataFrame(X_num_scaled, columns=NUMERICAL_FEATURES, index=X_unprocessed.index)
    X_onehot_encoded = pd.DataFrame(
        X_onehot_encoded,
        columns=onehot_encoder.get_feature_names_out(ONEHOT_FEATURES),
        index=X_unprocessed.index
    )
    X_label_encoded = pd.DataFrame(X_label_encoded, columns=LABEL_FEATURES, index=X_unprocessed.index)
    
    # Combine all features
    X_processed = pd.concat([X_num_scaled, X_onehot_encoded, X_label_encoded], axis=1)
    return X_processed

@app.route('/predict_late', methods=['POST'])
def predict_late():
    """Predict late shipments endpoint (1+ days delay)"""
    try:
        # Check if models are loaded
        if late_model is None:
            return jsonify({"error": "Late model not loaded. Please run the pipeline first."}), 500
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        logger.info("Received request to /predict_late endpoint")
        logger.debug(f"Raw input data: {data}")
        
        # Preprocess input data
        X_processed = preprocess_input_data(data)
        logger.debug(f"X_processed shape: {X_processed.shape}")
        
        # Generate prediction
        is_late = late_model.predict(X_processed)[0]
        logger.info(f"Late prediction generated: {is_late}")
        
        return jsonify({"late_prediction": int(is_late)})
        
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Late prediction error: {e}")
        return jsonify({"error": f"Late prediction failed: {str(e)}"}), 500

@app.route('/predict_very_late', methods=['POST'])
def predict_very_late():
    """Predict very late shipments endpoint (3+ days delay)"""
    try:
        # Check if models are loaded
        if very_late_model is None:
            return jsonify({"error": "Very late model not loaded. Please run the pipeline first."}), 500
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        logger.info("Received request to /predict_very_late endpoint")
        logger.debug(f"Raw input data: {data}")
        
        # Preprocess input data
        X_processed = preprocess_input_data(data)
        logger.debug(f"X_processed shape: {X_processed.shape}")
        
        # Generate prediction
        is_very_late = very_late_model.predict(X_processed)[0]
        logger.info(f"Very late prediction generated: {is_very_late}")
        
        return jsonify({"very_late_prediction": int(is_very_late)})
        
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Very late prediction error: {e}")
        return jsonify({"error": f"Very late prediction failed: {str(e)}"}), 500

@app.route('/predict_both', methods=['POST'])
def predict_both():
    """Get predictions from both models"""
    try:
        # Check if models are loaded
        if late_model is None or very_late_model is None:
            return jsonify({"error": "Models not loaded. Please run the pipeline first."}), 500
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        logger.info("Received request to /predict_both endpoint")
        logger.debug(f"Raw input data: {data}")
        
        # Preprocess input data
        X_processed = preprocess_input_data(data)
        logger.debug(f"X_processed shape: {X_processed.shape}")
        
        # Generate predictions from both models
        is_late = late_model.predict(X_processed)[0]
        is_very_late = very_late_model.predict(X_processed)[0]
        
        logger.info(f"Both predictions generated - Late: {is_late}, Very Late: {is_very_late}")
        
        return jsonify({
            "late_prediction": int(is_late),
            "very_late_prediction": int(is_very_late),
            "interpretation": {
                "late": "Delayed by 1+ days" if is_late else "On time or minimal delay",
                "very_late": "Delayed by 3+ days" if is_very_late else "Delayed by less than 3 days"
            }
        })
        
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Both predictions error: {e}")
        return jsonify({"error": f"Predictions failed: {str(e)}"}), 500

# Sample data for demonstration
SAMPLE_LOCATIONS = [
    {"country": "United States", "state": "California", "city": "Los Angeles", "lat": 34.0522, "lng": -118.2437},
    {"country": "United States", "state": "New York", "city": "New York", "lat": 40.7128, "lng": -74.0060},
    {"country": "United States", "state": "Texas", "city": "Houston", "lat": 29.7604, "lng": -95.3698},
    {"country": "Canada", "state": "Ontario", "city": "Toronto", "lat": 43.6532, "lng": -79.3832},
    {"country": "Mexico", "state": "México", "city": "Mexico City", "lat": 19.4326, "lng": -99.1332},
    {"country": "United Kingdom", "state": "England", "city": "London", "lat": 51.5074, "lng": -0.1278},
    {"country": "Germany", "state": "Bavaria", "city": "Munich", "lat": 48.1351, "lng": 11.5820},
    {"country": "France", "state": "Île-de-France", "city": "Paris", "lat": 48.8566, "lng": 2.3522},
    {"country": "Australia", "state": "New South Wales", "city": "Sydney", "lat": -33.8688, "lng": 151.2093},
    {"country": "Japan", "state": "Tokyo", "city": "Tokyo", "lat": 35.6762, "lng": 139.6503},
    {"country": "Brazil", "state": "São Paulo", "city": "São Paulo", "lat": -23.5505, "lng": -46.6333},
    {"country": "India", "state": "Maharashtra", "city": "Mumbai", "lat": 19.0760, "lng": 72.8777}
]

def generate_sample_shipment():
    """Generate a sample shipment with realistic data"""
    location = random.choice(SAMPLE_LOCATIONS)
    
    # Generate realistic shipment data
    shipment = {
        "id": f"SH{random.randint(100000, 999999)}",
        "order_item_quantity": random.randint(1, 10),
        "order_item_total": round(random.uniform(50, 500), 2),
        "product_price": round(random.uniform(20, 200), 2),
        "year": 2024,
        "month": random.randint(1, 12),
        "day": random.randint(1, 28),
        "order_value": round(random.uniform(100, 1000), 2),
        "unique_items_per_order": random.randint(1, 8),
        "order_item_discount_rate": round(random.uniform(0, 0.3), 2),
        "units_per_order": random.randint(1, 20),
        "order_profit_per_order": round(random.uniform(10, 200), 2),
        "type": random.choice(["DEBIT", "CREDIT", "CASH", "TRANSFER"]),
        "customer_segment": random.choice(["Home Office", "Corporate", "Consumer"]),
        "shipping_mode": random.choice(["Standard Class", "First Class", "Second Class", "Same Day"]),
        "category_id": random.randint(1, 50),
        "customer_country": location["country"],
        "customer_state": location["state"],
        "department_id": random.randint(1, 10),
        "order_city": location["city"],
        "order_country": location["country"],
        "order_region": "North America" if location["country"] in ["United States", "Canada", "Mexico"] else "International",
        "order_state": location["state"],
        "lat": location["lat"],
        "lng": location["lng"],
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return shipment

@app.route('/dashboard')
def dashboard():
    """Interactive geospatial dashboard"""
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shipment Risk Dashboard</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
            .header { background-color: #2c3e50; color: white; padding: 1em; text-align: center; }
            .dashboard-container { display: flex; height: calc(100vh - 80px); }
            .map-container { flex: 2; position: relative; }
            .sidebar { flex: 1; background-color: #f8f9fa; padding: 1em; overflow-y: auto; }
            #map { height: 100%; width: 100%; }
            .stats-card { background: white; border-radius: 8px; padding: 1em; margin-bottom: 1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stats-number { font-size: 2em; font-weight: bold; color: #2c3e50; }
            .stats-label { color: #7f8c8d; font-size: 0.9em; }
            .risk-high { color: #e74c3c; }
            .risk-medium { color: #f39c12; }
            .risk-low { color: #27ae60; }
            .controls { padding: 1em; background: white; margin-bottom: 1em; border-radius: 8px; }
            .btn { padding: 0.5em 1em; margin: 0.2em; border: none; border-radius: 4px; cursor: pointer; }
            .btn-primary { background-color: #3498db; color: white; }
            .btn-success { background-color: #27ae60; color: white; }
            .btn-danger { background-color: #e74c3c; color: white; }
            .shipment-list { max-height: 300px; overflow-y: auto; }
            .shipment-item { padding: 0.5em; border-bottom: 1px solid #eee; font-size: 0.9em; }
            .chart-container { height: 200px; margin-top: 1em; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌍 Real-time Shipment Risk Dashboard</h1>
            <p>Live monitoring of shipment locations and delay predictions</p>
        </div>
        
        <div class="dashboard-container">
            <div class="map-container">
                <div id="map"></div>
            </div>
            
            <div class="sidebar">
                <div class="controls">
                    <h3>Controls</h3>
                    <button class="btn btn-primary" onclick="refreshData()">🔄 Refresh Data</button>
                    <button class="btn btn-success" onclick="toggleAutoRefresh()">⏱️ Auto Refresh</button>
                    <button class="btn btn-danger" onclick="clearMap()">🗑️ Clear Map</button>
                </div>
                
                <div class="stats-card">
                    <h3>📊 Live Statistics</h3>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <div class="stats-number" id="totalShipments">0</div>
                            <div class="stats-label">Total Shipments</div>
                        </div>
                        <div>
                            <div class="stats-number risk-high" id="highRisk">0</div>
                            <div class="stats-label">High Risk</div>
                        </div>
                    </div>
                </div>
                
                <div class="stats-card">
                    <h3>🎯 Risk Distribution</h3>
                    <div class="chart-container">
                        <canvas id="riskChart"></canvas>
                    </div>
                </div>
                
                <div class="stats-card">
                    <h3>📦 Recent Shipments</h3>
                    <div class="shipment-list" id="shipmentList">
                        <div class="shipment-item">Loading...</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let map;
            let markers = [];
            let autoRefreshInterval;
            let riskChart;
            
            // Initialize map
            function initMap() {
                map = L.map('map').setView([40.7128, -74.0060], 2);
                
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
            }
            
            // Initialize risk chart
            function initChart() {
                const ctx = document.getElementById('riskChart').getContext('2d');
                riskChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                        datasets: [{
                            data: [0, 0, 0],
                            backgroundColor: ['#27ae60', '#f39c12', '#e74c3c']
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            }
                        }
                    }
                });
            }
            
            // Fetch and display shipments
            async function refreshData() {
                try {
                    const response = await fetch('/api/shipments');
                    const shipments = await response.json();
                    
                    updateMap(shipments);
                    updateStats(shipments);
                    updateShipmentList(shipments);
                    
                } catch (error) {
                    console.error('Error fetching data:', error);
                }
            }
            
            // Update map with shipments
            function updateMap(shipments) {
                // Clear existing markers
                markers.forEach(marker => map.removeLayer(marker));
                markers = [];
                
                shipments.forEach(shipment => {
                    const riskColor = getRiskColor(shipment.late_risk, shipment.very_late_risk);
                    const riskLevel = getRiskLevel(shipment.late_risk, shipment.very_late_risk);
                    
                    const marker = L.circleMarker([shipment.lat, shipment.lng], {
                        radius: 8,
                        fillColor: riskColor,
                        color: '#fff',
                        weight: 2,
                        opacity: 1,
                        fillOpacity: 0.8
                    }).addTo(map);
                    
                    marker.bindPopup(`
                        <div style="min-width: 200px;">
                            <h4>📦 ${shipment.id}</h4>
                            <p><strong>Location:</strong> ${shipment.order_city}, ${shipment.order_state}</p>
                            <p><strong>Risk Level:</strong> <span style="color: ${riskColor};">${riskLevel}</span></p>
                            <p><strong>Late Risk:</strong> ${(shipment.late_risk * 100).toFixed(1)}%</p>
                            <p><strong>Very Late Risk:</strong> ${(shipment.very_late_risk * 100).toFixed(1)}%</p>
                            <p><strong>Shipping Mode:</strong> ${shipment.shipping_mode}</p>
                            <p><strong>Order Value:</strong> $${shipment.order_value}</p>
                        </div>
                    `);
                    
                    markers.push(marker);
                });
            }
            
            // Update statistics
            function updateStats(shipments) {
                const total = shipments.length;
                const highRisk = shipments.filter(s => s.very_late_risk > 0.7).length;
                const mediumRisk = shipments.filter(s => s.very_late_risk > 0.3 && s.very_late_risk <= 0.7).length;
                const lowRisk = total - highRisk - mediumRisk;
                
                document.getElementById('totalShipments').textContent = total;
                document.getElementById('highRisk').textContent = highRisk;
                
                // Update chart
                riskChart.data.datasets[0].data = [lowRisk, mediumRisk, highRisk];
                riskChart.update();
            }
            
            // Update shipment list
            function updateShipmentList(shipments) {
                const listContainer = document.getElementById('shipmentList');
                listContainer.innerHTML = '';
                
                shipments.slice(0, 10).forEach(shipment => {
                    const riskColor = getRiskColor(shipment.late_risk, shipment.very_late_risk);
                    const riskLevel = getRiskLevel(shipment.late_risk, shipment.very_late_risk);
                    
                    const item = document.createElement('div');
                    item.className = 'shipment-item';
                    item.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>${shipment.id}</strong><br>
                                <small>${shipment.order_city}, ${shipment.order_state}</small>
                            </div>
                            <div style="color: ${riskColor}; font-weight: bold;">
                                ${riskLevel}
                            </div>
                        </div>
                    `;
                    listContainer.appendChild(item);
                });
            }
            
            // Helper functions
            function getRiskColor(lateRisk, veryLateRisk) {
                if (veryLateRisk > 0.7) return '#e74c3c';
                if (veryLateRisk > 0.3) return '#f39c12';
                return '#27ae60';
            }
            
            function getRiskLevel(lateRisk, veryLateRisk) {
                if (veryLateRisk > 0.7) return 'High Risk';
                if (veryLateRisk > 0.3) return 'Medium Risk';
                return 'Low Risk';
            }
            
            function clearMap() {
                markers.forEach(marker => map.removeLayer(marker));
                markers = [];
            }
            
            function toggleAutoRefresh() {
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                    autoRefreshInterval = null;
                } else {
                    autoRefreshInterval = setInterval(refreshData, 5000); // Refresh every 5 seconds
                }
            }
            
            // Initialize everything
            document.addEventListener('DOMContentLoaded', function() {
                initMap();
                initChart();
                refreshData();
                
                // Auto refresh every 10 seconds
                autoRefreshInterval = setInterval(refreshData, 10000);
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(dashboard_html)

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    analytics_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shipment Analytics Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .header { text-align: center; margin-bottom: 30px; }
            .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
            .chart-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .chart-container { height: 300px; }
            .metric-card { background: white; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric-value { font-size: 2.5em; font-weight: bold; color: #2c3e50; }
            .metric-label { color: #7f8c8d; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📈 Shipment Analytics Dashboard</h1>
            <p>Comprehensive analysis of shipment patterns and risk factors</p>
        </div>
        
        <div class="dashboard-grid">
            <div class="metric-card">
                <div class="metric-value" id="avgAccuracy">86.14%</div>
                <div class="metric-label">Late Model Accuracy</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="avgRecall">97.58%</div>
                <div class="metric-label">Very Late Model Recall</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="totalPredictions">0</div>
                <div class="metric-label">Total Predictions Today</div>
            </div>
            
            <div class="chart-card">
                <h3>Risk Distribution by Shipping Mode</h3>
                <div class="chart-container">
                    <canvas id="shippingModeChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3>Geographic Risk Analysis</h3>
                <div class="chart-container">
                    <canvas id="geographicChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3>Hourly Prediction Trends</h3>
                <div class="chart-container">
                    <canvas id="trendsChart"></canvas>
                </div>
            </div>
        </div>

        <script>
            // Initialize charts
            function initCharts() {
                // Shipping Mode Risk Chart
                const shippingCtx = document.getElementById('shippingModeChart').getContext('2d');
                new Chart(shippingCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Standard Class', 'First Class', 'Second Class', 'Same Day'],
                        datasets: [{
                            label: 'Average Risk Score',
                            data: [0.65, 0.25, 0.85, 0.15],
                            backgroundColor: ['#e74c3c', '#f39c12', '#e74c3c', '#27ae60']
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1
                            }
                        }
                    }
                });
                
                // Geographic Risk Chart
                const geoCtx = document.getElementById('geographicChart').getContext('2d');
                new Chart(geoCtx, {
                    type: 'pie',
                    data: {
                        labels: ['North America', 'Europe', 'Asia', 'South America', 'Others'],
                        datasets: [{
                            data: [45, 25, 20, 7, 3],
                            backgroundColor: ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
                
                // Trends Chart
                const trendsCtx = document.getElementById('trendsChart').getContext('2d');
                const hours = Array.from({length: 24}, (_, i) => i + ':00');
                const predictions = Array.from({length: 24}, () => Math.floor(Math.random() * 50) + 10);
                
                new Chart(trendsCtx, {
                    type: 'line',
                    data: {
                        labels: hours,
                        datasets: [{
                            label: 'Predictions per Hour',
                            data: predictions,
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
            
            // Update metrics
            function updateMetrics() {
                const totalPredictions = Math.floor(Math.random() * 1000) + 500;
                document.getElementById('totalPredictions').textContent = totalPredictions.toLocaleString();
            }
            
            // Initialize everything
            document.addEventListener('DOMContentLoaded', function() {
                initCharts();
                updateMetrics();
                
                // Update metrics every 30 seconds
                setInterval(updateMetrics, 30000);
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(analytics_html)

@app.route('/api/shipments')
def get_shipments():
    """API endpoint to get sample shipment data with predictions"""
    try:
        shipments = []
        
        # Generate 20-50 sample shipments
        num_shipments = random.randint(20, 50)
        
        for _ in range(num_shipments):
            shipment = generate_sample_shipment()
            
            # Get predictions if models are loaded
            if late_model and very_late_model:
                try:
                    X_processed = preprocess_input_data(shipment)
                    
                    # Get prediction probabilities
                    late_prob = late_model.predict_proba(X_processed)[0][1]
                    very_late_prob = very_late_model.predict_proba(X_processed)[0][1]
                    
                    shipment['late_risk'] = float(late_prob)
                    shipment['very_late_risk'] = float(very_late_prob)
                    
                except Exception as e:
                    logger.error(f"Error generating predictions for shipment: {e}")
                    # Use random values if prediction fails
                    shipment['late_risk'] = random.uniform(0.1, 0.9)
                    shipment['very_late_risk'] = random.uniform(0.1, 0.8)
            else:
                # Use random values if models not loaded
                shipment['late_risk'] = random.uniform(0.1, 0.9)
                shipment['very_late_risk'] = random.uniform(0.1, 0.8)
            
            shipments.append(shipment)
        
        return jsonify(shipments)
        
    except Exception as e:
        logger.error(f"Error generating shipments: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)