# backend/app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from model_processor import predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    API endpoint for task priority prediction.
    
    Expected JSON payload:
    {
        "task_text": "Your task description here"
    }
    
    Returns:
        JSON response with prediction or error message
    """
    # CORS preflight
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json(force=True)
        logger.info(f"Request JSON: {data}")

        task_text = data.get('task_text', '').strip()
        if not task_text:
            return jsonify({'error': 'No task_text provided'}), 400

        prediction = predictor.predict_priority(task_text)
        return jsonify({'prediction': prediction})

    except Exception as e:
        logger.error("Prediction error", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # In production, use a WSGI server instead
    app.run(host='0.0.0.0', port=8080)