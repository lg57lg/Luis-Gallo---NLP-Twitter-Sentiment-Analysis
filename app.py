import flask
from flask import Flask, request, jsonify, render_template_string
import joblib
import os

app = Flask(__name__)

# Load saved model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(VECTORIZER_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)

load_artifacts()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Web App</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f4f7f6;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            background: #ffffff;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 24px;
        }
        p {
            color: #666;
            font-size: 14px;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 15px;
            resize: vertical;
            box-sizing: border-box;
            margin-bottom: 20px;
            outline: none;
            transition: border-color 0.2s;
        }
        textarea:focus {
            border-color: #3498db;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            width: 100%;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result-box {
            margin-top: 25px;
            padding: 15px 20px;
            border-radius: 8px;
            display: none;
        }
        .result-title {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .result-sentiment {
            font-size: 22px;
            font-weight: bold;
        }
        .negative { background-color: #fde8e8; color: #c0392b; }
        .neutral { background-color: #fef5e7; color: #d35400; }
        .positive { background-color: #e8f8f5; color: #27ae60; }
        .probs {
            margin-top: 10px;
            font-size: 13px;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analyzer</h1>
        <p>Enter text below to predict whether the sentiment is Positive, Neutral, or Negative.</p>
        <textarea id="inputText" placeholder="Type or paste your text here..."></textarea>
        <button onclick="analyzeSentiment()">Analyze Sentiment</button>

        <div id="resultBox" class="result-box">
            <div class="result-title">Predicted Sentiment</div>
            <div id="sentimentResult" class="result-sentiment"></div>
            <div id="probsResult" class="probs"></div>
        </div>
    </div>

    <script>
        async function analyzeSentiment() {
            const text = document.getElementById('inputText').value.trim();
            if (!text) return;

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            const resultBox = document.getElementById('resultBox');
            const sentimentResult = document.getElementById('sentimentResult');
            const probsResult = document.getElementById('probsResult');

            if (data.error) {
                alert(data.error);
                return;
            }

            resultBox.className = 'result-box ' + data.prediction.toLowerCase();
            resultBox.style.display = 'block';
            sentimentResult.innerText = data.prediction;

            if (data.probabilities) {
                let probText = "Probabilities: ";
                for (const [cls, prob] of Object.entries(data.probabilities)) {
                    probText += `${cls}: ${(prob * 100).toFixed(1)}% | `;
                }
                probsResult.innerText = probText.slice(0, -3);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer files not found. Ensure model.pkl and vectorizer.pkl exist.'}), 500

    data = request.get_json(silent=True) or {}
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    # Transform input text using TfidfVectorizer
    features = vectorizer.transform([text])
    
    # Predict sentiment class and probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    class_probs = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}

    return jsonify({
        'text': text,
        'prediction': str(prediction),
        'probabilities': class_probs
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
