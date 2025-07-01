from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from classification import classify_models, classify_ensemble  # Import ensemble function
import upload_u

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ensemble')
def ensemble_page():
    return render_template('ensemble.html')  # New page for ensemble methods

@app.route('/upload', methods=['POST'])
def upload_dataset():
    global df
    df = upload_u.upload_dataset2(app)
    return jsonify({
        "message": "File uploaded successfully!",
        "head": df.head().to_dict(orient='records')
    })

@app.route("/classify", methods=["POST"])
def classify():
    if df is None:
        return jsonify({"error": "No dataset uploaded"}), 400
    results = classify_models(df)
    return jsonify(results)

@app.route("/ensemble_classify", methods=["POST"])
def ensemble_classify():
    if df is None:
        return jsonify({"error": "No dataset uploaded"}), 400
    
    data = request.json
    selected_models = data.get("models", [])
    voting_type = data.get("voting", "hard")


    results = classify_ensemble(df,selected_models,voting_type)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
