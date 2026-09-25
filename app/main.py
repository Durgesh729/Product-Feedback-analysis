# Product Feedback Analysis Using NLP — FastAPI Application & Web Interface (v2.0.1)

import os
import sys
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction import SentimentPredictor

app = FastAPI(
    title="Multi-Domain Product Feedback Sentiment Analysis API",
    description="Multi-domain customer sentiment analysis API using NLP and Logistic Regression across 31 product categories",
    version="2.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load sentiment predictor
try:
    predictor = SentimentPredictor()
except Exception as e:
    print(f"Warning: Could not initialize SentimentPredictor: {e}")
    predictor = None


class ReviewRequest(BaseModel):
    text: str


# Single review prediction endpoint
@app.post("/api/predict/single")
async def predict_single_review(request: ReviewRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Sentiment model is not trained or loaded properly.")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        
    result = predictor.predict_single(request.text)
    return result


# Batch CSV review prediction endpoint
@app.post("/api/predict/batch")
async def predict_batch_reviews(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Sentiment model is not trained or loaded properly.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
            
        processed_df, text_col = predictor.predict_batch(df)
        
        # Calculate summary metrics
        counts = processed_df['predicted_sentiment'].value_counts().to_dict()
        summary = {
            'total_reviews': len(processed_df),
            'positive_count': counts.get('Positive', 0),
            'negative_count': counts.get('Negative', 0),
            'neutral_count': counts.get('Neutral', 0),
            'detected_text_column': text_col
        }
        
        # Save temporary output CSV for download
        output_filename = f"analyzed_{file.filename}"
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        processed_df.to_csv(output_path, index=False)
        
        # Return sample data for table display + summary
        sample_rows = processed_df[[text_col, 'predicted_sentiment', 'confidence']].head(50).to_dict(orient='records')
        
        return {
            'summary': summary,
            'results': sample_rows,
            'download_filename': output_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")


# Endpoint to download analyzed CSV file
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested file not found.")
    return FileResponse(file_path, media_type='text/csv', filename=filename)


# Web Interface (HTML/CSS/JS Response)
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Product Feedback Sentiment Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563EB;
            --primary-hover: #1D4ED8;
            --bg: #F8FAFC;
            --card-bg: #FFFFFF;
            --text-main: #0F172A;
            --text-muted: #64748B;
            --border: #E2E8F0;
            --pos-bg: #DCFCE7;
            --pos-text: #15803D;
            --pos-border: #86EFAC;
            --neg-bg: #FEE2E2;
            --neg-text: #B91C1C;
            --neg-border: #FCA5A5;
            --neu-bg: #FEF3C7;
            --neu-text: #B45309;
            --neu-border: #FDE68A;
            --warn-bg: #FFFBEB;
            --warn-border: #FCD34D;
            --warn-text: #92400E;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            padding: 2rem 1rem;
            min-height: 100vh;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.05rem;
        }

        .nav-tabs {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.5rem;
        }

        .tab-btn {
            background: none;
            border: none;
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-muted);
            padding: 0.6rem 1.2rem;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.2s;
        }

        .tab-btn.active {
            color: var(--primary);
            background-color: #EFF6FF;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.8rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
        }

        .card h2 {
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: var(--text-main);
        }

        .sample-btns {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }

        .sample-btn {
            background: #F1F5F9;
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 0.4rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-main);
            cursor: pointer;
            transition: background 0.2s;
        }

        .sample-btn:hover {
            background: #E2E8F0;
        }

        textarea {
            width: 100%;
            height: 110px;
            padding: 0.9rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 1rem;
            outline: none;
            resize: vertical;
            margin-bottom: 1rem;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .btn-primary {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-primary:hover {
            background-color: var(--primary-hover);
        }

        /* Results Box */
        .result-panel {
            display: none;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }

        .domain-box {
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin-bottom: 1.2rem;
            font-size: 0.95rem;
        }

        .domain-valid {
            background-color: #F0FDF4;
            border: 1px solid var(--pos-border);
            color: #166534;
        }

        .domain-invalid {
            background-color: var(--warn-bg);
            border: 1px solid var(--warn-border);
            color: var(--warn-text);
        }

        .badge {
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }

        .badge-positive {
            background-color: var(--pos-bg);
            color: var(--pos-text);
            border: 1px solid var(--pos-border);
        }

        .badge-negative {
            background-color: var(--neg-bg);
            color: var(--neg-text);
            border: 1px solid var(--neg-border);
        }

        .badge-neutral {
            background-color: var(--neu-bg);
            color: var(--neu-text);
            border: 1px solid var(--neu-border);
        }

        .prob-group {
            margin-bottom: 0.8rem;
        }

        .prob-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }

        .progress-bar-bg {
            background-color: #F1F5F9;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.4s ease;
        }

        .fill-pos { background-color: #22C55E; }
        .fill-neg { background-color: #EF4444; }
        .fill-neu { background-color: #F59E0B; }

        .note-box {
            background-color: #F8FAFC;
            border-left: 4px solid var(--primary);
            padding: 0.9rem 1.1rem;
            font-size: 0.88rem;
            color: var(--text-muted);
            border-radius: 4px;
            margin-top: 1.2rem;
        }

        details {
            margin-top: 1.2rem;
            background: #F8FAFC;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }

        summary {
            font-weight: 600;
            cursor: pointer;
            color: var(--primary);
            outline: none;
        }

        .nlp-flow {
            margin-top: 0.8rem;
            font-size: 0.88rem;
            color: var(--text-main);
        }

        .nlp-step {
            padding: 0.4rem 0;
            border-bottom: 1px dashed var(--border);
        }

        .nlp-step:last-child {
            border-bottom: none;
        }

        /* Batch Upload */
        .upload-box {
            border: 2px dashed var(--border);
            border-radius: 10px;
            padding: 2.5rem;
            text-align: center;
            background: #FAFAFA;
            cursor: pointer;
            margin-bottom: 1rem;
        }

        .upload-box:hover {
            border-color: var(--primary);
            background: #F0F7FF;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .metric-card {
            background: #F8FAFC;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }

        .metric-card .num {
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.95rem;
        }

        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background-color: #F1F5F9;
            font-weight: 600;
            color: var(--text-main);
        }

        tr:hover {
            background-color: #F8FAFC;
        }

        .spinner {
            display: none;
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛍️ Product Feedback Sentiment Analysis</h1>
            <p>Supervised Machine Learning System for Customer Product Review Evaluation</p>
        </header>

        <div class="nav-tabs">
            <button class="tab-btn active" onclick="switchTab('single')">🔍 Analyze Single Review</button>
            <button class="tab-btn" onclick="switchTab('batch')">📁 Batch CSV Review Analysis</button>
        </div>

        <!-- TAB 1: SINGLE REVIEW ANALYSIS -->
        <div id="tab-single" class="tab-content active">
            <div class="card">
                <h2>Enter Product Feedback</h2>
                <div class="sample-btns">
                    <span style="font-size: 0.85rem; color: var(--text-muted); align-self: center;">Try sample:</span>
                    <button class="sample-btn" onclick="setSample('The product quality is excellent and I am extremely satisfied!')">🟢 Excellent Quality</button>
                    <button class="sample-btn" onclick="setSample('The product stopped working after two days and support is terrible.')">🔴 Defective Product</button>
                    <button class="sample-btn" onclick="setSample('The order arrived yesterday in standard packaging.')">🟡 Standard Delivery</button>
                    <button class="sample-btn" onclick="setSample('I am not feeling well')">⚠️ Non-Product Input</button>
                </div>
                
                <textarea id="single-text" placeholder="Type or paste customer product review here..."></textarea>
                
                <button class="btn-primary" onclick="analyzeSingle()">
                    <span>🚀 Analyze Sentiment</span>
                    <div id="single-spinner" class="spinner"></div>
                </button>

                <!-- Single Result Panel -->
                <div id="single-result" class="result-panel">
                    <!-- Valid Prediction Panel -->
                    <div id="valid-prediction-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                            <div>
                                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.2rem;">PREDICTED SENTIMENT</div>
                                <div id="sentiment-badge" class="badge">Positive</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.85rem; color: var(--text-muted);">MODEL PROBABILITY</div>
                                <div id="model-prob-score" style="font-size: 1.5rem; font-weight: 700; color: var(--primary);">0%</div>
                            </div>
                        </div>

                        <div style="margin-top: 1.2rem;">
                            <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.6rem; font-weight: 600;">CLASS PROBABILITIES</div>
                            <div class="prob-group">
                                <div class="prob-label"><span>🟢 Positive</span><span id="prob-pos-val">0%</span></div>
                                <div class="progress-bar-bg"><div id="prob-pos-bar" class="progress-bar-fill fill-pos" style="width: 0%;"></div></div>
                            </div>
                            <div class="prob-group">
                                <div class="prob-label"><span>🔴 Negative</span><span id="prob-neg-val">0%</span></div>
                                <div class="progress-bar-bg"><div id="prob-neg-bar" class="progress-bar-fill fill-neg" style="width: 0%;"></div></div>
                            </div>
                            <div class="prob-group">
                                <div class="prob-label"><span>🟡 Neutral</span><span id="prob-neu-val">0%</span></div>
                                <div class="progress-bar-bg"><div id="prob-neu-bar" class="progress-bar-fill fill-neu" style="width: 0%;"></div></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: BATCH CSV ANALYSIS -->
        <div id="tab-batch" class="tab-content">
            <div class="card">
                <h2>Batch CSV Product Feedback Processing</h2>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.2rem;">
                    Upload a CSV dataset containing customer reviews. Start Review from 2nd row.
                </p>

                <div class="upload-box" onclick="document.getElementById('csv-file').click()">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                    <div style="font-weight: 600; color: var(--text-main);">Click to select or drag & drop CSV file</div>
                    <div id="file-name" style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.4rem;">Supports .csv files containing product feedback</div>
                    <input type="file" id="csv-file" accept=".csv" style="display: none;" onchange="updateFileName()">
                </div>

                <button class="btn-primary" onclick="uploadBatch()">
                    <span>⚡ Process & Analyze CSV Batch</span>
                    <div id="batch-spinner" class="spinner"></div>
                </button>

                <!-- Batch Results Container -->
                <div id="batch-result" class="result-panel">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <h3>Batch Results Summary</h3>
                        <a id="download-btn" href="#" class="btn-primary" style="text-decoration: none; font-size: 0.9rem; padding: 0.5rem 1rem;">
                            📥 Download Analyzed CSV
                        </a>
                    </div>

                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div style="font-size: 0.85rem; color: var(--text-muted);">Total Reviews</div>
                            <div id="metric-total" class="num" style="color: var(--primary);">0</div>
                        </div>
                        <div class="metric-card">
                            <div style="font-size: 0.85rem; color: var(--text-muted);">Positive Reviews</div>
                            <div id="metric-pos" class="num" style="color: #16A34A;">0</div>
                        </div>
                        <div class="metric-card">
                            <div style="font-size: 0.85rem; color: var(--text-muted);">Negative Reviews</div>
                            <div id="metric-neg" class="num" style="color: #DC2626;">0</div>
                        </div>
                        <div class="metric-card">
                            <div style="font-size: 0.85rem; color: var(--text-muted);">Neutral Reviews</div>
                            <div id="metric-neu" class="num" style="color: #D97706;">0</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            if (tabName === 'single') {
                document.querySelectorAll('.tab-btn')[0].classList.add('active');
                document.getElementById('tab-single').classList.add('active');
            } else {
                document.querySelectorAll('.tab-btn')[1].classList.add('active');
                document.getElementById('tab-batch').classList.add('active');
            }
        }

        function setSample(text) {
            document.getElementById('single-text').value = text;
        }

        async function analyzeSingle() {
            const text = document.getElementById('single-text').value.trim();
            if (!text) {
                alert('Please enter a review text to analyze.');
                return;
            }

            const spinner = document.getElementById('single-spinner');
            spinner.style.display = 'inline-block';

            try {
                const response = await fetch('/api/predict/single', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();
                if (!response.ok) {
                    alert(data.detail || 'Error predicting sentiment.');
                    return;
                }

                const validBox = document.getElementById('valid-prediction-box');

                if (data.is_valid_domain) {
                    validBox.style.display = 'block';

                    // Update Sentiment & Probability
                    const badge = document.getElementById('sentiment-badge');
                    badge.innerText = data.sentiment;
                    badge.className = 'badge badge-' + (data.sentiment ? data.sentiment.toLowerCase() : 'neutral');

                    const probScore = data.predicted_class_probability || data.confidence || 0;
                    document.getElementById('model-prob-score').innerText = probScore + '%';

                    const probs = data.probabilities || {};
                    document.getElementById('prob-pos-val').innerText = (probs.Positive || 0) + '%';
                    document.getElementById('prob-pos-bar').style.width = (probs.Positive || 0) + '%';

                    document.getElementById('prob-neg-val').innerText = (probs.Negative || 0) + '%';
                    document.getElementById('prob-neg-bar').style.width = (probs.Negative || 0) + '%';

                    document.getElementById('prob-neu-val').innerText = (probs.Neutral || 0) + '%';
                    document.getElementById('prob-neu-bar').style.width = (probs.Neutral || 0) + '%';

                } else {
                    alert(data.domain_message || 'Input is outside product feedback domain.');
                    validBox.style.display = 'none';
                }

                document.getElementById('single-result').style.display = 'block';
            } catch (err) {
                alert('Failed to connect to API server: ' + err.message);
            } finally {
                spinner.style.display = 'none';
            }
        }

        function updateFileName() {
            const input = document.getElementById('csv-file');
            if (input.files.length > 0) {
                document.getElementById('file-name').innerText = 'Selected: ' + input.files[0].name;
            }
        }

        async function uploadBatch() {
            const input = document.getElementById('csv-file');
            if (input.files.length === 0) {
                alert('Please select a CSV file to upload.');
                return;
            }

            const spinner = document.getElementById('batch-spinner');
            spinner.style.display = 'inline-block';

            const formData = new FormData();
            formData.append('file', input.files[0]);

            try {
                const response = await fetch('/api/predict/batch', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) {
                    alert(data.detail || 'Error processing CSV file.');
                    return;
                }

                // Update Metrics
                const summary = data.summary;
                document.getElementById('metric-total').innerText = summary.total_reviews;
                document.getElementById('metric-pos').innerText = summary.positive_count;
                document.getElementById('metric-neg').innerText = summary.negative_count;
                document.getElementById('metric-neu').innerText = summary.neutral_count;

                // Update Download Link
                const downloadBtn = document.getElementById('download-btn');
                downloadBtn.href = '/api/download/' + data.download_filename;

                document.getElementById('batch-result').style.display = 'block';
            } catch (err) {
                alert('Failed to process batch CSV: ' + err.message);
            } finally {
                spinner.style.display = 'none';
            }
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
