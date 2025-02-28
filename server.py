import os
import pickle
import numpy as np
import pandas as pd
import PyPDF2
from flask import Flask, render_template, request, send_from_directory
from sentence_transformers import SentenceTransformer
import xgboost as xgb

app = Flask(__name__)

# -------------------------------
# ✅ Configurations
# -------------------------------
UPLOAD_FOLDER = "uploaded_resumes"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

# -------------------------------
# ✅ Load Trained Models
# -------------------------------
print("🔄 Loading Models...")

# Load XGBoost Model
with open("models/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Load BERT Model
with open("models/bert_model.pkl", "rb") as f:
    bert_model = pickle.load(f)

# Load BERT Embeddings
with open("models/bert_embeddings.pkl", "rb") as f:
    bert_embeddings = pickle.load(f)

print("✅ Models Loaded Successfully!")

# Load Job Requirements Data
job_data = pd.read_csv("D:/VS Code/mlproject/job_requirements_generated.csv")

# -------------------------------
# ✅ Extract Text from PDF Resumes
# -------------------------------
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text.strip()

# -------------------------------
# ✅ Compare Resumes using BERT
# -------------------------------
def compare_resume_with_job(resume_text, job_role):
    # Find Job Role Skills from Dataset
    job_skills = job_data[job_data["Job Role"] == job_role]["Required Skills"].values
    if len(job_skills) == 0:
        return 0  # If no matching job role, return lowest score

    # Convert Resume Text to BERT Embeddings
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)

    # Compute Similarity Scores
    job_embedding = bert_model.encode(job_skills[0], convert_to_tensor=True)
    similarity_score = np.dot(resume_embedding, job_embedding) / (np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding))

    return similarity_score.item()  # Convert tensor to float

# -------------------------------
# ✅ Flask Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    ranked_resumes = []

    if request.method == "POST":
        job_role = request.form["job_role"]
        num_openings = int(request.form["num_openings"])  # Get number of openings
        uploaded_files = request.files.getlist("resumes")

        resume_scores = []

        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            score = compare_resume_with_job(resume_text, job_role)

            # Save resume to disk for downloading
            resume_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(resume_path)

            resume_scores.append((file.filename, score))

        # Rank Resumes based on Score
        ranked_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)

        # Select top N resumes based on number of openings
        ranked_resumes = ranked_resumes[:num_openings]

    return render_template("index.html", ranked_resumes=ranked_resumes)

# -------------------------------
# ✅ Resume Download Route
# -------------------------------
@app.route("/download/<filename>")
def download_resume(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

# -------------------------------
# ✅ Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)