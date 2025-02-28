import pandas as pd
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sentence_transformers import SentenceTransformer

# Load CSV file (Job Role, Skills, Experience, Keywords)
df = pd.read_csv("D:/VS Code/mlproject/job_requirements_generated.csv")

# Label Encode Job Roles
label_encoder = LabelEncoder()
df["Job Role ID"] = label_encoder.fit_transform(df["Job Role"])  # Converts job roles to numbers

# Feature Encoding (One-Hot for Skills)
one_hot = OneHotEncoder()
skills_encoded = one_hot.fit_transform(df[["Required Skills"]]).toarray()

# Prepare X (features) and y (target)
X = skills_encoded  # Use skills as input features
y = df["Job Role ID"]  # Target: Job Role ID

# ✅ Add 'qid' column for ranking (Each job role gets its own group)
df["qid"] = df["Job Role ID"]  # Ensures XGBoost understands ranking groups

# Convert to DMatrix (with qid for ranking)
train_dmatrix = xgb.DMatrix(X, label=y)
train_dmatrix.set_group(df.groupby("Job Role ID")["Job Role ID"].count().tolist())

# Train XGBoost Ranking Model
xgb_model = xgb.XGBRanker(
    objective="rank:pairwise",
    booster="gbtree",
    eta=0.1,
    max_depth=6,
    eval_metric="ndcg"
)

xgb_model.fit(X, y, group=df.groupby("Job Role ID")["Job Role ID"].count().tolist())

# ✅ Save XGBoost Model
with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("✅ XGBoost Ranking Model trained and saved!")

# --------------------------
# ✅ Train & Save BERT Model
# --------------------------
print("🔄 Training BERT Model...")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & fast

# Convert "Required Skills" column to embeddings
bert_embeddings = bert_model.encode(df["Required Skills"].tolist(), convert_to_tensor=True)

# Save BERT embeddings & model
with open("models/bert_model.pkl", "wb") as f:
    pickle.dump(bert_model, f)

with open("models/bert_embeddings.pkl", "wb") as f:
    pickle.dump(bert_embeddings, f)

print("✅ BERT Model & embeddings saved successfully!")
