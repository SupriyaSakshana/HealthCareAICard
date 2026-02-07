# train_models.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score


# =====================================================
# 1️⃣ LOAD DATA
# =====================================================

DATA_PATH = r"G:\Supriya\healthcare\dataset\healthcare_advanced_dataset_1000_ai_ready.csv"

df = pd.read_csv(DATA_PATH)

print("✅ Dataset Loaded Successfully")


# =====================================================
# 2️⃣ BASIC CLEANING
# =====================================================

# Remove missing rows (if any)
df.dropna(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

print("✅ Data Cleaned")


# =====================================================
# 3️⃣ ENCODING
# =====================================================

# Gender Encoder
le_gender = LabelEncoder()
df["gender_enc"] = le_gender.fit_transform(df["gender"])

# Condition Encoder (Target)
le_condition = LabelEncoder()
df["condition_enc"] = le_condition.fit_transform(df["condition"])

print("✅ Encoding Completed")


# =====================================================
# 4️⃣ DISEASE PREDICTION MODEL
# =====================================================

X = df[[
    "age",
    "gender_enc",
    "doctor_visit_freq",
    "bmi",
    "sugar_level"
]]

y = df["condition_enc"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Random Forest (Best for tabular medical data)
disease_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

disease_model.fit(X_train, y_train)


# Accuracy Check
y_pred = disease_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Disease Model Accuracy: {acc*100:.2f}%")


# =====================================================
# 5️⃣ CONTENT BASED FILTERING (TF-IDF)
# =====================================================

# Combine Text Columns
df["content"] = (
    df["symptoms"].astype(str) + " " +
    df["lifestyle"].astype(str) + " " +
    df["recommendation_item"].astype(str)
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(df["content"])

content_similarity = cosine_similarity(tfidf_matrix)

print("✅ Content-Based Filtering Ready")


# =====================================================
# 6️⃣ COLLABORATIVE FILTERING
# =====================================================

user_item_matrix = df.pivot_table(
    index="user_id",
    columns="recommendation_item",
    values="rating",
    aggfunc="mean"
).fillna(0)

user_similarity = cosine_similarity(user_item_matrix)

print("✅ Collaborative Filtering Ready")


# =====================================================
# 7️⃣ SAVE ALL MODELS
# =====================================================

MODEL_PATH = "recommender_objects.pkl"

with open(MODEL_PATH, "wb") as f:

    pickle.dump({

        # Dataset
        "df": df,

        # Disease Model
        "disease_model": disease_model,
        "le_gender": le_gender,
        "le_condition": le_condition,

        # Content-Based
        "vectorizer": vectorizer,
        "content_similarity": content_similarity,

        # Collaborative
        "user_item_matrix": user_item_matrix,
        "user_similarity": user_similarity

    }, f)


print("🎯 ALL MODELS SAVED SUCCESSFULLY")
print(f"📁 File Saved As: {MODEL_PATH}")




