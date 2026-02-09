from flask import Flask, render_template, request, redirect, session
import pickle
import pandas as pd
import numpy as np
import os


# ===============================
# APP SETUP
# ===============================

app = Flask(__name__)
app.secret_key = "healthcare_ai"


# ===============================
# LOAD MODEL FILE
# ===============================

MODEL_FILE = "recommender_objects.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("❌ recommender_objects.pkl not found!")

with open(MODEL_FILE, "rb") as f:
    obj = pickle.load(f)


# ===============================
# EXTRACT DATA SAFELY
# ===============================

df = obj.get("df")

disease_model = obj.get("disease_model")

content_similarity = obj.get("content_similarity")

user_item = obj.get("user_item_matrix")

le_gender = obj.get("le_gender")

le_condition = obj.get("le_condition")


if df is None or disease_model is None:
    raise Exception("❌ Model data missing in pickle file")


# ===============================
# EXTRACT SYMPTOMS (FOR REGISTER)
# ===============================

all_symptoms = []

if "symptoms" in df.columns:

    temp = ",".join(df["symptoms"].astype(str))

    temp = temp.replace("nan", "")

    all_symptoms = sorted(set(temp.split(",")))


# ===============================
# FAKE DATABASE (DEMO)
# ===============================

users_db = {}


# ===============================
# DISEASE PREDICTION
# ===============================

def predict_disease(age, gender, visit, bmi, sugar):

    # Encode gender
    try:
        gender_enc = le_gender.transform([gender])[0]
    except:
        gender_enc = 0

    features = np.array([
        [age, gender_enc, visit, bmi, sugar]
    ])

    # Predict encoded value
    pred_enc = disease_model.predict(features)[0]

    # Decode disease name
    try:
        disease_name = le_condition.inverse_transform([pred_enc])[0]
    except:
        disease_name = str(pred_enc)

    return disease_name


# ===============================
# HYBRID RECOMMENDATION
# ===============================

def hybrid_recommend(top_n=5):

    scores = pd.Series(dtype=float)

    # Content based
    if content_similarity is not None:

        content_score = pd.Series(
            content_similarity.mean(axis=0)
        )

        scores = content_score


    # Collaborative
    if user_item is not None:

        collab_score = user_item.mean()

        scores = scores.add(collab_score, fill_value=0)


    # If empty
    if scores.empty:

        return pd.Series(["No data"] * top_n)


    scores = scores.fillna(0)

    return scores.sort_values(
        ascending=False
    ).head(top_n)


# ===============================
# LOGIN
# ===============================

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]

        if email in users_db:

            session["user"] = users_db[email]

            return redirect("/dashboard")

        return "❌ Invalid user"

    return render_template("login.html")


# ===============================
# REGISTER
# ===============================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        form = request.form.to_dict()

        # Handle multiple symptoms
        symptoms = request.form.getlist("symptoms")

        form["symptoms"] = ",".join(symptoms)

        email = form["email"]

        users_db[email] = form

        return redirect("/")


    return render_template(
        "register.html",
        symptoms=all_symptoms
    )


# ===============================
# DASHBOARD
# ===============================

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    if "user" not in session:
        return redirect("/")

    user = session["user"]

    disease = None

    recommendations = hybrid_recommend().to_dict()


    if request.method == "POST":

        try:

            disease = predict_disease(

                int(user["age"]),
                user["gender"],
                int(user["doctor_visit_freq"]),
                float(user["bmi"]),
                float(user["sugar_level"])

            )

        except Exception as e:

            print("Prediction Error:", e)

            disease = "Prediction Failed"


    return render_template(
        "dashboard.html",
        user=user,
        disease=disease,
        recommendations=recommendations
    )


# ===============================
# PROFILE
# ===============================

@app.route("/profile")
def profile():

    if "user" not in session:
        return redirect("/")

    return render_template(
        "profile.html",
        user=session["user"]
    )


# ===============================
# LOGOUT
# ===============================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# ===============================
# RUN SERVER
# ===============================

if __name__ == "__main__":

    app.run(debug=True)







