from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("disease_model (1).pkl")

# Read and clean dataset
df = pd.read_csv("datasets/Training.csv")
if "Unnamed: 133" in df.columns:
    df = df.drop(["Unnamed: 133"], axis=1)

all_symptoms = df.columns[:-1].tolist()  # 132 features

@app.route("/")
def index():
    return render_template("index.html", symptoms=all_symptoms)

@app.route("/predict", methods=["POST"])
def predict():
    selected = request.form.getlist("symptoms")
    input_vector = [1 if symptom in selected else 0 for symptom in all_symptoms]
    prediction = model.predict([input_vector])[0]
    return render_template("index.html", symptoms=all_symptoms, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
