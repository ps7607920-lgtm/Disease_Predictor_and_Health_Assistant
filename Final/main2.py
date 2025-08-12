import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, f1_score, recall_score

@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("Training.csv").drop_duplicates()
    x = df.drop("prognosis", axis=1)
    y = df["prognosis"]

    label = LabelEncoder()
    y_array = label.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y_array, test_size=0.3, random_state=42)

    gnb = GaussianNB()
    lr = LogisticRegression(random_state=42)
    voting_clf = VotingClassifier(estimators=[('gnb', gnb), ('lr', lr)], voting='soft')
    voting_clf.fit(x_train, y_train)

    pickle.dump(voting_clf, open('model.pkl', 'wb'))
    model = pickle.load(open('model.pkl', 'rb'))

    return df, x, y, y_array, model, label

@st.cache_data
def load_resources():
    return {
        "symtoms_df": pd.read_csv("symtoms_df.csv"),
        "precautions_df": pd.read_csv("precautions_df.csv"),
        "workout_df": pd.read_csv("workout_df.csv"),
        "description_df": pd.read_csv("description.csv"),
        "medications_df": pd.read_csv("medications.csv"),
        "diets_df": pd.read_csv("diets.csv"),
        "doctor_df": pd.read_csv("Doctor.csv")
    }

def build_dicts(df, y_array, y):
    disease_dict = {idx: name for idx, name in zip(np.unique(y_array), pd.Series(y).unique())}
    symptoms_dict = {symptom: i for i, symptom in enumerate(df.columns) if symptom != "prognosis"}
    return disease_dict, symptoms_dict

def get_prediction(symptoms, symptoms_dict, model, disease_dict):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    prediction_index = model.predict([input_vector])[0]
    return disease_dict[prediction_index]

def helper(disease, resources):
    description_df = resources["description_df"]
    precautions_df = resources["precautions_df"]
    symtoms_df = resources["symtoms_df"]
    medications_df = resources["medications_df"]
    diets_df = resources["diets_df"]
    workout_df = resources["workout_df"]
    doctor_df = resources["doctor_df"]

    desc = " ".join(description_df[description_df["Disease"] == disease]["Description"].tolist())

    precaution = precautions_df[precautions_df["Disease"] == disease][
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ].values.flatten().tolist()
    precaution = [p for p in precaution if pd.notna(p)]

    disease_symptoms_row = symtoms_df[symtoms_df["Disease"] == disease].head(1)
    symptoms = []
    if not disease_symptoms_row.empty:
        symptoms = disease_symptoms_row[
            ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]
        ].values.flatten().tolist()
        symptoms = [s for s in symptoms if pd.notna(s)]

    medicine = medications_df[medications_df["Disease"] == disease]["Medication"].tolist()
    medicine = [m for m in medicine if pd.notna(m)][0][2:-2].split("', '")

    diet = diets_df[diets_df["Disease"] == disease]["Diet"].tolist()
    diet = [d for d in diet if pd.notna(d)][0][2:-2].split("', '")

    workout = workout_df[workout_df["disease"] == disease]["workout"].tolist()
    workout = [w for w in workout if pd.notna(w)]

    doctor = doctor_df[doctor_df["Disease"] == disease]

    return desc, precaution, medicine, diet, workout, symptoms, doctor

# Streamlit UI
st.set_page_config(page_title="Disease Predictor", page_icon="🩺")
st.title("🩺 Disease Predictor & Health Assistant")
st.markdown("Enter your symptoms below to get disease prediction and health recommendations.")

df, x, y, y_array, model, label = load_data_and_model()
resources = load_resources()
disease_dict, symptoms_dict = build_dicts(df, y_array, y)

# Input
symptom_input = st.text_input(
    "Enter symptoms separated by commas (e.g., skin_rash, itching):", key="symptom_input"
)

if st.button("Predict Disease"):
    if symptom_input:
        patient_symptoms = [s.strip().lower() for s in symptom_input.split(',')]
        predicted_disease = get_prediction(patient_symptoms, symptoms_dict, model, disease_dict)

        st.markdown(f"## 🧾 Predicted Disease: **:blue[{predicted_disease}]**")

        desc, precaution, medicine, diet, workout, symptoms, doctor = helper(predicted_disease, resources)

        st.markdown("### 📖 Description")
        st.markdown(f"> {desc if desc else 'No description available.'}")

        st.markdown("### 🛡️ Precautions")
        if precaution:
            for p in precaution:
                st.markdown(f"- ✅ {p}")
        else:
            st.markdown("*No precautions found.*")

        st.markdown("### 🔎 Related Symptoms")
        if symptoms:
            for s in symptoms:
                s_clean = s.replace("_", " ").title()
                st.markdown(f"- 🤒 {s_clean}")
        else:
            st.markdown("*No symptoms listed.*")

        st.markdown("### 💊 Medications")
        if medicine:
            for m in medicine:
                st.markdown(f"- 💊 {m}")
        else:
            st.markdown("*No medications listed.*")

        st.markdown("### 🥗 Diet Recommendations")
        if diet:
            for d in diet:
                st.markdown(f"- 🥦 {d}")
        else:
            st.markdown("*No diet listed.*")

        st.markdown("### 🏋️ Workout Suggestions")
        if workout:
            for w in workout:
                st.markdown(f"- 🏃 {w}")
        else:
            st.markdown("*No workouts listed.*")

        st.markdown("### 👨‍⚕️ Doctor Recommendations")
        if not doctor.empty:
            for i in doctor.index:
                st.markdown(f"🧑‍⚕️ **Specialization:** `{doctor['Specialization'][i]}`")
                st.markdown(f"🆔 **Doctor ID:** `{doctor['DoctorID'][i]}`")
                st.markdown(f"🚻 **Gender:** `{doctor['Gender'][i]}`")
        else:
            st.markdown("*No doctor available for this disease.*")
    else:
        st.warning("⚠️ Please enter symptoms to get a prediction.")
