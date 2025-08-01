# Disease_Predictor_and_Health_Assistant

Disease Predictor and Health Assistant is a machine learning-powered web application built using Python and Streamlit. It helps users identify possible diseases based on entered symptoms and provides them with useful health suggestions, including:

📖 Disease Description

🛡️ Precautions

💊 Medications

🥗 Diet Recommendations

🏋️ Workout Suggestions

👨‍⚕ Doctor Recommendations

This project was developed as a part of a summer internship using Data Science and Machine Learning techniques to support proactive health awareness.

🔍 Features

* Accepts multiple symptoms from users via a text input.
* Predicts the most probable disease using a Voting Classifier (combination of Logistic Regression and Naive Bayes).
* Provides additional resources such as:

   1. Disease description
   2. Recommended precautions
   3. Common symptoms
   4. Suggested medications
   5. Healthy diet tips
   6. Beneficial workouts
   7. Nearby or specialized doctors

🛠️ Tech Stack

1. Frontend/UI: Streamlit
2.Backend: Python
3. Machine Learning: Scikit-learn (Logistic Regression, Naive Bayes, Voting Classifier)
4. Data Handling: Pandas, NumPy
5. Model Storage: Pickle
6. Data Files: CSV-based health knowledgebase

📂 Project Structure

├── main2.py                   # Streamlit app

├── model.pkl                  # Trained ML model (generated on first run)

├── Training.csv               # Training dataset for symptoms and prognosis

├── symtoms_df.csv             # Symptom database

├── precautions_df.csv         # Precaution suggestions per disease

├── description.csv            # Disease descriptions

├── medications.csv            # Medication info

├── diets.csv                  # Diet plans

├── workout_df.csv             # Workout suggestions

├── Doctor.csv                 # Doctor recommendation database

└── requirements.txt           # Python dependencies

▶️ How to Run the Project Locally

✅ Prerequisites
Make sure you have Python 3.8+ and pip installed.

🔧 Installation Steps

  1. Clone the repository

        git clone https://github.com/your-username/disease-predictor.git
        cd disease-predictor

  2. Install dependencies

        pip install -r requirements.txt
   
  3. Ensure all CSV files are in the same directory as main2.py.

  4. Run the app

        streamlit run main2.py

  5. Open the browser and navigate to http://localhost:8501 to use the app.

🙏 Acknowledgment

This project was developed as part of the Summer Internship conducted by GRAStech at Babu Banarasi Das University.
Special thanks to our mentor Arpit Sir for his continuous guidance and support throughout the project.

📌 Example Input

skin_rash, itching, nodal_skin_eruptions

🔗 License

This project is open-source.
