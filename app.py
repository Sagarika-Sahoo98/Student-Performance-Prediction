import streamlit as st
import pandas as pd
import joblib
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main{
    background-color:#f5f7fa;
}
h1{
    text-align:center;
    color:#1565C0;
}
.stButton>button{
    width:100%;
    background-color:#1565C0;
    color:white;
    font-size:20px;
    border-radius:10px;
    height:50px;
}
.stButton>button:hover{
    background-color:#0D47A1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("student_model.pkl")
    scaler = joblib.load("student_scaler.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎓 Student Performance Prediction")
st.sidebar.markdown("""
### Machine Learning Project

**Model Used**
- Logistic Regression

**Developed Using**
- Python
- Streamlit
- Scikit-learn
""")

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Prediction System")
st.write("Enter the student's details below and click **Predict Result**.")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 10, 30, 18)

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    student_class = st.number_input("Class", 1, 12, 10)

    study_hours = st.number_input("Study Hours Per Day", 0.0, 24.0, 4.0)

    attendance = st.number_input("Attendance Percentage", 0, 100, 80)

    parent = st.selectbox(
        "Parental Education",
        ["High School", "Graduate", "Post Graduate"]
    )

    edu = {
        "High School": 0,
        "Graduate": 1,
        "Post Graduate": 2
    }

with col2:
    internet = st.selectbox("Internet Access", ["No", "Yes"])
    internet = 1 if internet == "Yes" else 0

    activity = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    activity = 1 if activity == "Yes" else 0

    math = st.number_input("Math Score", 0, 100, 70)

    science = st.number_input("Science Score", 0, 100, 70)

    english = st.number_input("English Score", 0, 100, 70)

    previous = st.number_input("Previous Year Score", 0, 100, 70)

    final_percentage = st.number_input("Final Percentage", 0, 100, 75)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Result"):

    input_data = pd.DataFrame([[
        age,
        gender,
        student_class,
        study_hours,
        attendance,
        edu[parent],
        internet,
        activity,
        math,
        science,
        english,
        previous,
        final_percentage
    ]], columns=[
        "Age",
        "Gender",
        "Class",
        "Study_Hours_Per_Day",
        "Attendance_Percentage",
        "Parental_Education",
        "Internet_Access",
        "Extracurricular_Activities",
        "Math_Score",
        "Science_Score",
        "English_Score",
        "Previous_Year_Score",
        "Final_Percentage"
    ])

    input_data = scaler.transform(input_data)

    with st.spinner("Predicting..."):
        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    prediction = model.predict(input_data)

    st.divider()

    if prediction[0] == 1:

        st.snow()

        st.markdown(
            "<h2 style='text-align:center;color:green;'>🎉 Congratulations! 🎉</h2>",
            unsafe_allow_html=True
        )

        st.success("✅ The Student is Predicted to PASS")

    else:

        st.markdown(
            "<h2 style='text-align:center;color:red;'>❌ Better Luck Next Time!</h2>",
            unsafe_allow_html=True
        )

        st.error("❌ The Student is Predicted to FAIL")
        st.warning("📚 Keep studying consistently and improve your performance!")
