import streamlit as st
import pandas as pd
import joblib
import requests
import time
from streamlit_lottie import st_lottie

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main{
    background-color:#f8f9fa;
}
h1{
    text-align:center;
    color:#0E76A8;
}
.stButton>button{
    width:100%;
    background:#0E76A8;
    color:white;
    font-size:20px;
    border-radius:10px;
    height:50px;
}
.stButton>button:hover{
    background:#065A82;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Fireworks Animation ----------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

fireworks = load_lottie(
    "https://assets2.lottiefiles.com/packages/lf20_touohxv0.json"
)

# ---------------- Load Model ----------------
model = joblib.load("student_model.pkl")
scaler = joblib.load("student_scaler.pkl")

# ---------------- Sidebar ----------------
st.sidebar.title("🎓 Student Performance Prediction")
st.sidebar.success("Machine Learning Project")
st.sidebar.write("""
Model Used:
- Logistic Regression

Technology:
- Python
- Streamlit
- Scikit-learn
""")

# ---------------- Title ----------------
st.title("🎓 Student Performance Prediction System")
st.write("Fill all the details and click **Predict Result**.")

# ---------------- Input ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age",10,30,18)

    gender = st.selectbox("Gender",["Male","Female"])
    gender = 1 if gender=="Male" else 0

    student_class = st.number_input("Class",1,12,10)

    study_hours = st.number_input("Study Hours Per Day",0.0,24.0,4.0)

    attendance = st.number_input("Attendance Percentage",0,100,80)

    parent = st.selectbox(
        "Parental Education",
        ["High School","Graduate","Post Graduate"]
    )

    edu={
        "High School":0,
        "Graduate":1,
        "Post Graduate":2
    }

with col2:

    internet = st.selectbox("Internet Access",["No","Yes"])
    internet = 1 if internet=="Yes" else 0

    activity = st.selectbox("Extracurricular Activities",["No","Yes"])
    activity = 1 if activity=="Yes" else 0

    math = st.number_input("Math Score",0,100,70)

    science = st.number_input("Science Score",0,100,70)

    english = st.number_input("English Score",0,100,70)

    previous = st.number_input("Previous Year Score",0,100,70)

    final = st.number_input("Final Percentage",0,100,75)

# ---------------- Prediction ----------------
if st.button("🚀 Predict Result"):

    df = pd.DataFrame([[
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
        final
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

    df = scaler.transform(df)

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    prediction = model.predict(df)

    st.divider()

    if prediction[0] == 1:

        st_lottie(fireworks, height=280)

        st.markdown(
            "<h2 style='text-align:center;color:green;'>🎉 Congratulations! 🎉</h2>",
            unsafe_allow_html=True
        )

        st.success("✅ The Student is Predicted to PASS")

    else:

        st.error("❌ The Student is Predicted to FAIL")

        st.warning("📚 Keep practicing and work harder for better results.")