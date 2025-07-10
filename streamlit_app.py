import streamlit as st
st.set_page_config(page_title="Prediksi Rekrutmen Kandidat")  # HARUS PALING ATAS

import numpy as np
import joblib
import os

# ======== Konfigurasi Path Model ========
MODEL_PATH = "recruitment_model.joblib"  # ⬅️ Ubah ke file di root folder

# ======== Load Model ========
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model belum dilatih. Jalankan model_training.py terlebih dahulu.")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ======== Judul Aplikasi ========
st.title("🎯 Prediksi Hasil Rekrutmen Kandidat")
st.write("Masukkan informasi kandidat untuk memprediksi apakah mereka akan diterima.")

# ======== Form Input User ========
with st.form("form_prediksi"):
    city = st.selectbox("Lokasi Kandidat", ["city_1", "city_2", "city_3"], index=0)
    city_development_index = st.slider("City Development Index (0.0 - 1.0)", 0.0, 1.0, 0.5)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
    relevent_experience = st.selectbox("Pengalaman Relevan", ["Has relevent experience", "No relevent experience"])
    enrolled_university = st.selectbox("Status Kuliah", ["no_enrollment", "Full time course", "Part time course"])
    education_level = st.selectbox("Tingkat Pendidikan", ["Graduate", "Masters", "PhD", "Unknown"])
    major_discipline = st.selectbox("Jurusan", ["STEM", "Business Degree", "Arts", "Humanities", "Other", "Unknown"])
    experience = st.slider("Pengalaman (tahun)", 0, 20, 2)
    company_size = st.selectbox("Ukuran Perusahaan Terakhir", ["Unknown", "<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
    company_type = st.selectbox("Jenis Perusahaan", ["Unknown", "Private", "Public", "NGO", "Startup", "Other"])
    last_new_job = st.slider("Tahun sejak terakhir pindah kerja", 0, 5, 1)
    training_hours = st.number_input("Jumlah Training Hours", 0, 300, 50)

    submitted = st.form_submit_button("🔍 Prediksi")

# ======== Mapping Manual Encode ========
manual_encode = {
    "city": {"city_1": 0, "city_2": 1, "city_3": 2},
    "gender": {"Male": 1, "Female": 0, "Other": 2},
    "relevent_experience": {"Has relevent experience": 1, "No relevent experience": 0},
    "enrolled_university": {"no_enrollment": 0, "Full time course": 1, "Part time course": 2},
    "education_level": {"Graduate": 1, "Masters": 2, "PhD": 3, "Unknown": 0},
    "major_discipline": {"STEM": 5, "Business Degree": 0, "Arts": 1, "Humanities": 2, "Other": 3, "Unknown": 4},
    "company_size": {"<10": 0, "10/49": 1, "50-99": 2, "100-500": 3, "500-999": 4,
                     "1000-4999": 5, "5000-9999": 6, "10000+": 7, "Unknown": 8},
    "company_type": {"Private": 4, "Public": 5, "NGO": 2, "Startup": 6, "Other": 1, "Unknown": 0},
}

# ======== Prediksi ========
if submitted:
    if model is None:
        st.stop()

    input_data = np.array([[  
        manual_encode["city"][city],
        city_development_index,
        manual_encode["gender"][gender],
        manual_encode["relevent_experience"][relevent_experience],
        manual_encode["enrolled_university"][enrolled_university],
        manual_encode["education_level"][education_level],
        manual_encode["major_discipline"][major_discipline],
        experience,
        manual_encode["company_size"][company_size],
        manual_encode["company_type"][company_type],
        last_new_job,
        training_hours
    ]])

    prediction = model.predict(input_data)[0]

    st.success("✅ Kandidat kemungkinan **akan diterima.**" if prediction == 1 else "❌ Kandidat kemungkinan **tidak diterima.**")
