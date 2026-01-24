import streamlit as st
import pandas as pd
import numpy as np
import pickle

# LOAD MODEL
reg_model = pickle.load(open("house_price_regression.pkl", "rb"))
cls_model = pickle.load(open("house_price_classification.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# PAGE CONFIG
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .price-display {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .category-badge {
        display: inline-block;
        background: rgba(255,255,255,0.3);
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .team-member {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    h2 {
        color: #667eea;
        font-weight: 600;
    }
    
    .stNumberInput>div>div>input {
        border-radius: 8px;
    }
    
    .stSelectbox>div>div>select {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
    <div class="main-header">
        <h1>🏠 Prediksi Harga Rumah Jakarta</h1>
        <p>Aplikasi Machine Learning untuk Prediksi Harga Rumah Di Jakarta</p>
    </div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("""
        <div class="sidebar-info">
            <h2 style="color: white; margin-top: 0;">📋 Informasi Project</h2>
            <p style="margin-bottom: 0;">Aplikasi ini dibuat untuk memenuhi UAS Mata Kuliah Data Science.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div class="sidebar-info">
            <h3 style="color: white; margin-top: 0;">👥 Kelompok 2</h3>
            <div class="team-member">
                <strong>1. Husain Abdul Ghani</strong><br>
                <small>D112311004</small>
            </div>
            <div class="team-member">
                <strong>2. Yesa Aradya Pasha</strong><br>
                <small>D112311010</small>
            </div>
        </div>
    """, unsafe_allow_html=True)

# INPUT SECTION
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown("### 🔢 Masukkan Spesifikasi Rumah")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🛏️ Spesifikasi Ruangan")
    bed_rooms = st.number_input("Jumlah Kamar Tidur", min_value=0, value=2, key="bedrooms")
    bath_rooms = st.number_input("Jumlah Kamar Mandi", min_value=0, value=1, key="bathrooms")
    carport = st.number_input("Jumlah Carport", min_value=0, value=1, key="carport")

with col2:
    st.markdown("#### 📐 Spesifikasi Luas")
    land_area = st.number_input("Luas Tanah (m²)", min_value=0, value=100, key="land")
    building_area = st.number_input("Luas Bangunan (m²)", min_value=0, value=80, key="building")
    st.markdown("---")
    st.markdown("#### 📍 Lokasi")
    district = st.selectbox("District", ["Jakarta Selatan", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara", "Jakarta Pusat"], key="district")
    city = st.selectbox("City", ["Jakarta"], key="city")

st.markdown('</div>', unsafe_allow_html=True)

# PREPROCESS INPUT
input_data = {
    "bed_rooms": bed_rooms,
    "bath_rooms": bath_rooms,
    "carport": carport,
    "land_area": land_area,
    "building_area": building_area
}

# one-hot manual
for col in model_columns:
    if col.startswith("district_"):
        input_data[col] = 1 if col == f"district_{district}" else 0
    elif col.startswith("city_"):
        input_data[col] = 1 if col == f"city_{city}" else 0

input_df = pd.DataFrame([input_data])

# samakan kolom
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# PREDIKSI BUTTON
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("🔮 Prediksi Harga Rumah", use_container_width=True)

if predict_button:
    price_pred = reg_model.predict(input_df)[0]
    category_pred = cls_model.predict(input_df)[0]
    
    # Format price
    formatted_price = f"Rp {price_pred:,.0f}"
    
    # Display prediction with modern card
    st.markdown(f"""
        <div class="prediction-card">
            <h2 style="color: white; margin-top: 0;">✅ Prediksi Berhasil!</h2>
            <div class="price-display">{formatted_price}</div>
            <div class="category-badge">🏷️ Kategori: {category_pred}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Additional info cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Luas Tanah", f"{land_area} m²")
    
    with col2:
        st.metric("Luas Bangunan", f"{building_area} m²")
    
    with col3:
        st.metric("Total Kamar", f"{bed_rooms + bath_rooms}")

# FOOTER
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Made with Husain Abdul Ghani & Yesa Aradya Pasha | Machine Learning Prediction System</p>
    </div>
""", unsafe_allow_html=True)