from __future__ import annotations

import html
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def load_assets():
    reg_path = APP_DIR / "house_price_regression.pkl"
    cls_path = APP_DIR / "house_price_classification.pkl"
    cols_path = APP_DIR / "model_columns.pkl"

    missing = [p.name for p in (reg_path, cls_path, cols_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "File model tidak ditemukan: " + ", ".join(missing) + ". "
            "Pastikan file .pkl ada di folder yang sama dengan app.py."
        )

    with reg_path.open("rb") as f:
        reg_model = pickle.load(f)
    with cls_path.open("rb") as f:
        cls_model = pickle.load(f)
    with cols_path.open("rb") as f:
        model_columns = pickle.load(f)

    required = {"bed_rooms", "bath_rooms", "carport", "land_area", "building_area"}
    if not required.issubset(set(model_columns)):
        missing_cols = sorted(required - set(model_columns))
        raise ValueError(f"model_columns.pkl tidak lengkap. Missing: {missing_cols}")

    return reg_model, cls_model, model_columns


def format_idr(value: float) -> str:
    try:
        n = float(value)
    except Exception:
        return "Rp -"
    return "Rp " + f"{n:,.0f}".replace(",", ".")


# PAGE CONFIG
st.set_page_config(
    page_title="Prediksi Harga Rumah Jakarta",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS — diselaraskan dengan dark theme .streamlit/config.toml
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Hero */
    .ux-hero {
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        padding: 2.25rem 2rem 2rem 2rem;
        margin-bottom: 1.75rem;
        background: linear-gradient(135deg,
            rgba(102, 126, 234, 0.22) 0%,
            rgba(118, 75, 162, 0.18) 45%,
            rgba(6, 182, 212, 0.12) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 20px 50px -20px rgba(0, 0, 0, 0.45);
    }
    .ux-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(800px 200px at 10% 0%, rgba(139, 92, 246, 0.15), transparent 50%),
                    radial-gradient(600px 180px at 90% 100%, rgba(6, 182, 212, 0.12), transparent 50%);
        pointer-events: none;
    }
    .ux-hero h1 {
        position: relative;
        margin: 0 0 0.5rem 0;
        font-size: clamp(1.6rem, 3vw, 2.15rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
        background: linear-gradient(90deg, #e2e8f0, #a5b4fc, #5eead4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .ux-hero p.ux-sub {
        position: relative;
        margin: 0;
        font-size: 1rem;
        color: #94a3b8;
        max-width: 36rem;
    }
    .ux-hero .ux-badges {
        position: relative;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.1rem;
    }
    .ux-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
    }

    /* Stat strip */
    .ux-stat-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    @media (max-width: 768px) {
        .ux-stat-strip { grid-template-columns: 1fr; }
    }
    .ux-stat {
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .ux-stat .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; }
    .ux-stat .value { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin-top: 0.2rem; }

    .ux-section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
    }
    .ux-h2 {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Panels (key Streamlit: suffix di data-testid, mis. ...-input_panel) */
    [data-testid$="input_panel"] {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.55) 0%, rgba(15, 23, 42, 0.35) 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }
    [data-testid$="summary_panel"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 27, 75, 0.25) 100%) !important;
        border: 1px solid rgba(129, 140, 248, 0.2) !important;
        border-radius: 16px !important;
    }
    [data-testid$="input_panel"] [data-testid="stVerticalBlock"],
    [data-testid$="summary_panel"] [data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }

    .ux-hint { font-size: 0.8rem; color: #64748b; line-height: 1.4; }

    .ux-kv { margin: 0.35rem 0; padding: 0.55rem 0; border-bottom: 1px solid rgba(148, 163, 184, 0.1); }
    .ux-kv:last-of-type { border-bottom: none; }
    .ux-kv .k { font-size: 0.75rem; color: #94a3b8; }
    .ux-kv .v { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-top: 0.15rem; }
    .ux-pill-ok {
        display: inline-block;
        margin-top: 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    /* Primary button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #0ea5e9 100%) !important;
        color: white !important;
        font-size: 1.05rem;
        font-weight: 700;
        padding: 0.9rem 1.5rem;
        border-radius: 14px;
        border: none !important;
        box-shadow: 0 8px 32px -8px rgba(99, 102, 241, 0.55);
        transition: transform 0.15s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 36px -6px rgba(99, 102, 241, 0.65);
    }
    .stButton>button:focus { outline: 2px solid rgba(129, 140, 248, 0.5); outline-offset: 2px; }

    /* Result card */
    .ux-result {
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        margin: 1.5rem 0 0.5rem 0;
        padding: 2rem 1.5rem 2rem 1.5rem;
        text-align: center;
        background: linear-gradient(140deg, rgba(139, 92, 246, 0.28) 0%, rgba(236, 72, 153, 0.2) 50%, rgba(6, 182, 212, 0.18) 100%);
        border: 1px solid rgba(167, 139, 250, 0.25);
        box-shadow: 0 20px 50px -20px rgba(0, 0, 0, 0.5);
        animation: uxFade 0.45s ease-out;
    }
    .ux-result::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 50%;
        height: 200%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04));
        transform: rotate(20deg);
        pointer-events: none;
    }
    @keyframes uxFade {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .ux-result .ux-lbl { font-size: 0.8rem; color: #c4b5fd; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
    .ux-result .ux-price { font-size: clamp(1.5rem, 4vw, 2.4rem); font-weight: 800; margin: 0.5rem 0; color: #f8fafc; letter-spacing: -0.02em; text-shadow: 0 2px 20px rgba(0,0,0,0.2); }
    .ux-tag {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #e0e7ff;
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* Post-prediction metrics */
    [data-testid="stMetricValue"] { font-size: 1.15rem; }

    .ux-footer { text-align: center; color: #64748b; font-size: 0.85rem; padding: 1.5rem 0.5rem 0.5rem; }
    .ux-footer a { color: #a5b4fc; text-decoration: none; }
    .ux-footer a:hover { text-decoration: underline; }

    /* Sidebar cards */
    .ux-side-card {
        background: linear-gradient(180deg, rgba(51, 65, 85, 0.45) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
    }
    .ux-side-card h3 { margin: 0 0 0.4rem; font-size: 0.9rem; color: #e2e8f0; }
    .ux-side-card p { margin: 0; font-size: 0.8rem; line-height: 1.5; color: #94a3b8; }
    .ux-avatar {
        display: flex; align-items: center; gap: 0.75rem; margin-top: 0.75rem; padding: 0.6rem; border-radius: 10px;
        background: rgba(0,0,0,0.2);
    }
    .ux-avatar .dot { width: 2.4rem; height: 2.4rem; border-radius: 10px; background: linear-gradient(135deg, #6366f1, #8b5cf6);
        display: flex; align-items: center; justify-content: center; font-size: 1.1rem; }
    .ux-avatar .name { font-weight: 700; color: #f1f5f9; font-size: 0.9rem; }
    .ux-avatar .sub { font-size: 0.75rem; color: #94a3b8; }

    h2, h3 { color: #e2e8f0 !important; }
    .stNumberInput>div>div>input, .stSelectbox>div>div>select { border-radius: 10px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# LOAD MODELS
try:
    reg_model, cls_model, model_columns = load_assets()
except Exception as e:
    st.error("Gagal memuat model. Periksa file `.pkl` di folder project.")
    st.exception(e)
    st.stop()

# HERO
st.markdown(
    """
    <div class="ux-hero">
        <h1>Prediksi harga rumah DKI Jakarta</h1>
        <p class="ux-sub">Estimasi harga dan kategori dengan model regresi + klasifikasi. Input rapi, hasil jelas—siap buat presentasi UAS.</p>
        <div class="ux-badges">
            <span class="ux-badge">✦ Streamlit</span>
            <span class="ux-badge">✦ scikit-learn</span>
            <span class="ux-badge">✦ One-hot distrik + kota</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick stat strip
st.markdown(
    """
    <div class="ux-stat-strip">
        <div class="ux-stat"><div class="label">Model</div><div class="value">Regresi + Klasifikasi</div></div>
        <div class="ux-stat"><div class="label">Cakupan</div><div class="value">5 distrik DKI</div></div>
        <div class="ux-stat"><div class="label">Format harga</div><div class="value">Rupiah (IDR)</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# LAYOUT: input + ringkasan
left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    with st.container(border=True, key="input_panel"):
        st.markdown(
            """<p class="ux-section-title">Langkah 1</p><h2 class="ux-h2">📝 Isi spesifikasi</h2>""",
            unsafe_allow_html=True,
        )
        st.caption("Sesuaikan angka dan lokasi. Semua field dipakai model untuk membangun vektor fitur.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ruangan**")
            bed_rooms = st.number_input("Kamar tidur", min_value=0, max_value=20, value=2, key="bedrooms")
            bath_rooms = st.number_input("Kamar mandi", min_value=0, max_value=20, value=1, key="bathrooms")
            carport = st.number_input("Carport", min_value=0, max_value=20, value=1, key="carport")
        with c2:
            st.markdown("**Luas & lokasi**")
            land_area = st.number_input("Luas tanah (m²)", min_value=0, max_value=20000, value=100, key="land")
            building_area = st.number_input("Luas bangunan (m²)", min_value=0, max_value=20000, value=80, key="building")
            district = st.selectbox(
                "Distrik",
                ["Jakarta Selatan", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara", "Jakarta Pusat"],
                key="district",
            )
            city = st.selectbox("Kota", ["Jakarta"], key="city")
        st.markdown(
            """<p class="ux-hint">💡 Tip: bandingkan skenario dengan mengubah luas tanah / distrik, lalu tekan prediksi lagi.</p>""",
            unsafe_allow_html=True,
        )

# PREPROCESS
input_data = {
    "bed_rooms": bed_rooms,
    "bath_rooms": bath_rooms,
    "carport": carport,
    "land_area": land_area,
    "building_area": building_area,
}
for col in model_columns:
    if col.startswith("district_"):
        input_data[col] = 1 if col == f"district_{district}" else 0
    elif col.startswith("city_"):
        input_data[col] = 1 if col == f"city_{city}" else 0
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=model_columns, fill_value=0)

with right_col:
    with st.container(border=True, key="summary_panel"):
        st.markdown(
            """<p class="ux-section-title">Pratinjau</p><h2 class="ux-h2">👁 Ringkasan input</h2>""",
            unsafe_allow_html=True,
        )
        b_ratio = (building_area / land_area * 100) if land_area else 0.0
        st.markdown(
            f"""
            <div class="ux-kv"><div class="k">Lokasi</div><div class="v">{html.escape(city)} · {html.escape(district)}</div></div>
            <div class="ux-kv"><div class="k">Luas</div><div class="v">{land_area} m² tanah · {building_area} m² bangunan</div></div>
            <div class="ux-kv"><div class="k">Kamar & parkir</div><div class="v">{bed_rooms} tidur, {bath_rooms} mandi, {carport} carport</div></div>
            <div class="ux-kv"><div class="k">Rasio bangunan / tanah</div><div class="v">{b_ratio:.1f}%</div></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<span class="ux-pill-ok">Input siap diproses</span>', unsafe_allow_html=True)

st.divider()

# SIDEBAR
with st.sidebar:
    st.markdown("### Aplikasi")
    st.caption("Prediksi harga rumah berbasis machine learning")
    st.markdown(
        """
        <div class="ux-side-card">
            <h3>📋 Tentang</h3>
            <p>Proyek solo untuk UAS Data Science: regresi memprediksi harga, klasifikasi memberi kategori harga.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="ux-side-card">
            <h3>👤 Author</h3>
            <div class="ux-avatar">
                <div class="dot">H</div>
                <div><div class="name">Husain Abdul Ghani</div><div class="sub">Solo project</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### Pengaturan")
    show_debug = st.toggle("Tampilkan vektor fitur (debug)", value=False)
    with st.expander("Cara memakai", expanded=False):
        st.write(
            "1. Isi jumlah kamar, luas, dan pilih distrik.\n"
            "2. Cek ringkasan di panel kanan.\n"
            "3. Klik **Prediksi harga** untuk estimasi harga + kategori."
        )

if show_debug:
    st.markdown("#### Debug — dataframe input (1 baris)")
    st.dataframe(input_df, use_container_width=True, height=180)

# PREDICT
st.markdown("<div style='height:0.25rem;'></div>", unsafe_allow_html=True)
predict_button = st.button("✨ Hitung prediksi harga", use_container_width=True, type="primary")

if predict_button:
    if building_area > land_area and land_area > 0:
        st.warning("Luas bangunan melebihi luas tanah. Cek kembali angka; prediksi tetap dihitung.")
    with st.status("Memproses prediksi model…", expanded=False):
        price_pred = float(reg_model.predict(input_df)[0])
        category_raw = cls_model.predict(input_df)[0]
    category_pred = str(category_raw)
    safe_cat = html.escape(category_pred)
    formatted_price = format_idr(price_pred)

    st.markdown(
        f"""
        <div class="ux-result">
            <div class="ux-lbl">Estimasi harga</div>
            <div class="ux-price">{html.escape(formatted_price)}</div>
            <div class="ux-tag">Kategori: {safe_cat}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Luas tanah", f"{land_area} m²")
    m2.metric("Luas bangunan", f"{building_area} m²")
    m3.metric("Kamar (tidur+mandi)", f"{bed_rooms + bath_rooms}")
    m4.metric("Carport", f"{carport}")

st.markdown(
    """
    <div class="ux-footer">Made with care by <strong style="color:#94a3b8;">Husain Abdul Ghani</strong> — Streamlit · pandas · scikit-learn</div>
    """,
    unsafe_allow_html=True,
)
