import base64
import os
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# ==========================================
# KONFIGURASI HALAMAN & TEMA MODERN
# ==========================================
st.set_page_config(
    page_title="Smart Home AI - Asisten Pantau Listrik Rumah",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def render_html(raw_html: str):
    """Membersihkan spasi awal agar HTML dirender sempurna di halaman utama."""
    clean_html = "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)


def scroll_ke_atas():
    """Menggulir layar otomatis ke bagian paling atas saat berpindah tampilan."""
    components.html(
        """
        <script>
            window.parent.scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """,
        height=0,
        width=0,
    )


def get_usm_logo_data_uri() -> str:
    """
    Membaca file logo kampus asli berformat JPG/JPEG/PNG dari folder proyek
    (logo_usm.jpg atau logo.jpg) agar tampil persis seperti gambar asli.
    """
    kandidat_file = [
        "logo_usm.jpg",
        "logo_usm.jpeg",
        "logo_usm.png",
        "logo.jpg",
        "logo.jpeg",
        "logo.png",
    ]
    for nama_file in kandidat_file:
        if os.path.exists(nama_file):
            ext = os.path.splitext(nama_file)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            with open(nama_file, "rb") as img_file:
                b64_img = base64.b64encode(img_file.read()).decode("utf-8")
                return f"data:{mime};base64,{b64_img}"

    # Cadangan vektor otomatis jika file logo_usm.jpg belum dimasukkan ke folder
    usm_svg = """
    <svg width="110" height="130" viewBox="0 0 220 260" xmlns="http://www.w3.org/2000/svg">
        <path d="M110,8 C132,32 168,36 196,48 C188,92 198,136 176,174 C156,192 132,196 110,214 C88,196 64,192 44,174 C22,136 32,92 24,48 C52,36 88,32 110,8 Z"
              fill="#facc15" stroke="#a16207" stroke-width="6" stroke-linejoin="round"/>
        <path d="M110,32 C104,42 104,49 110,52 C116,49 116,42 110,32 Z" fill="#dc2626"/>
        <polygon points="106,58 114,58 117,122 103,122" fill="#111827"/>
        <rect x="70" y="136" width="80" height="10" fill="#111827"/>
        <text x="110" y="250" text-anchor="middle" fill="#facc15" stroke="#111827" stroke-width="2" paint-order="stroke"
              font-family="Arial Black, Arial, sans-serif" font-size="38" font-weight="900">USM</text>
    </svg>
    """
    b64_svg = base64.b64encode(usm_svg.strip().encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64_svg}"


# ==========================================
# GLOBAL CSS: MOBILE-FIRST, MODE LIGHT & DARK
# ==========================================
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: dark;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif !important;
    }

    .stApp {
        background: radial-gradient(circle at 12% 10%, rgba(14, 165, 233, 0.14) 0%, transparent 38%),
                    radial-gradient(circle at 88% 18%, rgba(99, 102, 241, 0.14) 0%, transparent 40%),
                    linear-gradient(180deg, #070b14 0%, #0b1120 55%, #070b14 100%);
        color: #f8fafc;
    }

    /* Sembunyikan sidebar sepenuhnya karena input langsung di halaman utama */
    section[data-testid="stSidebar"],
    button[data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Bilah atas Streamlit agar tidak putih menyilaukan saat Mode Light */
    header[data-testid="stHeader"] {
        background-color: rgba(7, 11, 20, 0.85) !important;
        backdrop-filter: blur(8px);
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg {
        color: #f8fafc !important;
        fill: #f8fafc !important;
    }

    /* Ruang atas diperbesar agar judul tidak tertutup bilah menu Streamlit Cloud */
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
        max-width: 1220px;
    }

    /* Kontras Tinggi untuk Semua Teks & Label di Mode Light maupun Dark */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #f8fafc !important;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    .stApp label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #e2e8f0 !important;
    }
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b {
        color: #ffffff !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #94a3b8 !important;
    }

    .stApp code {
        background-color: rgba(30, 41, 59, 0.92) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        padding: 2px 6px !important;
        border-radius: 6px !important;
    }

    [data-testid="stTooltipHoverTarget"] svg {
        color: #94a3b8 !important;
        stroke: #94a3b8 !important;
    }

    /* Kotak Input, Selectbox, Number Input & Dropdown saat Mode Light */
    [data-baseweb="select"] > div,
    [data-baseweb="input"],
    [data-baseweb="base-input"] {
        background-color: #1e293b !important;
        border-color: rgba(148, 163, 184, 0.38) !important;
        color: #f8fafc !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div,
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }
    [data-baseweb="input"] input::placeholder,
    [data-baseweb="base-input"] input::placeholder {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
        opacity: 1 !important;
    }
    [data-baseweb="select"] svg {
        fill: #cbd5e1 !important;
    }
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #0f172a !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
    }
    ul[role="listbox"] li,
    [role="option"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }
    ul[role="listbox"] li:hover,
    [role="option"]:hover,
    [aria-selected="true"][role="option"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }

    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        background-color: #334155 !important;
        color: #f8fafc !important;
        border-color: rgba(148, 163, 184, 0.3) !important;
    }
    button[data-testid="stNumberInputStepDown"]:hover,
    button[data-testid="stNumberInputStepUp"]:hover {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }

    /* Tombol Sekunder, Tombol Unduh & Expander saat Mode Light */
    button[kind="secondary"],
    [data-testid="stDownloadButton"] button {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
    }
    button[kind="secondary"]:hover,
    [data-testid="stDownloadButton"] button:hover {
        background-color: #0f172a !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    [data-testid="stExpander"] {
        background-color: rgba(15, 23, 42, 0.78) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 14px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: #f8fafc !important;
    }
    [data-testid="stExpander"] summary svg {
        fill: #38bdf8 !important;
    }

    .katex, .katex * {
        color: #f8fafc !important;
    }

    /* Header Utama */
    .main-app-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding-top: 4px;
        margin-bottom: 14px;
    }
    .main-app-title {
        font-size: 25px;
        font-weight: 800;
        background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 55%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.4px;
        line-height: 1.35;
    }
    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.35);
        backdrop-filter: blur(10px);
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 700;
        color: #7dd3fc;
    }
    @keyframes dotPulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.45); opacity: 0.45; }
    }
    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #38bdf8;
        box-shadow: 0 0 10px #38bdf8;
        animation: dotPulse 1.6s ease-in-out infinite;
    }

    /* Banner Petunjuk Singkat di Atas Form Input */
    .input-intro-banner {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.14), rgba(37, 99, 235, 0.14));
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-left: 4px solid #38bdf8;
        border-radius: 14px;
        padding: 13px 16px;
        color: #e2e8f0;
        font-size: 13.5px;
        line-height: 1.5;
        margin-bottom: 14px;
    }

    /* Header Tiap Bagian Formulir Input */
    .form-section-header {
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.75));
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-left: 5px solid #38bdf8;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 14px;
        margin-bottom: 12px;
    }
    .form-section-title {
        font-size: 16px;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 2px;
    }
    .form-section-sub {
        font-size: 12.5px;
        color: #cbd5e1;
    }

    /* Kartu Sub-Alat pada Form Input */
    .device-box-label {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 6px;
    }
    .device-box-name {
        font-size: 14.5px;
        font-weight: 800;
        color: #f8fafc;
    }
    .device-box-watt {
        font-size: 12px;
        font-weight: 700;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 2px 9px;
        border-radius: 999px;
    }

    /* Kartu KPI Glassmorphism */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-top: 14px;
        margin-bottom: 16px;
    }
    .kpi-card {
        position: relative;
        overflow: hidden;
        background: rgba(15, 23, 42, 0.68);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 16px 18px;
        color: #f8fafc;
        transition: all 0.28s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        opacity: 0.75;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.55);
        box-shadow: 0 12px 26px rgba(2, 132, 199, 0.18);
    }
    .kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .kpi-label {
        font-size: 11.5px;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-icon {
        font-size: 16px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        padding: 3px 7px;
    }
    .kpi-value {
        font-size: 23px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Kartu Fitur Token & Anggaran Glassmorphism */
    .glass-card {
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 18px 20px;
        color: #f8fafc;
        margin-bottom: 14px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        transition: border-color 0.25s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.45);
    }
    .feature-title {
        font-size: 15.5px;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    @keyframes geserProgress {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    .progress-track {
        width: 100%;
        height: 14px;
        background-color: rgba(30, 41, 59, 0.9);
        border-radius: 999px;
        overflow: hidden;
        margin: 10px 0;
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 2px;
    }
    .progress-fill-anim {
        height: 100%;
        border-radius: 999px;
        background-size: 200% 200%;
        animation: geserProgress 2.2s linear infinite;
        transition: width 0.55s cubic-bezier(0.22, 1, 0.36, 1);
    }

    /* Styling Tabs Modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.65);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 700;
        font-size: 13.5px;
        color: #cbd5e1 !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7, #2563eb) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }

    /* Kartu Diagnosis Pemborosan */
    .diag-card {
        background: linear-gradient(145deg, rgba(38, 14, 21, 0.88), rgba(24, 12, 18, 0.92));
        border: 1px solid rgba(244, 63, 94, 0.45);
        border-left: 5px solid #f43f5e;
        border-radius: 14px;
        padding: 16px 20px;
        color: #f8fafc;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(244, 63, 94, 0.12);
    }
    .diag-header {
        font-size: 15.5px;
        font-weight: 800;
        color: #fda4af;
        margin-bottom: 8px;
    }
    .diag-row {
        font-size: 13.5px;
        color: #e2e8f0;
        margin-bottom: 6px;
        line-height: 1.55;
    }

    /* Kotak Penjelasan Mudah Dipahami */
    .explain-box {
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(12px);
        border-left: 4px solid #38bdf8;
        border-top: 1px solid rgba(148, 163, 184, 0.16);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 16px 20px;
        color: #e2e8f0;
        margin-top: 10px;
        margin-bottom: 16px;
    }
    .explain-title {
        font-size: 15px;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 8px;
    }
    .explain-item {
        font-size: 13.5px;
        margin-bottom: 6px;
        line-height: 1.6;
        color: #cbd5e1;
    }

    /* Footer Hak Cipta, Logo USM & Kontak WhatsApp */
    .footer-card {
        margin-top: 28px;
        padding: 22px 26px;
        background: linear-gradient(135deg, rgba(11, 19, 43, 0.92) 0%, rgba(15, 23, 42, 0.95) 60%, rgba(23, 37, 84, 0.92) 100%);
        border: 1px solid rgba(59, 130, 246, 0.45);
        border-radius: 18px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        color: #e2e8f0;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.3);
    }
    .footer-brand {
        display: flex;
        align-items: center;
        gap: 18px;
        flex: 1 1 320px;
    }
    .wa-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: #ffffff !important;
        text-decoration: none;
        font-weight: 700;
        font-size: 13px;
        padding: 10px 18px;
        border-radius: 999px;
        border: 1px solid #4ade80;
        margin-top: 6px;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35);
        transition: all 0.22s ease;
    }
    .wa-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.5);
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 4.8rem !important;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
        }
        .main-app-title {
            font-size: 19.5px;
            line-height: 1.35;
        }
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .kpi-value {
            font-size: 18px;
        }
    }
</style>
""")

# ==========================================
# KONSTANTA & OPSI PERALATAN (SEMUA DEFAULT KOSONG / 0)
# ==========================================
PLACEHOLDER_PLN = "-- Pilih Golongan Daya PLN --"

GOLONGAN_PLN = {
    "450 VA (Subsidi - Rp 415/kWh)": {"va": 450, "tarif": 415.0},
    "900 VA (Subsidi - Rp 605/kWh)": {"va": 900, "tarif": 605.0},
    "900 VA (Non-Subsidi - Rp 1.352/kWh)": {"va": 900, "tarif": 1352.0},
    "1300 VA (Rumah Tangga - Rp 1.444,70/kWh)": {"va": 1300, "tarif": 1444.70},
    "2200 VA (Rumah Tangga - Rp 1.444,70/kWh)": {"va": 2200, "tarif": 1444.70},
    "3500 - 5500 VA (R-2 - Rp 1.699,53/kWh)": {"va": 3500, "tarif": 1699.53},
}

OPSI_STANDBY_AWAM = {
    "Tidak Ada (0 Watt - Semua dicabut)": 0,
    "Sedikit: Hanya Charger HP & Lampu Tidur (~10 Watt)": 10,
    "Standar Rumah: Wi-Fi, Charger HP & STB TV (~25 Watt)": 25,
    "Banyak: Wi-Fi, STB TV, CCTV & Banyak Charger (~45 Watt)": 45,
}

OPSI_KULKAS = {
    "Kulkas 1 Pintu (Rata-rata 110 Watt)": 110,
    "Kulkas 2 Pintu (Rata-rata 165 Watt)": 165,
}

OPSI_AC = {
    "AC 1/2 PK (400 Watt)": 400,
    "AC 1 PK (850 Watt)": 850,
}

WATT_MAGICOM_COOK = 380
WATT_MAGICOM_WARM = 50
JAM_PER_MASAK = 0.75
WATT_MESIN_CUCI = 450

PERANGKAT_LAIN = {
    "Lampu LED (Per Titik)": {"watt": 12, "normal_jam": 10.0, "std_jam": 2.0, "max_unit": 30},
    "Kipas Angin": {"watt": 55, "normal_jam": 7.0, "std_jam": 2.0, "max_unit": 10},
    "TV LED": {"watt": 100, "normal_jam": 6.0, "std_jam": 1.5, "max_unit": 5},
    "Sanyo (Pompa Air Otomatis)": {"watt": 200, "normal_jam": 1.5, "std_jam": 0.5, "max_unit": 3},
    "Water Heater": {"watt": 1200, "normal_jam": 1.0, "std_jam": 0.4, "max_unit": 3},
}


# ==========================================
# INISIALISASI STATE KOSONG SAAT AWAL BUKA
# ==========================================
def inisialisasi_state_kosong():
    if "sudah_hitung" not in st.session_state:
        st.session_state["sudah_hitung"] = False
    if "perlu_scroll_atas" not in st.session_state:
        st.session_state["perlu_scroll_atas"] = False

    defaults = {
        "pilihan_golongan": PLACEHOLDER_PLN,
        "target_anggaran": 0,
        "pilihan_standby": list(OPSI_STANDBY_AWAM.keys())[0],
        "pilihan_kulkas": list(OPSI_KULKAS.keys())[0],
        "unit_kulkas": 0,
        "jam_kulkas": 0.0,
        "pilihan_ac": list(OPSI_AC.keys())[0],
        "unit_ac": 0,
        "jam_ac": 0.0,
        "unit_magicom": 0,
        "masak_magicom": 0,
        "warm_magicom": 0.0,
        "unit_cuci": 0,
        "jam_cuci_minggu": 0.0,
        "nama_c1": "",
        "watt_c1": 0,
        "unit_c1": 0,
        "jam_c1": 0.0,
        "nama_c2": "",
        "watt_c2": 0,
        "unit_c2": 0,
        "jam_c2": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    for idx_init in range(len(PERANGKAT_LAIN)):
        if f"unit_{idx_init}" not in st.session_state:
            st.session_state[f"unit_{idx_init}"] = 0
        if f"jam_{idx_init}" not in st.session_state:
            st.session_state[f"jam_{idx_init}"] = 0.0


def reset_semua_isian():
    st.session_state["sudah_hitung"] = False
    st.session_state["perlu_scroll_atas"] = True
    st.session_state["pilihan_golongan"] = PLACEHOLDER_PLN
    st.session_state["target_anggaran"] = 0
    st.session_state["pilihan_standby"] = list(OPSI_STANDBY_AWAM.keys())[0]
    st.session_state["pilihan_kulkas"] = list(OPSI_KULKAS.keys())[0]
    st.session_state["unit_kulkas"] = 0
    st.session_state["jam_kulkas"] = 0.0
    st.session_state["pilihan_ac"] = list(OPSI_AC.keys())[0]
    st.session_state["unit_ac"] = 0
    st.session_state["jam_ac"] = 0.0
    st.session_state["unit_magicom"] = 0
    st.session_state["masak_magicom"] = 0
    st.session_state["warm_magicom"] = 0.0
    st.session_state["unit_cuci"] = 0
    st.session_state["jam_cuci_minggu"] = 0.0
    st.session_state["nama_c1"] = ""
    st.session_state["watt_c1"] = 0
    st.session_state["unit_c1"] = 0
    st.session_state["jam_c1"] = 0.0
    st.session_state["nama_c2"] = ""
    st.session_state["watt_c2"] = 0
    st.session_state["unit_c2"] = 0
    st.session_state["jam_c2"] = 0.0
    for idx_init in range(len(PERANGKAT_LAIN)):
        st.session_state[f"unit_{idx_init}"] = 0
        st.session_state[f"jam_{idx_init}"] = 0.0


inisialisasi_state_kosong()

if st.session_state["perlu_scroll_atas"]:
    scroll_ke_atas()
    st.session_state["perlu_scroll_atas"] = False


# ==========================================
# 1. PEMBANGKIT DATASET HISTORIS & PELATIHAN MODEL AI
# ==========================================
@st.cache_resource
def latih_model_ai(tarif_pln, konfig_umum_tuple, unit_magicom, unit_cuci, watt_standby):
    rng = np.random.default_rng(42)
    jumlah_hari = 120

    data_fitur = {}
    total_kwh_harian = np.zeros(jumlah_hari, dtype=float)

    for nama, watt, unit, normal_jam, std_jam in konfig_umum_tuple:
        if unit == 0 or watt == 0:
            sampel = np.zeros(jumlah_hari, dtype=float)
        else:
            sampel = rng.normal(loc=normal_jam * unit, scale=std_jam * np.sqrt(unit), size=jumlah_hari)
            sampel = np.clip(sampel, 0.0, 24.0 * unit)

        data_fitur[nama] = sampel
        total_kwh_harian += (sampel * watt) / 1000.0

    if unit_magicom == 0:
        sampel_cook = np.zeros(jumlah_hari, dtype=float)
        sampel_warm = np.zeros(jumlah_hari, dtype=float)
    else:
        sampel_cook = rng.normal(loc=0.75 * unit_magicom, scale=0.20 * np.sqrt(unit_magicom), size=jumlah_hari)
        sampel_cook = np.clip(sampel_cook, 0.0, 4.0 * unit_magicom)

        sampel_warm = rng.normal(loc=14.0 * unit_magicom, scale=3.5 * np.sqrt(unit_magicom), size=jumlah_hari)
        sampel_warm = np.clip(sampel_warm, 0.0, 24.0 * unit_magicom)

    data_fitur["Magicom_Cook"] = sampel_cook
    data_fitur["Magicom_Warm"] = sampel_warm
    total_kwh_harian += ((sampel_cook * WATT_MAGICOM_COOK) + (sampel_warm * WATT_MAGICOM_WARM)) / 1000.0

    if unit_cuci == 0:
        sampel_cuci = np.zeros(jumlah_hari, dtype=float)
    else:
        sampel_cuci = rng.normal(loc=0.5 * unit_cuci, scale=0.25 * np.sqrt(unit_cuci), size=jumlah_hari)
        sampel_cuci = np.clip(sampel_cuci, 0.0, 4.0 * unit_cuci)

    data_fitur["Mesin_Cuci"] = sampel_cuci
    total_kwh_harian += (sampel_cuci * WATT_MESIN_CUCI) / 1000.0

    kwh_standby_harian = (watt_standby * 24.0) / 1000.0
    total_kwh_harian += kwh_standby_harian

    df_hist = pd.DataFrame(data_fitur)
    tagihan_bulanan = total_kwh_harian * 30.0 * tarif_pln

    model_kewajaran = IsolationForest(
        n_estimators=100,
        contamination=0.06,
        random_state=42
    )
    model_kewajaran.fit(df_hist.to_numpy())

    model_regresi = LinearRegression()
    model_regresi.fit(total_kwh_harian.reshape(-1, 1), tagihan_bulanan)

    return df_hist, total_kwh_harian, tagihan_bulanan, model_kewajaran, model_regresi


# ==========================================
# 2. KOMPONEN HERO ANIMASI MODERN (SMART GRID & RUMAH)
# ==========================================
def render_animasi_dashboard(is_boros, indeks_wajar, total_kwh_hari, total_watt, batas_va, nama_golongan, tarif_pln):
    rasio_beban = float(np.clip(total_kwh_hari / 25.0, 0.08, 1.0))
    persen_mcb = int(np.clip((total_watt / max(batas_va, 1)) * 100, 0, 999))
    keliling_ring = 238.8
    offset_ring = keliling_ring * (1.0 - rasio_beban)

    durasi_aliran = 0.45 if is_boros else max(0.85, 2.2 - rasio_beban * 1.2)
    durasi_denyut = 0.7 if is_boros else 1.8

    if is_boros:
        warna_utama = "#f43f5e"
        warna_aksen = "#fbbf24"
        glow_rgba = "rgba(244, 63, 94, 0.22)"
        badge_bg = "rgba(244, 63, 94, 0.16)"
        judul_status = "PERINGATAN: TERDETEKSI PEMBOROSAN LISTRIK!"
        sub_status = "Ada peralatan yang menyala jauh lebih lama dari kebiasaan wajar. Cek cara hematnya pada tab pertama di bawah."
        label_meter = "BOROS"
        teks_indeks = "Perlu Penghematan"
    else:
        warna_utama = "#10b981"
        warna_aksen = "#38bdf8"
        glow_rgba = "rgba(16, 185, 129, 0.18)"
        badge_bg = "rgba(16, 185, 129, 0.16)"
        judul_status = "STATUS AMAN: PEMAKAIAN LISTRIK WAJAR & EFISIEN"
        sub_status = "Seluruh peralatan elektronik di rumah Anda bekerja dalam batas durasi harian yang wajar."
        label_meter = "EFISIEN"
        teks_indeks = "Wajar & Terkendali"

    render_html(f"""
    <style>
        @keyframes aliranListrik {{
            from {{ stroke-dashoffset: 36; }}
            to {{ stroke-dashoffset: 0; }}
        }}
        @keyframes denyutRadar {{
            0% {{ transform: scale(0.8); opacity: 0.95; }}
            75% {{ transform: scale(1.85); opacity: 0; }}
            100% {{ transform: scale(1.85); opacity: 0; }}
        }}
        @keyframes gelombangDaya {{
            0% {{ stroke-dashoffset: 120; }}
            100% {{ stroke-dashoffset: 0; }}
        }}
        @keyframes kilauBorder {{
            0%, 100% {{ box-shadow: 0 10px 30px {glow_rgba}; }}
            50% {{ box-shadow: 0 14px 38px rgba(56, 189, 248, 0.22); }}
        }}
        @keyframes kedipJendela {{
            0%, 100% {{ fill: {warna_aksen}; opacity: 0.9; }}
            50% {{ fill: #fef08a; opacity: 0.35; }}
        }}
        .hero-glass-banner {{
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(19, 28, 49, 0.92) 60%, rgba(15, 23, 42, 0.88) 100%);
            backdrop-filter: blur(16px);
            border: 1.5px solid {warna_utama};
            border-radius: 18px;
            padding: 20px 24px;
            color: #f8fafc;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            animation: kilauBorder 3.2s ease-in-out infinite;
            margin-bottom: 8px;
        }}
        .kabel-arus {{
            stroke-dasharray: 10 8;
            animation: aliranListrik {durasi_aliran:.2f}s linear infinite;
        }}
        .gelombang-radar {{
            transform-origin: 204px 16px;
            animation: denyutRadar {durasi_denyut * 1.4:.2f}s ease-out infinite;
        }}
        .gelombang-meter {{
            stroke-dasharray: 60 60;
            animation: gelombangDaya {durasi_aliran * 1.4:.2f}s linear infinite;
        }}
        .jendela-rumah {{
            animation: kedipJendela 2.2s ease-in-out infinite;
        }}
        @media (max-width: 640px) {{
            .judul-hero {{ font-size: 17px !important; }}
            .panel-svg {{ width: 100% !important; justify-content: center !important; }}
        }}
    </style>
    <div class="hero-glass-banner">
        <div style="flex: 1 1 320px;">
            <div style="display: inline-flex; align-items: center; gap: 7px; background-color: {badge_bg}; color: {warna_utama}; border: 1px solid {warna_utama}; padding: 4px 13px; border-radius: 999px; font-size: 11px; font-weight: 800; letter-spacing: 0.4px;">
                <span style="width: 8px; height: 8px; background: {warna_utama}; border-radius: 50%; display: inline-block;"></span>
                PLN {nama_golongan.split('(')[0].strip()} • TARIF Rp {tarif_pln:,.2f}/kWh
            </div>
            <div class="judul-hero" style="font-size: 21px; font-weight: 800; margin-top: 10px; color: {warna_utama}; line-height: 1.25;">
                {judul_status}
            </div>
            <div style="font-size: 13.5px; color: #cbd5e1; margin-top: 6px; line-height: 1.5;">
                {sub_status}
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px;">
                <span style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148,163,184,0.25); padding: 4px 11px; border-radius: 8px; font-size: 12px; color: #cbd5e1;">
                    Pola Pemakaian: <b style="color: {warna_utama};">{teks_indeks}</b>
                </span>
                <span style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148,163,184,0.25); padding: 4px 11px; border-radius: 8px; font-size: 12px; color: #cbd5e1;">
                    Beban Meteran: <b style="color: {warna_aksen};">{persen_mcb}%</b> dari {batas_va} VA
                </span>
            </div>
        </div>
        <div class="panel-svg" style="flex: 0 1 auto; display: flex; flex-wrap: wrap; align-items: center; gap: 12px;">
            <svg width="250" height="118" viewBox="0 0 265 124">
                <defs>
                    <linearGradient id="cardSvgBg" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stop-color="#070b14"/>
                        <stop offset="100%" stop-color="#11192c"/>
                    </linearGradient>
                </defs>
                <rect x="3" y="4" width="259" height="116" rx="14" fill="url(#cardSvgBg)" stroke="rgba(148,163,184,0.28)" stroke-width="1.5" />
                <circle cx="42" cy="61" r="22" fill="{badge_bg}" stroke="{warna_utama}" stroke-width="2" />
                <polygon points="44,46 34,62 42,62 39,76 52,59 43,59" fill="{warna_aksen}" />
                <text x="42" y="100" text-anchor="middle" fill="#94a3b8" font-size="9" font-weight="700">METER PLN</text>
                <line x1="66" y1="53" x2="144" y2="53" stroke="#1e293b" stroke-width="4" stroke-linecap="round" />
                <line class="kabel-arus" x1="66" y1="53" x2="144" y2="53" stroke="{warna_utama}" stroke-width="3.5" stroke-linecap="round" />
                <line x1="66" y1="69" x2="144" y2="69" stroke="#1e293b" stroke-width="4" stroke-linecap="round" />
                <line class="kabel-arus" x1="66" y1="69" x2="144" y2="69" stroke="{warna_aksen}" stroke-width="3" stroke-linecap="round" />
                <rect x="76" y="22" width="58" height="17" rx="8" fill="{badge_bg}" stroke="{warna_utama}" stroke-width="1" />
                <text x="105" y="34" text-anchor="middle" fill="{warna_utama}" font-size="8.5" font-weight="800">{label_meter}</text>
                <polygon points="150,51 204,19 252,51" fill="#1e293b" stroke="{warna_utama}" stroke-width="2.3" stroke-linejoin="round" />
                <rect x="160" y="51" width="82" height="50" rx="5" fill="#1e293b" stroke="{warna_utama}" stroke-width="2" />
                <rect x="171" y="59" width="60" height="23" rx="4" fill="#060911" stroke="#475569" stroke-width="1" />
                <polyline class="gelombang-meter" points="175,70 184,70 189,62 195,79 201,63 207,77 213,70 227,70" fill="none" stroke="{warna_utama}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />
                <rect class="jendela-rumah" x="174" y="87" width="14" height="9" rx="2" />
                <rect class="jendela-rumah" x="214" y="87" width="14" height="9" rx="2" />
                <circle class="gelombang-radar" cx="204" cy="16" r="6" fill="none" stroke="{warna_utama}" stroke-width="2" />
                <circle cx="204" cy="16" r="4" fill="{warna_utama}" />
                <text x="201" y="114" text-anchor="middle" fill="#94a3b8" font-size="9" font-weight="700">RUMAH ANDA</text>
            </svg>
            <svg width="108" height="118" viewBox="0 0 108 118">
                <rect x="2" y="4" width="104" height="112" rx="14" fill="#070b14" stroke="rgba(148,163,184,0.28)" stroke-width="1.5" />
                <circle cx="54" cy="54" r="38" fill="none" stroke="#1e293b" stroke-width="8" />
                <circle cx="54" cy="54" r="38" fill="none" stroke="{warna_utama}" stroke-width="8" stroke-dasharray="{keliling_ring:.1f}" stroke-dashoffset="{offset_ring:.1f}" stroke-linecap="round" transform="rotate(-90 54 54)" />
                <text x="54" y="52" text-anchor="middle" fill="#f8fafc" font-size="14" font-weight="800">{total_kwh_hari:.2f}</text>
                <text x="54" y="66" text-anchor="middle" fill="#94a3b8" font-size="9" font-weight="600">kWh/hari</text>
                <text x="54" y="104" text-anchor="middle" fill="{warna_aksen}" font-size="8.5" font-weight="800">ENERGI HARIAN</text>
            </svg>
        </div>
    </div>
    """)


# ==========================================
# HEADER UTAMA APLIKASI
# ==========================================
render_html(f"""
<div class="main-app-header">
    <div class="main-app-title">⚡ Smart Home AI: Asisten Pintar Pantau Listrik Rumah</div>
    <div class="live-pill">
        <span class="live-dot"></span>
        {'HASIL PERHITUNGAN' if st.session_state['sudah_hitung'] else 'FORMULIR PENGISIAN'}
    </div>
</div>
""")


# ==========================================
# HALAMAN 1: FORMULIR PENGISIAN LANGSUNG DI HALAMAN UTAMA
# ==========================================
if not st.session_state["sudah_hitung"]:
    render_html("""
    <div class="input-intro-banner">
        📋 <b>Silakan isi data listrik rumah Anda dari atas ke bawah.</b> Semua kolom awal bernilai kosong (0). Setelah selesai mengisi peralatan yang ada di rumah, tekan tombol <b>Hitung &amp; Lihat Hasil Perhitungan</b> di bagian paling bawah.
    </div>
    """)

    if st.button("🔄 Kosongkan / Reset Semua Isian", use_container_width=True):
        reset_semua_isian()
        st.rerun()

    # --- BAGIAN 1: DAYA LISTRIK & TARGET ANGGARAN ---
    render_html("""
    <div class="form-section-header">
        <div class="form-section-title">1. Daya Listrik &amp; Target Anggaran</div>
        <div class="form-section-sub">Wajib pilih golongan daya meteran PLN rumah Anda terlebih dahulu.</div>
    </div>
    """)

    opsi_pln_lengkap = [PLACEHOLDER_PLN] + list(GOLONGAN_PLN.keys())
    col_b1_a, col_b1_b = st.columns(2)
    with col_b1_a:
        pilihan_golongan = st.selectbox(
            "Golongan Daya Listrik (PLN) *",
            options=opsi_pln_lengkap,
            key="pilihan_golongan"
        )
    with col_b1_b:
        target_anggaran = st.number_input(
            "Target Maksimal Tagihan / Bulan (Rp — Opsional)",
            min_value=0,
            max_value=50000000,
            step=25000,
            key="target_anggaran",
            help="Biarkan 0 jika tidak ingin membatasi anggaran bulanan."
        )

    pilihan_standby = st.selectbox(
        "Colokan Kecil yang Terus Nempel 24 Jam (Wi-Fi, Charger HP, STB TV)",
        options=list(OPSI_STANDBY_AWAM.keys()),
        key="pilihan_standby"
    )

    # --- BAGIAN 2: PERALATAN ELEKTRONIK DI RUMAH ---
    render_html("""
    <div class="form-section-header">
        <div class="form-section-title">2. Peralatan Elektronik di Rumah</div>
        <div class="form-section-sub">Isi jumlah unit dan geser lama nyala per hari hanya untuk alat yang Anda miliki.</div>
    </div>
    """)

    col_dev_left, col_dev_right = st.columns(2)

    with col_dev_left:
        # KULKAS
        watt_kulkas_preview = OPSI_KULKAS[st.session_state["pilihan_kulkas"]]
        render_html(f"""
        <div class="device-box-label">
            <span class="device-box-name">❄️ Kulkas (Lemari Es)</span>
            <span class="device-box-watt">{watt_kulkas_preview} W/unit</span>
        </div>
        """)
        pilihan_kulkas = st.selectbox("Tipe Kulkas", options=list(OPSI_KULKAS.keys()), key="pilihan_kulkas", label_visibility="collapsed")
        ck1, ck2 = st.columns([1, 1.8])
        unit_kulkas = ck1.number_input("Jumlah Unit Kulkas", min_value=0, max_value=5, step=1, key="unit_kulkas")
        jam_kulkas_raw = ck2.slider("Nyala Kulkas (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, key="jam_kulkas", disabled=(unit_kulkas == 0))

        # AC
        watt_ac_preview = OPSI_AC[st.session_state["pilihan_ac"]]
        render_html(f"""
        <div class="device-box-label">
            <span class="device-box-name">🌬️ Air Conditioner (AC)</span>
            <span class="device-box-watt">{watt_ac_preview} W/unit</span>
        </div>
        """)
        pilihan_ac = st.selectbox("Kapasitas AC", options=list(OPSI_AC.keys()), key="pilihan_ac", label_visibility="collapsed")
        ca1, ca2 = st.columns([1, 1.8])
        unit_ac = ca1.number_input("Jumlah Unit AC", min_value=0, max_value=5, step=1, key="unit_ac")
        jam_ac_raw = ca2.slider("Nyala AC (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, key="jam_ac", disabled=(unit_ac == 0))

        # MAGICOM
        render_html(f"""
        <div class="device-box-label">
            <span class="device-box-name">🍚 Magicom (Penanak Nasi)</span>
            <span class="device-box-watt">Masak {WATT_MAGICOM_COOK}W / Hangat {WATT_MAGICOM_WARM}W</span>
        </div>
        """)
        cm1, cm2 = st.columns([1, 1.8])
        unit_magicom = cm1.number_input("Jumlah Unit Magicom", min_value=0, max_value=5, step=1, key="unit_magicom")
        masak_kali = cm2.slider("Masak Nasi (Kali/Hari)", min_value=0, max_value=4, step=1, key="masak_magicom", disabled=(unit_magicom == 0))
        warm_jam = st.slider("Lama Mode Menghangatkan (Jam/Hari)", min_value=0.0, max_value=24.0, step=1.0, key="warm_magicom", disabled=(unit_magicom == 0))

        # MESIN CUCI
        render_html(f"""
        <div class="device-box-label">
            <span class="device-box-name">🧺 Mesin Cuci</span>
            <span class="device-box-watt">{WATT_MESIN_CUCI} W/unit</span>
        </div>
        """)
        cc1, cc2 = st.columns([1, 1.8])
        unit_cuci = cc1.number_input("Jumlah Unit Mesin Cuci", min_value=0, max_value=3, step=1, key="unit_cuci")
        jam_cuci_minggu = cc2.slider("Total Jam Cuci / Minggu", min_value=0.0, max_value=21.0, step=0.5, key="jam_cuci_minggu", disabled=(unit_cuci == 0))

    with col_dev_right:
        ikon_perangkat = {
            "Lampu LED (Per Titik)": "💡",
            "Kipas Angin": "🌀",
            "TV LED": "📺",
            "Sanyo (Pompa Air Otomatis)": "💧",
            "Water Heater": "🚿",
        }
        for idx, (nama, spec) in enumerate(PERANGKAT_LAIN.items()):
            label_satuan = "W/lampu" if nama == "Lampu LED (Per Titik)" else "W/unit"
            ikon = ikon_perangkat.get(nama, "🔌")
            render_html(f"""
            <div class="device-box-label">
                <span class="device-box-name">{ikon} {nama}</span>
                <span class="device-box-watt">{spec['watt']} {label_satuan}</span>
            </div>
            """)
            c_unit, c_jam = st.columns([1, 1.8])
            lbl_unit = "Jumlah Lampu" if nama == "Lampu LED (Per Titik)" else "Jumlah Unit"
            unit_val = c_unit.number_input(lbl_unit, min_value=0, max_value=spec["max_unit"], step=1, key=f"unit_{idx}")
            c_jam.slider("Lama Nyala (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, key=f"jam_{idx}", disabled=(unit_val == 0))

    # --- BAGIAN 3: ALAT TAMBAHAN LAINNYA (OPSIONAL) ---
    render_html("""
    <div class="form-section-header">
        <div class="form-section-title">3. Alat Tambahan Lainnya (Opsional)</div>
        <div class="form-section-sub">Isi jika ada alat lain seperti Setrika, PC, atau Dispenser. Biarkan 0 jika tidak ada.</div>
    </div>
    """)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        raw_nama_c1 = st.text_input("Nama Alat Tambahan 1", placeholder="Contoh: Setrika Listrik", key="nama_c1")
        cw1, cu1 = st.columns(2)
        watt_c1 = cw1.number_input("Daya Alat 1 (Watt)", min_value=0, max_value=5000, step=5, key="watt_c1")
        unit_c1 = cu1.number_input("Unit Alat 1", min_value=0, max_value=10, step=1, key="unit_c1")
        st.slider("Lama Nyala Alat 1 (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, disabled=(unit_c1 == 0 or watt_c1 == 0), key="jam_c1")

    with col_c2:
        raw_nama_c2 = st.text_input("Nama Alat Tambahan 2", placeholder="Contoh: PC / Komputer", key="nama_c2")
        cw2, cu2 = st.columns(2)
        watt_c2 = cw2.number_input("Daya Alat 2 (Watt)", min_value=0, max_value=5000, step=5, key="watt_c2")
        unit_c2 = cu2.number_input("Unit Alat 2", min_value=0, max_value=10, step=1, key="unit_c2")
        st.slider("Lama Nyala Alat 2 (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, disabled=(unit_c2 == 0 or watt_c2 == 0), key="jam_c2")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Validasi jumlah input terisi
    golongan_ok = (st.session_state["pilihan_golongan"] != PLACEHOLDER_PLN)
    jumlah_alat_ok = 0
    if st.session_state["unit_kulkas"] > 0 and st.session_state["jam_kulkas"] > 0:
        jumlah_alat_ok += 1
    if st.session_state["unit_ac"] > 0 and st.session_state["jam_ac"] > 0:
        jumlah_alat_ok += 1
    if st.session_state["unit_magicom"] > 0 and (st.session_state["masak_magicom"] > 0 or st.session_state["warm_magicom"] > 0):
        jumlah_alat_ok += 1
    if st.session_state["unit_cuci"] > 0 and st.session_state["jam_cuci_minggu"] > 0:
        jumlah_alat_ok += 1
    for idx_cek in range(len(PERANGKAT_LAIN)):
        if st.session_state[f"unit_{idx_cek}"] > 0 and st.session_state[f"jam_{idx_cek}"] > 0:
            jumlah_alat_ok += 1
    if st.session_state["unit_c1"] > 0 and st.session_state["watt_c1"] > 0 and st.session_state["jam_c1"] > 0:
        jumlah_alat_ok += 1
    if st.session_state["unit_c2"] > 0 and st.session_state["watt_c2"] > 0 and st.session_state["jam_c2"] > 0:
        jumlah_alat_ok += 1
    if OPSI_STANDBY_AWAM[st.session_state["pilihan_standby"]] > 0:
        jumlah_alat_ok += 1

    if st.button("🚀 Hitung & Lihat Hasil Perhitungan Sekarang", type="primary", use_container_width=True):
        if not golongan_ok:
            st.error("⚠️ Silakan pilih **Golongan Daya Listrik (PLN)** terlebih dahulu pada **Bagian 1** di atas.")
        elif jumlah_alat_ok == 0:
            st.error("⚠️ Data peralatan masih kosong. Silakan isi minimal **1 peralatan elektronik** (jumlah unit dan lama nyala per hari) pada **Bagian 2 atau 3** di atas.")
        else:
            st.session_state["sudah_hitung"] = True
            st.session_state["perlu_scroll_atas"] = True
            st.rerun()

else:
    # ==========================================
    # HALAMAN 2: OUTPUT PERHITUNGAN LENGKAP
    # ==========================================
    pilihan_golongan = st.session_state["pilihan_golongan"]
    target_anggaran = st.session_state["target_anggaran"]
    pilihan_standby = st.session_state["pilihan_standby"]
    watt_standby = OPSI_STANDBY_AWAM[pilihan_standby]

    pilihan_kulkas = st.session_state["pilihan_kulkas"]
    watt_kulkas = OPSI_KULKAS[pilihan_kulkas]
    unit_kulkas = st.session_state["unit_kulkas"]
    jam_kulkas = st.session_state["jam_kulkas"] if unit_kulkas > 0 else 0.0

    pilihan_ac = st.session_state["pilihan_ac"]
    watt_ac = OPSI_AC[pilihan_ac]
    unit_ac = st.session_state["unit_ac"]
    jam_ac = st.session_state["jam_ac"] if unit_ac > 0 else 0.0

    unit_magicom = st.session_state["unit_magicom"]
    masak_kali = st.session_state["masak_magicom"] if unit_magicom > 0 else 0
    warm_jam = st.session_state["warm_magicom"] if unit_magicom > 0 else 0.0

    unit_cuci = st.session_state["unit_cuci"]
    jam_cuci_minggu = st.session_state["jam_cuci_minggu"] if unit_cuci > 0 else 0.0

    input_unit_lain = {}
    input_jam_lain = {}
    for idx_read, nama_read in enumerate(PERANGKAT_LAIN.keys()):
        u_val = st.session_state[f"unit_{idx_read}"]
        j_val = st.session_state[f"jam_{idx_read}"] if u_val > 0 else 0.0
        input_unit_lain[nama_read] = u_val
        input_jam_lain[nama_read] = j_val

    nama_c1 = st.session_state["nama_c1"].strip() or "Alat Tambahan 1"
    watt_c1 = st.session_state["watt_c1"]
    unit_c1 = st.session_state["unit_c1"]
    jam_c1 = st.session_state["jam_c1"] if (unit_c1 > 0 and watt_c1 > 0) else 0.0

    nama_c2 = st.session_state["nama_c2"].strip() or "Alat Tambahan 2"
    if nama_c2 == nama_c1:
        nama_c2 = f"{nama_c2} (2)"
    watt_c2 = st.session_state["watt_c2"]
    unit_c2 = st.session_state["unit_c2"]
    jam_c2 = st.session_state["jam_c2"] if (unit_c2 > 0 and watt_c2 > 0) else 0.0

    batas_va = GOLONGAN_PLN[pilihan_golongan]["va"]
    TARIF_PLN = GOLONGAN_PLN[pilihan_golongan]["tarif"]

    # Tombol Navigasi Cepat Kembali ke Pengisian Input / Reset
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("✏️ Ubah / Edit Data Input", use_container_width=True):
            st.session_state["sudah_hitung"] = False
            st.session_state["perlu_scroll_atas"] = True
            st.rerun()
    with col_nav2:
        if st.button("🔄 Kosongkan / Reset Semua", use_container_width=True):
            reset_semua_isian()
            st.rerun()

    PERANGKAT_UMUM = {
        pilihan_kulkas: {"watt": watt_kulkas, "normal_jam": 24.0, "std_jam": 1.5, "is_custom": False},
        pilihan_ac: {"watt": watt_ac, "normal_jam": 7.5, "std_jam": 1.2, "is_custom": False},
    }
    for k, v in PERANGKAT_LAIN.items():
        PERANGKAT_UMUM[k] = {"watt": v["watt"], "normal_jam": v["normal_jam"], "std_jam": v["std_jam"], "is_custom": False}

    PERANGKAT_UMUM[nama_c1] = {"watt": watt_c1, "normal_jam": 2.5, "std_jam": 1.0, "is_custom": True}
    PERANGKAT_UMUM[nama_c2] = {"watt": watt_c2, "normal_jam": 4.0, "std_jam": 1.5, "is_custom": True}

    input_unit_umum = {pilihan_kulkas: unit_kulkas, pilihan_ac: unit_ac}
    input_unit_umum.update(input_unit_lain)
    input_unit_umum[nama_c1] = unit_c1 if watt_c1 > 0 else 0
    input_unit_umum[nama_c2] = unit_c2 if watt_c2 > 0 else 0

    input_jam_umum = {pilihan_kulkas: jam_kulkas, pilihan_ac: jam_ac}
    input_jam_umum.update(input_jam_lain)
    input_jam_umum[nama_c1] = jam_c1
    input_jam_umum[nama_c2] = jam_c2

    konfig_umum_tuple = tuple(
        (nama, spec["watt"], input_unit_umum[nama], spec["normal_jam"], spec["std_jam"])
        for nama, spec in PERANGKAT_UMUM.items()
    )

    df_hist, hist_kwh, hist_tagihan, model_kewajaran, model_regresi = latih_model_ai(
        TARIF_PLN, konfig_umum_tuple, unit_magicom, unit_cuci, watt_standby
    )

    jam_cook_aktual = (masak_kali * JAM_PER_MASAK) if unit_magicom > 0 else 0.0
    jam_warm_aktual = warm_jam if unit_magicom > 0 else 0.0
    jam_cuci_harian = (jam_cuci_minggu / 7.0) if unit_cuci > 0 else 0.0

    fitur_list = [input_jam_umum[nama] * input_unit_umum[nama] for nama in PERANGKAT_UMUM.keys()]
    fitur_list.extend([
        jam_cook_aktual * unit_magicom,
        jam_warm_aktual * unit_magicom,
        jam_cuci_harian * unit_cuci
    ])
    fitur_input = np.array([fitur_list], dtype=float)

    total_unit_aktif = sum(input_unit_umum.values()) + unit_magicom + unit_cuci
    if total_unit_aktif == 0:
        status_pred = 1
        indeks_wajar = 0.1
    else:
        status_pred = model_kewajaran.predict(fitur_input)[0]
        indeks_wajar = model_kewajaran.decision_function(fitur_input)[0]

    is_boros = (status_pred == -1)

    DETAIL_DIAGNOSIS = {
        pilihan_kulkas: {
            "penyebab": "Pintu kulkas terlalu sering dibuka-tutup, karet pintu mulai renggang, atau makanan panas langsung dimasukkan sehingga mesin pendingin bekerja terus.",
            "solusi": "Tunggu makanan panas hingga dingin sebelum dimasukkan, pastikan pintu tertutup rapat, dan beri jarak minimal 10 cm dari dinding belakang."
        },
        pilihan_ac: {
            "penyebab": "Mesin AC bekerja terlalu lama dalam sehari atau suhu disetel terlalu dingin (di bawah 22°C) sehingga kompresor terus menarik daya penuh.",
            "solusi": "Atur suhu ideal di <b>24°C–25°C</b>, aktifkan fitur <i>Timer/Sleep</i> saat dini hari, pastikan pintu dan jendela tertutup rapat, serta bersihkan filter AC secara rutin."
        },
        "Lampu LED (Per Titik)": {
            "penyebab": "Banyak titik lampu dibiarkan menyala pada siang hari atau tetap hidup di ruangan yang sedang kosong.",
            "solusi": "Matikan lampu ruangan yang tidak dipakai, manfaatkan cahaya matahari lewat jendela di siang hari, dan gunakan sensor otomatis untuk lampu teras."
        },
        "Kipas Angin": {
            "penyebab": "Kipas angin dibiarkan berputar terus-menerus meski tidak ada orang di ruangan, atau debu menumpuk pada baling-baling sehingga putaran motor berat.",
            "solusi": "Matikan kipas saat meninggalkan ruangan, gunakan tingkat kecepatan sedang, dan bersihkan debu baling-baling secara berkala."
        },
        "Sanyo (Pompa Air Otomatis)": {
            "penyebab": "Pompa air otomatis terlalu sering hidup-mati sendiri akibat kebocoran halus pada pipa/keran air atau kerusakan pelampung otomatis toren.",
            "solusi": "Periksa seluruh keran air dan sambungan pipa dari kebocoran menetes, serta cek pelampung otomatis toren agar pompa hanya menyala saat air benar-benar habis."
        },
        "TV LED": {
            "penyebab": "TV dibiarkan menyala seharian tanpa ditonton atau terus berada dalam mode <i>standby</i> saat malam hari.",
            "solusi": "Matikan TV melalui tombol daya utama atau cabut steker setelah selesai menonton, dan aktifkan fitur <i>Auto Sleep</i>."
        },
        "Water Heater": {
            "penyebab": "Pemanas air dibiarkan menyala sepanjang hari untuk menjaga suhu air tetap panas meski tidak sedang dipakai mandi.",
            "solusi": "Nyalakan pemanas air hanya <b>15–20 menit sebelum mandi</b>, lalu matikan kembali saklarnya setelah selesai digunakan."
        },
        "Magicom (Penanak Nasi)": {
            "penyebab": "Frekuensi memasak nasi (daya tinggi 380 W) dilakukan berkali-kali dalam sehari sehingga menarik lonjakan daya besar berulang.",
            "solusi": "Masak nasi sekaligus 1 kali di pagi hari untuk kebutuhan seharian, dan segera cabut steker Magicom bila nasi sudah habis di malam hari."
        },
        "Mesin Cuci": {
            "penyebab": "Durasi pencucian dalam seminggu terlalu panjang karena mencuci pakaian sedikit demi sedikit hampir setiap hari.",
            "solusi": "Kumpulkan pakaian kotor dan cuci sekaligus 1–2 kali seminggu sesuai kapasitas maksimal tabung mesin cuci."
        }
    }

    DEFAULT_DIAG_KUSTOM = {
        "penyebab": "Durasi pemakaian alat tambahan ini terlalu lama dalam sehari sehingga akumulasi kWh hariannya melonjak.",
        "solusi": "Batasi penggunaan alat hanya saat benar-benar dibutuhkan dan matikan/cabut dari stopkontak setelah selesai dipakai."
    }

    rincian_list = []
    total_kwh_hari = 0.0
    total_watt_terpasang = 0
    alat_aktif_daya = []
    diagnosis_boros_list = []
    potensi_hemat_rp_total = 0.0

    for nama, spec in PERANGKAT_UMUM.items():
        unit = input_unit_umum[nama]
        jam = input_jam_umum[nama]
        daya_total_alat = unit * spec["watt"]
        total_watt_terpasang += daya_total_alat

        if unit > 0:
            alat_aktif_daya.append((nama, daya_total_alat))

        kwh_hari = (daya_total_alat * jam) / 1000.0
        biaya_bulan = (kwh_hari * 30.0) * TARIF_PLN
        total_kwh_hari += kwh_hari

        if unit > 0 and spec["normal_jam"] < 24.0:
            z_score = (jam - spec["normal_jam"]) / spec["std_jam"]
            if z_score > 2.0:
                selisih_biaya = ((jam - spec["normal_jam"]) * daya_total_alat / 1000.0) * 30.0 * TARIF_PLN
                potensi_hemat_rp_total += selisih_biaya
                satuan_txt = "lampu" if nama == "Lampu LED (Per Titik)" else "unit"
                info_diag = DETAIL_DIAGNOSIS.get(nama, DEFAULT_DIAG_KUSTOM)

                diagnosis_boros_list.append({
                    "judul": f"{nama} ({unit} {satuan_txt}) — Menyala {jam:.1f} jam/hari (Batas wajar {spec['normal_jam']} jam/hari)",
                    "dampak": f"Menambah pengeluaran listrik sekitar <b>+Rp {selisih_biaya:,.0f}/bulan</b> (atau <b>+Rp {selisih_biaya * 12:,.0f}/tahun</b>).",
                    "penyebab": info_diag["penyebab"],
                    "solusi": info_diag["solusi"]
                })

        rincian_list.append({
            "Peralatan Rumah": nama,
            "Jumlah": unit,
            "Daya (Watt)": f"{spec['watt']} W",
            "Beban Puncak (W)": daya_total_alat,
            "Lama Pakai": f"{jam:.1f} jam/hari",
            "Energi Harian (kWh)": round(kwh_hari, 2),
            "Biaya Listrik/Bulan (Rp)": round(biaya_bulan, 0)
        })

    daya_puncak_magicom = unit_magicom * WATT_MAGICOM_COOK
    total_watt_terpasang += daya_puncak_magicom
    if unit_magicom > 0:
        alat_aktif_daya.append(("Magicom (saat memasak)", daya_puncak_magicom))

    kwh_cook = (unit_magicom * WATT_MAGICOM_COOK * jam_cook_aktual) / 1000.0
    kwh_warm = (unit_magicom * WATT_MAGICOM_WARM * jam_warm_aktual) / 1000.0
    kwh_magicom_total = kwh_cook + kwh_warm
    biaya_magicom_bulan = (kwh_magicom_total * 30.0) * TARIF_PLN
    total_kwh_hari += kwh_magicom_total

    if unit_magicom > 0 and masak_kali >= 3:
        hemat_masak = ((masak_kali - 1) * JAM_PER_MASAK * unit_magicom * WATT_MAGICOM_COOK / 1000.0) * 30.0 * TARIF_PLN
        potensi_hemat_rp_total += hemat_masak
        info_diag = DETAIL_DIAGNOSIS["Magicom (Penanak Nasi)"]
        diagnosis_boros_list.append({
            "judul": f"Magicom ({unit_magicom} unit) — Memasak nasi {masak_kali}x/hari (Normalnya 1x/hari)",
            "dampak": f"Menambah pengeluaran listrik sekitar <b>+Rp {hemat_masak:,.0f}/bulan</b> (atau <b>+Rp {hemat_masak * 12:,.0f}/tahun</b>).",
            "penyebab": info_diag["penyebab"],
            "solusi": info_diag["solusi"]
        })

    rincian_list.append({
        "Peralatan Rumah": "Magicom (Penanak Nasi)",
        "Jumlah": unit_magicom,
        "Daya (Watt)": f"{WATT_MAGICOM_COOK}W / {WATT_MAGICOM_WARM}W",
        "Beban Puncak (W)": daya_puncak_magicom,
        "Lama Pakai": f"Masak {masak_kali}x + Hangat {jam_warm_aktual:.0f}j",
        "Energi Harian (kWh)": round(kwh_magicom_total, 2),
        "Biaya Listrik/Bulan (Rp)": round(biaya_magicom_bulan, 0)
    })

    daya_puncak_cuci = unit_cuci * WATT_MESIN_CUCI
    total_watt_terpasang += daya_puncak_cuci
    if unit_cuci > 0:
        alat_aktif_daya.append(("Mesin Cuci", daya_puncak_cuci))

    kwh_cuci_harian = (daya_puncak_cuci * jam_cuci_harian) / 1000.0
    biaya_cuci_bulan = (kwh_cuci_harian * 30.0) * TARIF_PLN
    total_kwh_hari += kwh_cuci_harian

    if unit_cuci > 0 and jam_cuci_minggu >= 8.0:
        hemat_cuci = (((jam_cuci_minggu - 3.5) / 7.0) * daya_puncak_cuci / 1000.0) * 30.0 * TARIF_PLN
        potensi_hemat_rp_total += hemat_cuci
        info_diag = DETAIL_DIAGNOSIS["Mesin Cuci"]
        diagnosis_boros_list.append({
            "judul": f"Mesin Cuci ({unit_cuci} unit) — Dipakai {jam_cuci_minggu:.1f} jam/minggu (Normalnya 2–4 jam/minggu)",
            "dampak": f"Menambah pengeluaran listrik sekitar <b>+Rp {hemat_cuci:,.0f}/bulan</b> (atau <b>+Rp {hemat_cuci * 12:,.0f}/tahun</b>).",
            "penyebab": info_diag["penyebab"],
            "solusi": info_diag["solusi"]
        })

    rincian_list.append({
        "Peralatan Rumah": "Mesin Cuci",
        "Jumlah": unit_cuci,
        "Daya (Watt)": f"{WATT_MESIN_CUCI} W",
        "Beban Puncak (W)": daya_puncak_cuci,
        "Lama Pakai": f"{jam_cuci_minggu:.1f} jam/minggu",
        "Energi Harian (kWh)": round(kwh_cuci_harian, 2),
        "Biaya Listrik/Bulan (Rp)": round(biaya_cuci_bulan, 0)
    })

    if watt_standby > 0:
        total_watt_terpasang += watt_standby
        kwh_standby_hari = (watt_standby * 24.0) / 1000.0
        biaya_standby_bulan = (kwh_standby_hari * 30.0) * TARIF_PLN
        total_kwh_hari += kwh_standby_hari

        rincian_list.append({
            "Peralatan Rumah": "Colokan Kecil 24 Jam (Wi-Fi, Charger, STB)",
            "Jumlah": 1,
            "Daya (Watt)": f"{watt_standby} W",
            "Beban Puncak (W)": watt_standby,
            "Lama Pakai": "24.0 jam/hari",
            "Energi Harian (kWh)": round(kwh_standby_hari, 2),
            "Biaya Listrik/Bulan (Rp)": round(biaya_standby_bulan, 0)
        })

    if is_boros and len(diagnosis_boros_list) == 0:
        for nama, spec in PERANGKAT_UMUM.items():
            if input_unit_umum[nama] > 0 and spec["normal_jam"] < 24.0 and input_jam_umum[nama] > spec["normal_jam"] * 1.2:
                selisih_biaya = ((input_jam_umum[nama] - spec["normal_jam"]) * input_unit_umum[nama] * spec["watt"] / 1000.0) * 30.0 * TARIF_PLN
                potensi_hemat_rp_total += selisih_biaya
                info_diag = DETAIL_DIAGNOSIS.get(nama, DEFAULT_DIAG_KUSTOM)
                diagnosis_boros_list.append({
                    "judul": f"{nama} ({input_unit_umum[nama]} unit) — Menyala {input_jam_umum[nama]:.1f} jam/hari (Normalnya {spec['normal_jam']} jam/hari)",
                    "dampak": f"Berkontribusi menaikkan pengeluaran sekitar <b>+Rp {selisih_biaya:,.0f}/bulan</b>.",
                    "penyebab": info_diag["penyebab"],
                    "solusi": info_diag["solusi"]
                })

    df_rincian = pd.DataFrame(rincian_list)

    total_biaya_murni = df_rincian["Biaya Listrik/Bulan (Rp)"].sum()
    if total_biaya_murni > 0:
        df_rincian["Porsi Beban (%)"] = (df_rincian["Biaya Listrik/Bulan (Rp)"] / total_biaya_murni * 100).round(1)
    else:
        df_rincian["Porsi Beban (%)"] = 0.0

    if total_kwh_hari == 0:
        prediksi_tagihan_murni = 0.0
    else:
        prediksi_tagihan_murni = float(model_regresi.predict([[total_kwh_hari]])[0])
        prediksi_tagihan_murni = max(0.0, prediksi_tagihan_murni)

    total_kwh_bulan = total_kwh_hari * 30.0
    potensi_hemat_tahun = potensi_hemat_rp_total * 12.0

    render_animasi_dashboard(is_boros, indeks_wajar, total_kwh_hari, total_watt_terpasang, batas_va, pilihan_golongan, TARIF_PLN)

    if total_watt_terpasang > batas_va:
        alat_aktif_daya.sort(key=lambda item: item[1], reverse=True)
        top_alat_str = [f"**{n} ({w:,} W)**" for n, w in alat_aktif_daya[:4]]

        if len(top_alat_str) > 1:
            daftar_alat_warning = ", ".join(top_alat_str[:-1]) + ", dan " + top_alat_str[-1]
            kalimat_hindari = f"Hindari menyalakan {daftar_alat_warning} secara bersamaan agar listrik tidak turun (*jeglek*)."
        elif len(top_alat_str) == 1:
            kalimat_hindari = f"Daya {top_alat_str[0]} sudah melebihi batas meteran listrik rumah Anda sehingga berisiko membuat listrik turun (*jeglek*)."
        else:
            kalimat_hindari = "Kurangi beban pemakaian secara bersamaan agar listrik tidak turun (*jeglek*)."

        st.warning(
            f"**Peringatan Beban Maksimal Meteran ({batas_va} VA):** Jika seluruh alat aktif dinyalakan serentak, total dayanya mencapai "
            f"**{total_watt_terpasang:,} Watt** (melebihi batas **{batas_va} VA**). {kalimat_hindari}"
        )

    rasio_default_bersih = ((100000.0 - 2500.0) * 0.92) / 100000.0
    estimasi_belanja_awal = prediksi_tagihan_murni / rasio_default_bersih

    render_html(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Total Daya Terpasang</span>
                <span class="kpi-icon">🔌</span>
            </div>
            <div class="kpi-value">{total_watt_terpasang:,} W</div>
            <div class="kpi-sub">Batas Meteran: {batas_va} VA</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Pemakaian per Hari</span>
                <span class="kpi-icon">☀️</span>
            </div>
            <div class="kpi-value">{total_kwh_hari:.2f} kWh</div>
            <div class="kpi-sub">Rata-rata 24 jam</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Pemakaian per Bulan</span>
                <span class="kpi-icon">📅</span>
            </div>
            <div class="kpi-value">{total_kwh_bulan:.1f} kWh</div>
            <div class="kpi-sub">Akumulasi 30 hari</div>
        </div>
        <div class="kpi-card" style="border-color: {'rgba(244,63,94,0.6)' if is_boros else 'rgba(16,185,129,0.6)'};">
            <div class="kpi-top">
                <span class="kpi-label">Estimasi Biaya / Bulan</span>
                <span class="kpi-icon">💰</span>
            </div>
            <div class="kpi-value" style="color: {'#fda4af' if is_boros else '#6ee7b7'};">Rp {estimasi_belanja_awal:,.0f}</div>
            <div class="kpi-sub">Listrik murni: Rp {prediksi_tagihan_murni:,.0f}</div>
        </div>
    </div>
    """)

    tab_ringkasan, tab_grafik, tab_tabel = st.tabs([
        "⚡ Simulasi Token, Anggaran & Solusi Hemat",
        "📊 Grafik Analisis & Tren Biaya",
        "📋 Rincian Per Alat & Unduh Laporan"
    ])

    with tab_ringkasan:
        col_token, col_budget = st.columns(2)

        with col_token:
            tk1, tk2 = st.columns([1.3, 1])
            nominal_token = tk1.selectbox(
                "Simulasi Isi Pulsa / Token Listrik",
                options=[50000, 100000, 200000, 500000, 1000000],
                index=1,
                format_func=lambda x: f"Beli Token Rp {x:,.0f}"
            )
            persen_ppj = tk2.slider("Pajak Daerah / PPJ (%)", min_value=3, max_value=10, value=8, step=1, help="Pajak Penerangan Jalan (PPJ) dipotong otomatis saat beli token (di Semarang/Jawa Tengah sekitar 8%).")

            biaya_admin_bank = 2500.0
            saldo_setelah_admin = max(0.0, nominal_token - biaya_admin_bank)
            rupiah_bersih_token = saldo_setelah_admin * (1.0 - (persen_ppj / 100.0))
            kwh_didapat = rupiah_bersih_token / TARIF_PLN

            if total_kwh_hari > 0:
                estimasi_hari_token = kwh_didapat / total_kwh_hari
                teks_bertahan = f"<span style='color:#38bdf8; font-size:20px; font-weight:800;'>±{estimasi_hari_token:.1f} Hari</span>"
                persen_sisa_visual = min(100.0, (estimasi_hari_token / 30.0) * 100.0)
            else:
                teks_bertahan = "<span style='color:#38bdf8; font-weight:800;'>Tidak terpakai (0 kWh/hari)</span>"
                persen_sisa_visual = 100.0

            rasio_kotor_ke_bersih = rupiah_bersih_token / max(nominal_token, 1)
            estimasi_belanja_token_bulan = prediksi_tagihan_murni / max(rasio_kotor_ke_bersih, 0.5)

            render_html(f"""
            <div class="glass-card">
                <div class="feature-title">🔋 Ketahanan Pulsa / Token Listrik Prabayar</div>
                <div style="font-size:13.5px; color:#cbd5e1; line-height:1.6;">
                    • Saldo listrik murni (potong Admin Rp 2.500 & PPJ {persen_ppj}%): <b>Rp {rupiah_bersih_token:,.0f}</b> (<b>{kwh_didapat:.2f} kWh</b>)<br>
                    • Token <b>Rp {nominal_token:,.0f}</b> diperkirakan habis dalam: {teks_bertahan}
                </div>
                <div class="progress-track">
                    <div class="progress-fill-anim" style="width:{persen_sisa_visual:.1f}%; background: linear-gradient(90deg, #0284c7, #38bdf8, #818cf8, #0284c7);"></div>
                </div>
                <div style="font-size:12px; color:#94a3b8;">
                    Setara pengeluaran beli token: <b>±Rp {estimasi_belanja_token_bulan:,.0f} / bulan</b> (termasuk PPJ {persen_ppj}% & admin).
                </div>
            </div>
            """)

        with col_budget:
            if target_anggaran > 0:
                persen_anggaran = (estimasi_belanja_token_bulan / max(target_anggaran, 1)) * 100.0
                lebar_bar = min(persen_anggaran, 100.0)
                selisih_anggaran = target_anggaran - estimasi_belanja_token_bulan

                if selisih_anggaran >= 0:
                    warna_bar_budget = "linear-gradient(90deg, #059669, #34d399, #10b981, #059669)"
                    status_budget_txt = f"<span style='color:#34d399; font-weight:800;'>MASUK ANGGARAN</span> — Masih hemat <b>Rp {selisih_anggaran:,.0f}/bulan</b> dari batas maksimal Anda."
                else:
                    warna_bar_budget = "linear-gradient(90deg, #e11d48, #fb7185, #f43f5e, #e11d48)"
                    status_budget_txt = f"<span style='color:#fda4af; font-weight:800;'>MELEBIHI ANGGARAN</span> — Lebih besar <b>+Rp {abs(selisih_anggaran):,.0f}/bulan</b> (atau <b>+Rp {abs(selisih_anggaran) * 12:,.0f}/tahun</b>)."

                render_html(f"""
                <div class="glass-card">
                    <div class="feature-title">🎯 Kesesuaian dengan Target Anggaran Bulanan</div>
                    <div style="display:flex; justify-content:space-between; font-size:13.5px; color:#cbd5e1; margin-top:8px;">
                        <span>Estimasi Belanja: <b>Rp {estimasi_belanja_token_bulan:,.0f}</b></span>
                        <span>Target Anda: <b>Rp {target_anggaran:,.0f}</b> ({persen_anggaran:.1f}%)</span>
                    </div>
                    <div class="progress-track" style="margin-top:16px; margin-bottom:16px;">
                        <div class="progress-fill-anim" style="width:{lebar_bar:.1f}%; background: {warna_bar_budget};"></div>
                    </div>
                    <div style="font-size:13.5px; color:#e2e8f0; line-height:1.55;">
                        {status_budget_txt}
                    </div>
                </div>
                """)
            else:
                render_html(f"""
                <div class="glass-card">
                    <div class="feature-title">🎯 Target Anggaran Bulanan (Opsional)</div>
                    <div style="font-size:13.5px; color:#cbd5e1; line-height:1.6; margin-top:8px;">
                        Anda belum mengisi batas maksimal anggaran bulanan (nilai saat ini <b>Rp 0</b>).<br>
                        Jika ingin membandingkan apakah tagihan <b>±Rp {estimasi_belanja_token_bulan:,.0f}/bulan</b> masuk anggaran Anda, tekan tombol <b>Ubah / Edit Data Input</b> di atas.
                    </div>
                </div>
                """)

        if is_boros or len(diagnosis_boros_list) > 0:
            st.markdown("#### 🚨 Daftar Pemakaian Berlebih (Boros) & Cara Mengatasinya")

            for item in diagnosis_boros_list:
                render_html(f"""
                <div class="diag-card">
                    <div class="diag-header">⚠️ {item['judul']}</div>
                    <div class="diag-row">• <b>Dampak Pengeluaran:</b> {item['dampak']}</div>
                    <div class="diag-row">• <b>Penyebab Utama:</b> {item['penyebab']}</div>
                    <div class="diag-row" style="color: #86efac;">• <b>Cara Mengatasi:</b> {item['solusi']}</div>
                </div>
                """)

            if potensi_hemat_rp_total > 0:
                st.success(
                    f"**Total Uang yang Bisa Dihemat:** Dengan menerapkan langkah penghematan di atas, pengeluaran listrik rumah Anda bisa berkurang sekitar "
                    f"**Rp {potensi_hemat_rp_total:,.0f}/bulan** atau **Rp {potensi_hemat_tahun:,.0f}/tahun**!"
                )
        else:
            df_aktif_sort = df_rincian[df_rincian["Energi Harian (kWh)"] > 0].sort_values(by="Energi Harian (kWh)", ascending=False)
            alat_terbesar = df_aktif_sort.iloc[0]["Peralatan Rumah"] if len(df_aktif_sort) > 0 else "-"
            porsi_terbesar = df_aktif_sort.iloc[0]["Porsi Beban (%)"] if len(df_aktif_sort) > 0 else 0.0
            solusi_terbesar = DETAIL_DIAGNOSIS.get(alat_terbesar, DEFAULT_DIAG_KUSTOM)["solusi"]

            render_html(f"""
            <div class="explain-box" style="border-left-color: #10b981;">
                <div class="explain-title" style="color: #34d399;">✅ Pemakaian Sudah Wajar — Tips Supaya Lebih Hemat Lagi</div>
                <div class="explain-item">• Tidak ditemukan peralatan yang menyala secara berlebihan atau di luar kewajaran harian.</div>
                <div class="explain-item">• Peralatan yang paling banyak menyerap biaya di rumah Anda saat ini adalah <b>{alat_terbesar}</b> (sebesar <b>{porsi_terbesar:.1f}%</b> dari total tagihan).</div>
                <div class="explain-item">• <b>Saran Penghematan:</b> {solusi_terbesar}</div>
            </div>
            """)

    with tab_grafik:
        fig1, ax1 = plt.subplots(figsize=(10.5, 3.6))
        fig1.patch.set_facecolor("#0f172a")
        ax1.set_facecolor("#131c31")

        df_aktif = df_rincian[df_rincian["Energi Harian (kWh)"] > 0]

        if len(df_aktif) > 0:
            nama_alat = [f"{r['Peralatan Rumah']} ({r['Jumlah']}x)" for _, r in df_aktif.iterrows()]
            kwh_aktual = df_aktif["Energi Harian (kWh)"].to_numpy()

            kwh_normal_list = []
            for _, r in df_aktif.iterrows():
                nama_row = r["Peralatan Rumah"]
                u_row = r["Jumlah"]
                if nama_row == "Magicom (Penanak Nasi)":
                    norm_val = u_row * ((JAM_PER_MASAK * WATT_MAGICOM_COOK) + (14.0 * WATT_MAGICOM_WARM)) / 1000.0
                elif nama_row == "Mesin Cuci":
                    norm_val = u_row * (0.5 * WATT_MESIN_CUCI) / 1000.0
                elif nama_row == "Colokan Kecil 24 Jam (Wi-Fi, Charger, STB)":
                    norm_val = (25.0 * 24.0) / 1000.0
                else:
                    norm_val = (PERANGKAT_UMUM[nama_row]["normal_jam"] * PERANGKAT_UMUM[nama_row]["watt"] * u_row) / 1000.0
                kwh_normal_list.append(norm_val)

            kwh_normal = np.array(kwh_normal_list)
            x = np.arange(len(nama_alat))
            lebar = 0.36

            ax1.bar(x - lebar / 2, kwh_normal, width=lebar, label="Batas Wajar Harian (kWh)", color="#475569", edgecolor="#64748b")
            warna_bar = ["#f43f5e" if akt > norm * 1.35 else "#10b981" for akt, norm in zip(kwh_aktual, kwh_normal)]
            bars_akt = ax1.bar(x + lebar / 2, kwh_aktual, width=lebar, label="Pemakaian Rumah Anda (kWh)", color=warna_bar)

            for bar in bars_akt:
                tinggi = bar.get_height()
                ax1.annotate(
                    f"{tinggi:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, tinggi),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#f8fafc"
                )

            ax1.set_title("1. Perbandingan Pemakaian Listrik Harian per Alat (kWh)", fontsize=11.5, fontweight="bold", color="#f8fafc", pad=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(nama_alat, rotation=15, ha="right", fontsize=9, color="#cbd5e1")
            ax1.set_ylabel("Energi (kWh / hari)", fontsize=9, color="#94a3b8")
            ax1.tick_params(axis="y", colors="#94a3b8", labelsize=8.5)
            ax1.set_ylim(0, max(max(kwh_aktual), max(kwh_normal)) * 1.24 + 0.2)
            leg1 = ax1.legend(fontsize=8.5, loc="upper right", facecolor="#0f172a", edgecolor="#334155")
            for text in leg1.get_texts():
                text.set_color("#e2e8f0")
            ax1.grid(axis="y", color="#334155", alpha=0.5, linestyle="--")
        else:
            ax1.text(0.5, 0.5, "Belum ada peralatan yang aktif", ha="center", va="center", color="#94a3b8")

        for spine in ["top", "right"]:
            ax1.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax1.spines[spine].set_color("#334155")

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        if len(df_aktif) > 0:
            baris_max_kwh = df_aktif.sort_values(by="Energi Harian (kWh)", ascending=False).iloc[0]
            nama_max_g1 = baris_max_kwh["Peralatan Rumah"]
            nilai_max_g1 = baris_max_kwh["Energi Harian (kWh)"]
            alat_merah = [
                r["Peralatan Rumah"]
                for (_, r), akt, norm in zip(df_aktif.iterrows(), kwh_aktual, kwh_normal)
                if akt > norm * 1.35
            ]
            ket_merah = (
                f"Terdapat batang berwarna <b>merah</b> pada <b>{', '.join(alat_merah)}</b> karena lama nyalanya melewati batas kewajaran harian."
                if len(alat_merah) > 0 else
                "Semua batang pemakaian rumah Anda berwarna <b>hijau</b>, artinya tidak ada peralatan yang dipakai secara berlebihan."
            )

            render_html(f"""
            <div class="explain-box">
                <div class="explain-title">📖 Penjelasan Grafik 1 (Pemakaian Listrik Tiap Alat)</div>
                <div class="explain-item">• <b>Arti Warna Batang:</b> Batang <b>abu-abu</b> adalah patokan pemakaian normal rumah tangga, batang <b>hijau</b> berarti pemakaian Anda aman, sedangkan batang <b>merah</b> menandakan alat tersebut dipakai terlalu lama (boros).</div>
                <div class="explain-item">• <b>Kesimpulan Rumah Anda:</b> Pemakaian energi terbesar berasal dari <b>{nama_max_g1}</b> sebesar <b>{nilai_max_g1:.2f} kWh/hari</b>. {ket_merah}</div>
            </div>
            """)

        fig2, ax2 = plt.subplots(figsize=(10.5, 3.6))
        fig2.patch.set_facecolor("#0f172a")
        ax2.set_facecolor("#131c31")

        ax2.scatter(hist_kwh, hist_tagihan / 1000.0, color="#38bdf8", alpha=0.65, s=38, label="Data Pemakaian Harian Normal (120 Hari)")

        x_max_plot = max(hist_kwh.max(), total_kwh_hari, 1.0) * 1.06
        x_line = np.linspace(0, x_max_plot, 100)
        y_line = model_regresi.predict(x_line.reshape(-1, 1)) / 1000.0
        ax2.plot(x_line, y_line, color="#fbbf24", linewidth=2.4, label="Garis Prediksi Kenaikan Biaya")

        ax2.scatter(
            [total_kwh_hari],
            [prediksi_tagihan_murni / 1000.0],
            color="#f43f5e" if is_boros else "#10b981",
            s=220,
            marker="*",
            edgecolors="#ffffff",
            linewidths=1.0,
            zorder=5,
            label=f"Posisi Rumah Anda ({total_kwh_hari:.2f} kWh → Rp {prediksi_tagihan_murni:,.0f} murni)"
        )

        ax2.set_title(f"2. Grafik Kenaikan Biaya Listrik Bulanan — {pilihan_golongan.split('(')[0].strip()}", fontsize=11.5, fontweight="bold", color="#f8fafc", pad=12)
        ax2.set_xlabel("Total Pemakaian Listrik Harian (kWh / hari)", fontsize=9, color="#94a3b8")
        ax2.set_ylabel("Biaya Listrik Murni (Ribu Rp)", fontsize=9, color="#94a3b8")
        ax2.tick_params(colors="#94a3b8", labelsize=8.5)
        leg2 = ax2.legend(fontsize=8.5, loc="upper left", facecolor="#0f172a", edgecolor="#334155")
        for text in leg2.get_texts():
            text.set_color("#e2e8f0")
        ax2.grid(True, color="#334155", alpha=0.5, linestyle="--")

        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax2.spines[spine].set_color("#334155")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        render_html(f"""
        <div class="explain-box">
            <div class="explain-title">📖 Penjelasan Grafik 2 (Hubungan Pemakaian kWh dengan Tagihan Rupiah)</div>
            <div class="explain-item">• <b>Arti Simbol Grafik:</b> Semakin ke kanan posisi <b>ikon bintang</b>, semakin besar konsumsi kWh harian Anda dan semakin tinggi pula tagihan rupiahnya mengikuti <b>garis miring kuning</b>.</div>
            <div class="explain-item">• <b>Mengapa Hasilnya Rp {estimasi_belanja_token_bulan:,.0f}/bulan?</b> Total pemakaian rumah Anda sebesar <b>{total_kwh_hari:.2f} kWh/hari</b> (atau <b>{total_kwh_bulan:.1f} kWh/bulan</b>) dikalikan tarif PLN <b>Rp {TARIF_PLN:,.2f}/kWh</b> menghasilkan biaya listrik murni <b>Rp {prediksi_tagihan_murni:,.0f}</b>, dan jika ditambah pajak PPJ ({persen_ppj}%) serta admin token menjadi <b>±Rp {estimasi_belanja_token_bulan:,.0f}/bulan</b>.</div>
        </div>
        """)

    with tab_tabel:
        st.markdown("#### 📋 Rincian Perhitungan Daya, Biaya & Porsi Beban Tiap Alat")
        st.dataframe(
            df_rincian,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Biaya Listrik/Bulan (Rp)": st.column_config.NumberColumn(format="Rp %d"),
                "Porsi Beban (%)": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
            }
        )

        df_export = df_rincian.copy()
        baris_total = pd.DataFrame([{
            "Peralatan Rumah": f"TOTAL KESELURUHAN ({pilihan_golongan.split('(')[0].strip()})",
            "Jumlah": int(df_rincian["Jumlah"].sum()),
            "Daya (Watt)": "-",
            "Beban Puncak (W)": int(total_watt_terpasang),
            "Lama Pakai": "30 Hari / Bulan",
            "Energi Harian (kWh)": round(total_kwh_hari, 2),
            "Biaya Listrik/Bulan (Rp)": round(total_biaya_murni, 0),
            "Porsi Beban (%)": 100.0 if total_biaya_murni > 0 else 0.0
        }])
        df_export = pd.concat([df_export, baris_total], ignore_index=True)
        csv_bytes = ("\ufeff" + df_export.to_csv(index=False)).encode("utf-8")

        st.download_button(
            label="📥 Simpan Laporan Perhitungan Listrik (Excel / CSV)",
            data=csv_bytes,
            file_name="Laporan_Listrik_Rumah_FikyArdiyan.csv",
            mime="text/csv",
            help="Unduh tabel rincian ini agar bisa dibuka dan disimpan di Microsoft Excel."
        )

        df_top2 = df_rincian[df_rincian["Energi Harian (kWh)"] > 0].sort_values(by="Porsi Beban (%)", ascending=False).head(2)
        if len(df_top2) >= 2:
            p1 = df_top2.iloc[0]
            p2 = df_top2.iloc[1]
            ringkasan_porsi = (
                f"Dua peralatan yang paling besar menyedot biaya listrik di rumah Anda adalah <b>{p1['Peralatan Rumah']}</b> "
                f"(<b>{p1['Porsi Beban (%)']:.1f}%</b> atau <b>Rp {p1['Biaya Listrik/Bulan (Rp)']:,.0f}/bulan</b>) "
                f"dan <b>{p2['Peralatan Rumah']}</b> (<b>{p2['Porsi Beban (%)']:.1f}%</b> atau <b>Rp {p2['Biaya Listrik/Bulan (Rp)']:,.0f}/bulan</b>)."
            )
        elif len(df_top2) == 1:
            p1 = df_top2.iloc[0]
            ringkasan_porsi = (
                f"Satu-satunya penyumbang beban listrik aktif saat ini adalah <b>{p1['Peralatan Rumah']}</b> "
                f"(<b>{p1['Porsi Beban (%)']:.1f}%</b> atau <b>Rp {p1['Biaya Listrik/Bulan (Rp)']:,.0f}/bulan</b>)."
            )
        else:
            ringkasan_porsi = "Belum ada peralatan listrik yang diaktifkan."

        render_html(f"""
        <div class="explain-box">
            <div class="explain-title">📖 Cara Membaca Tabel di Atas</div>
            <div class="explain-item">• <b>Mengapa alat berdaya kecil bisa mahal?</b> Tagihan bulanan tidak hanya bergantung pada besarnya Watt, tetapi juga <b>berapa jam alat itu menyala</b>. Alat yang menyala terus-menerus setiap hari tetap bisa menyumbang biaya besar meskipun Watt-nya kecil.</div>
            <div class="explain-item">• <b>Ringkasan Rumah Anda:</b> {ringkasan_porsi}</div>
        </div>
        """)

    with st.expander("🔬 Penjelasan Teknis, Rumus Fisika & Model Kecerdasan Buatan (Untuk Presentasi / Akademik)"):
        st.markdown("""
        **1. Rumus Perhitungan Energi Listrik (kWh) dan Biaya PLN**
        """)
        st.latex(r"E_{\text{harian}} = \frac{P \times t}{1.000} \quad (\text{kWh/hari}), \qquad \text{Biaya}_{\text{murni}} = \left(\sum E_{\text{harian}}\right) \times 30 \times T_{\text{PLN}}")

        st.markdown("""
        **2. Rumus Ketahanan Token Prabayar (Dengan Potongan Admin & Pajak PPJ Daerah)**
        """)
        st.latex(r"E_{\text{token}} = \frac{(N_{\text{token}} - \text{Admin}) \times \left(1 - \frac{p_{\text{PPJ}}}{100}\right)}{T_{\text{PLN}}}, \qquad \text{Ketahanan (Hari)} = \frac{E_{\text{token}}}{\sum E_{\text{harian}}}")

        st.markdown("""
        **3. Algoritma Machine Learning yang Digunakan**
        * **Isolation Forest:** Menganalisis kombinasi durasi seluruh peralatan rumah tangga terhadap 120 data historis pola harian untuk mendeteksi pemakaian yang menyimpang dari kewajaran.
        * **Z-Score Per Perangkat:** Mengukur simpangan durasi masing-masing alat ($Z = \\frac{t - \\mu}{\\sigma}$) untuk menentukan peralatan spesifik penyebab pemborosan.
        * **Linear Regression:** Memproyeksikan garis tren biaya bulanan ($\hat{y} = \\beta_0 + \\beta_1 x$) berdasarkan total konsumsi kWh harian.
        """)


# ==========================================
# FOOTER HAK CIPTA, LOGO UNIVERSITAS SEMARANG (USM) & KONTAK WHATSAPP
# ==========================================
logo_usm_src = get_usm_logo_data_uri()

render_html(f"""
<div class="footer-card">
    <div class="footer-brand">
        <img src="{logo_usm_src}" alt="Logo Universitas Semarang (USM)" style="width: 82px; height: auto; border-radius: 12px; border: 1.5px solid #facc15; background-color: #000000; padding: 3px; flex-shrink: 0;">
        <div>
            <div style="font-size: 15px; font-weight: 800; color: #facc15;">
                Hak Cipta © 2026 Fiky Ardiyan
            </div>
            <div style="font-size: 13.5px; color: #f8fafc; margin-top: 2px;">
                Program Studi S1 Teknik Elektro — <b>Universitas Semarang (USM)</b>
            </div>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                Smart Home AI: Asisten Pintar Pemantauan Energi &amp; Deteksi Pemborosan Listrik Rumah Tangga
            </div>
        </div>
    </div>
    <div style="flex: 0 1 auto;">
        <div style="font-size: 12.5px; color: #cbd5e1; margin-bottom: 4px;">
            Bila ada kritik dan saran, silakan hubungi saya:
        </div>
        <a class="wa-btn" href="https://wa.me/62895637350002" target="_blank">
            💬 WhatsApp: 0895637350002
        </a>
    </div>
</div>
""")