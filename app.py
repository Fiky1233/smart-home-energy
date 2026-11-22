import base64
import os
import time
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# ==========================================
# KONFIGURASI HALAMAN
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
    """Memaksa layar menggulir ke paling atas pada wadah scroll Streamlit (HP & Laptop)."""
    uid = int(time.time() * 1000)
    components.html(
        f"""
        <script>
            // Trigger ID: {uid}
            function paksaScrollAtas() {{
                try {{
                    window.parent.scrollTo({{top: 0, left: 0, behavior: 'instant'}});
                    const doc = window.parent.document;
                    if (doc.documentElement) doc.documentElement.scrollTop = 0;
                    if (doc.body) doc.body.scrollTop = 0;

                    const wadahScroll = [
                        'section.main',
                        'div[data-testid="stAppViewContainer"]',
                        'div[data-testid="stMain"]',
                        'div[data-testid="stMainBlockContainer"]',
                        '.main'
                    ];
                    wadahScroll.forEach(sel => {{
                        const el = doc.querySelector(sel);
                        if (el) {{
                            el.scrollTop = 0;
                            el.scrollTo({{top: 0, left: 0, behavior: 'instant'}});
                        }}
                    }});

                    const jangkarAtas = doc.getElementById('paling-atas-halaman');
                    if (jangkarAtas) {{
                        jangkarAtas.scrollIntoView({{behavior: 'instant', block: 'start'}});
                    }}
                }} catch (e) {{}}
            }}
            paksaScrollAtas();
            setTimeout(paksaScrollAtas, 60);
            setTimeout(paksaScrollAtas, 180);
            setTimeout(paksaScrollAtas, 400);
        </script>
        """,
        height=0,
        width=0,
    )


def get_usm_logo_data_uri() -> str:
    """Membaca file logo USM dari folder proyek atau memakai SVG cadangan."""
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
# KONFIGURASI TEMA DINAMIS (4 PILIHAN SUASANA)
# ==========================================
TEMAS = {
    "⚡ Biru Elektrik": {
        "is_light": False,
        "bg_app": "radial-gradient(circle at 12% 10%, rgba(14, 165, 233, 0.16) 0%, transparent 40%), radial-gradient(circle at 88% 18%, rgba(99, 102, 241, 0.15) 0%, transparent 42%), linear-gradient(180deg, #070b14 0%, #0b1120 55%, #070b14 100%)",
        "header_bg": "rgba(7, 11, 20, 0.88)",
        "card_bg": "rgba(15, 23, 42, 0.78)",
        "card_solid": "#0f172a",
        "input_bg": "#1e293b",
        "text_main": "#f8fafc",
        "text_sub": "#cbd5e1",
        "text_muted": "#94a3b8",
        "accent": "#38bdf8",
        "accent_2": "#818cf8",
        "btn_grad": "linear-gradient(135deg, #0284c7, #2563eb)",
        "border": "rgba(56, 189, 248, 0.35)",
        "border_soft": "rgba(148, 163, 184, 0.22)",
        "chart_fig": "#0f172a",
        "chart_ax": "#131c31",
        "chart_text": "#e2e8f0",
        "chart_grid": "#334155",
    },
    "🔥 Energi Oranye": {
        "is_light": False,
        "bg_app": "radial-gradient(circle at 15% 10%, rgba(249, 115, 22, 0.18) 0%, transparent 42%), radial-gradient(circle at 85% 20%, rgba(245, 158, 11, 0.15) 0%, transparent 42%), linear-gradient(180deg, #0c0a09 0%, #17120e 55%, #0c0a09 100%)",
        "header_bg": "rgba(12, 10, 9, 0.88)",
        "card_bg": "rgba(28, 22, 18, 0.82)",
        "card_solid": "#1c1917",
        "input_bg": "#292524",
        "text_main": "#fafaf9",
        "text_sub": "#e7e5e4",
        "text_muted": "#a8a29e",
        "accent": "#fb923c",
        "accent_2": "#fbbf24",
        "btn_grad": "linear-gradient(135deg, #ea580c, #d97706)",
        "border": "rgba(251, 146, 60, 0.42)",
        "border_soft": "rgba(168, 162, 158, 0.22)",
        "chart_fig": "#1c1917",
        "chart_ax": "#26211d",
        "chart_text": "#f5f5f4",
        "chart_grid": "#44403c",
    },
    "🌿 Hijau Eco": {
        "is_light": False,
        "bg_app": "radial-gradient(circle at 12% 10%, rgba(16, 185, 129, 0.17) 0%, transparent 40%), radial-gradient(circle at 88% 18%, rgba(20, 184, 166, 0.15) 0%, transparent 42%), linear-gradient(180deg, #04100c 0%, #081c16 55%, #04100c 100%)",
        "header_bg": "rgba(4, 16, 12, 0.88)",
        "card_bg": "rgba(10, 31, 24, 0.80)",
        "card_solid": "#0a221b",
        "input_bg": "#13352b",
        "text_main": "#f0fdf4",
        "text_sub": "#d1fae5",
        "text_muted": "#86efac",
        "accent": "#34d399",
        "accent_2": "#2dd4bf",
        "btn_grad": "linear-gradient(135deg, #059669, #0d9488)",
        "border": "rgba(52, 211, 153, 0.38)",
        "border_soft": "rgba(134, 239, 172, 0.2)",
        "chart_fig": "#0a221b",
        "chart_ax": "#0f2e25",
        "chart_text": "#ecfdf5",
        "chart_grid": "#1f4d3f",
    },
    "☀️ Terang Bersih": {
        "is_light": True,
        "bg_app": "linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%)",
        "header_bg": "rgba(255, 255, 255, 0.92)",
        "card_bg": "rgba(255, 255, 255, 0.96)",
        "card_solid": "#ffffff",
        "input_bg": "#ffffff",
        "text_main": "#0f172a",
        "text_sub": "#1e293b",
        "text_muted": "#475569",
        "accent": "#0284c7",
        "accent_2": "#2563eb",
        "btn_grad": "linear-gradient(135deg, #0284c7, #1d4ed8)",
        "border": "rgba(2, 132, 199, 0.38)",
        "border_soft": "rgba(15, 23, 42, 0.14)",
        "chart_fig": "#ffffff",
        "chart_ax": "#f8fafc",
        "chart_text": "#0f172a",
        "chart_grid": "#cbd5e1",
    },
}

# ==========================================
# KONSTANTA & OPSI PERALATAN
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

DAFTAR_CHIP_ALAT = [
    "❄️ Kulkas",
    "🌬️ AC",
    "🍚 Magicom",
    "🧺 Mesin Cuci",
    "💡 Lampu LED",
    "🌀 Kipas Angin",
    "📺 TV LED",
    "💧 Pompa Air (Sanyo)",
    "🚿 Water Heater",
    "🔌 Alat Tambahan Lainnya",
]


# ==========================================
# INISIALISASI, PRESET & CALLBACK STATE
# ==========================================
def get_default_inputs() -> dict:
    data = {
        "pilihan_golongan": PLACEHOLDER_PLN,
        "target_anggaran": 0,
        "pilihan_standby": list(OPSI_STANDBY_AWAM.keys())[0],
        "alat_terpilih": [],
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
    for idx_init in range(len(PERANGKAT_LAIN)):
        data[f"unit_{idx_init}"] = 0
        data[f"jam_{idx_init}"] = 0.0
    return data


def inisialisasi_state():
    if "sudah_hitung" not in st.session_state:
        st.session_state["sudah_hitung"] = False
    if "perlu_scroll_atas" not in st.session_state:
        st.session_state["perlu_scroll_atas"] = False
    if "tema_aktif" not in st.session_state:
        st.session_state["tema_aktif"] = "⚡ Biru Elektrik"
    if "mode_tampilan" not in st.session_state:
        st.session_state["mode_tampilan"] = "👤 Mode Warga Awam"
    if "saved_inputs" not in st.session_state:
        st.session_state["saved_inputs"] = get_default_inputs()

    for k, v in st.session_state["saved_inputs"].items():
        if k not in st.session_state:
            st.session_state[k] = v


def simpan_input_ke_memori():
    defaults = get_default_inputs()
    saved = {}
    for k, default_val in defaults.items():
        saved[k] = st.session_state.get(k, default_val)
    st.session_state["saved_inputs"] = saved


def pulihkan_input_dari_memori():
    for k, v in st.session_state.get("saved_inputs", get_default_inputs()).items():
        st.session_state[k] = v


def cb_reset_semua():
    defaults = get_default_inputs()
    st.session_state["saved_inputs"] = defaults.copy()
    st.session_state["sudah_hitung"] = False
    st.session_state["perlu_scroll_atas"] = True
    for k, v in defaults.items():
        st.session_state[k] = v


def cb_edit_input():
    pulihkan_input_dari_memori()
    st.session_state["sudah_hitung"] = False
    st.session_state["perlu_scroll_atas"] = True


def cb_pilih_semua_alat():
    st.session_state["alat_terpilih"] = DAFTAR_CHIP_ALAT.copy()


def cb_set_kebiasaan(updates: dict):
    for k, v in updates.items():
        st.session_state[k] = v


def cb_terapkan_preset(jenis: str):
    data = get_default_inputs()
    if jenis == "kost":
        data["pilihan_golongan"] = "900 VA (Non-Subsidi - Rp 1.352/kWh)"
        data["target_anggaran"] = 180000
        data["pilihan_standby"] = "Sedikit: Hanya Charger HP & Lampu Tidur (~10 Watt)"
        data["alat_terpilih"] = ["🍚 Magicom", "💡 Lampu LED", "🌀 Kipas Angin", "🔌 Alat Tambahan Lainnya"]
        data["unit_magicom"] = 1
        data["masak_magicom"] = 1
        data["warm_magicom"] = 8.0
        data["unit_0"] = 2
        data["jam_0"] = 7.0
        data["unit_1"] = 1
        data["jam_1"] = 8.0
        data["nama_c1"] = "Laptop Kerja / Kuliah"
        data["watt_c1"] = 65
        data["unit_c1"] = 1
        data["jam_c1"] = 6.0
    elif jenis == "hemat":
        data["pilihan_golongan"] = "1300 VA (Rumah Tangga - Rp 1.444,70/kWh)"
        data["target_anggaran"] = 450000
        data["pilihan_standby"] = "Standar Rumah: Wi-Fi, Charger HP & STB TV (~25 Watt)"
        data["alat_terpilih"] = ["❄️ Kulkas", "🌬️ AC", "🍚 Magicom", "🧺 Mesin Cuci", "💡 Lampu LED", "📺 TV LED", "💧 Pompa Air (Sanyo)"]
        data["pilihan_kulkas"] = "Kulkas 1 Pintu (Rata-rata 110 Watt)"
        data["unit_kulkas"] = 1
        data["jam_kulkas"] = 24.0
        data["pilihan_ac"] = "AC 1/2 PK (400 Watt)"
        data["unit_ac"] = 1
        data["jam_ac"] = 7.0
        data["unit_magicom"] = 1
        data["masak_magicom"] = 1
        data["warm_magicom"] = 12.0
        data["unit_cuci"] = 1
        data["jam_cuci_minggu"] = 3.0
        data["unit_0"] = 6
        data["jam_0"] = 9.0
        data["unit_2"] = 1
        data["jam_2"] = 4.0
        data["unit_3"] = 1
        data["jam_3"] = 1.5
    elif jenis == "boros":
        data["pilihan_golongan"] = "1300 VA (Rumah Tangga - Rp 1.444,70/kWh)"
        data["target_anggaran"] = 500000
        data["pilihan_standby"] = "Banyak: Wi-Fi, STB TV, CCTV & Banyak Charger (~45 Watt)"
        data["alat_terpilih"] = ["❄️ Kulkas", "🌬️ AC", "🍚 Magicom", "🧺 Mesin Cuci", "💡 Lampu LED", "🌀 Kipas Angin", "📺 TV LED", "💧 Pompa Air (Sanyo)", "🚿 Water Heater"]
        data["pilihan_kulkas"] = "Kulkas 2 Pintu (Rata-rata 165 Watt)"
        data["unit_kulkas"] = 1
        data["jam_kulkas"] = 24.0
        data["pilihan_ac"] = "AC 1 PK (850 Watt)"
        data["unit_ac"] = 1
        data["jam_ac"] = 16.0
        data["unit_magicom"] = 1
        data["masak_magicom"] = 3
        data["warm_magicom"] = 20.0
        data["unit_cuci"] = 1
        data["jam_cuci_minggu"] = 9.0
        data["unit_0"] = 10
        data["jam_0"] = 18.0
        data["unit_1"] = 2
        data["jam_1"] = 12.0
        data["unit_2"] = 1
        data["jam_2"] = 10.0
        data["unit_3"] = 1
        data["jam_3"] = 3.5
        data["unit_4"] = 1
        data["jam_4"] = 2.5

    for k, v in data.items():
        st.session_state[k] = v
    st.session_state["saved_inputs"] = data.copy()


inisialisasi_state()

T = TEMAS[st.session_state["tema_aktif"]]

# ==========================================
# GLOBAL CSS DINAMIS SESUAI TEMA AKTIF
# ==========================================
render_html(f"""
<div id="paling-atas-halaman" style="position:relative; top:-85px; height:1px;"></div>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {{
        color-scheme: {'light' if T['is_light'] else 'dark'};
    }}

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif !important;
    }}

    .stApp {{
        background: {T['bg_app']};
        color: {T['text_main']};
    }}

    section[data-testid="stSidebar"],
    button[data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    header[data-testid="stHeader"] {{
        background-color: {T['header_bg']} !important;
        backdrop-filter: blur(8px);
    }}
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg {{
        color: {T['text_main']} !important;
        fill: {T['text_main']} !important;
    }}

    .block-container {{
        padding-top: 4.3rem !important;
        padding-bottom: 5.2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1220px;
    }}

    /* Baris Tombol Atas Kiri (Tetap Sejajar di HP & Laptop) */
    div[data-testid="stVerticalBlock"]:has(> div.element-container .top-nav-anchor) > div[data-testid="stHorizontalBlock"] {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        justify-content: flex-start !important;
        align-items: center !important;
        gap: 8px !important;
        margin-bottom: 6px !important;
    }}
    div[data-testid="stVerticalBlock"]:has(> div.element-container .top-nav-anchor) > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
    div[data-testid="stVerticalBlock"]:has(> div.element-container .top-nav-anchor) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
        width: auto !important;
        flex: 0 0 auto !important;
        min-width: unset !important;
    }}
    div[data-testid="stVerticalBlock"]:has(> div.element-container .top-nav-anchor) > div[data-testid="stHorizontalBlock"] > div:first-child button {{
        background: {T['btn_grad']} !important;
        color: #ffffff !important;
        border: 1px solid {T['accent']} !important;
        font-weight: 700 !important;
        padding: 0.42rem 0.95rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.28) !important;
    }}

    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {T['text_main']} !important;
    }}

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    .stApp label {{
        color: {T['text_main']} !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {{
        color: {T['text_sub']} !important;
    }}
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b {{
        color: {T['text_main']} !important;
    }}

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {{
        color: {T['text_muted']} !important;
    }}

    [data-baseweb="select"] > div,
    [data-baseweb="input"],
    [data-baseweb="base-input"] {{
        background-color: {T['input_bg']} !important;
        border-color: {T['border']} !important;
        color: {T['text_main']} !important;
    }}
    [data-baseweb="select"] span,
    [data-baseweb="select"] div,
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input {{
        color: {T['text_main']} !important;
        -webkit-text-fill-color: {T['text_main']} !important;
    }}
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"] {{
        background-color: {T['card_solid']} !important;
        border: 1px solid {T['border']} !important;
    }}
    ul[role="listbox"] li,
    [role="option"] {{
        background-color: {T['card_solid']} !important;
        color: {T['text_main']} !important;
    }}
    ul[role="listbox"] li:hover,
    [role="option"]:hover {{
        background-color: {T['input_bg']} !important;
        color: {T['accent']} !important;
    }}

    span[data-baseweb="tag"] {{
        background: {T['btn_grad']} !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }}
    span[data-baseweb="tag"] span,
    span[data-baseweb="tag"] svg {{
        color: #ffffff !important;
        fill: #ffffff !important;
    }}

    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {{
        background-color: {T['input_bg']} !important;
        color: {T['text_main']} !important;
        border-color: {T['border_soft']} !important;
    }}

    button[kind="secondary"],
    [data-testid="stDownloadButton"] button {{
        background-color: {T['card_solid']} !important;
        color: {T['text_main']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 10px !important;
    }}
    button[kind="secondary"]:hover,
    [data-testid="stDownloadButton"] button:hover {{
        border-color: {T['accent']} !important;
        color: {T['accent']} !important;
    }}

    button[kind="primary"] {{
        background: {T['btn_grad']} !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 800 !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.2rem !important;
    }}

    [data-testid="stExpander"] {{
        background-color: {T['card_bg']} !important;
        border: 1px solid {T['border_soft']} !important;
        border-radius: 14px !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {{
        color: {T['text_main']} !important;
    }}

    .katex, .katex * {{
        color: {T['text_main']} !important;
    }}

    .main-app-header {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding-top: 2px;
        margin-bottom: 10px;
    }}
    .main-app-title {{
        font-size: 24px;
        font-weight: 800;
        color: {T['text_main']};
        letter-spacing: -0.4px;
        line-height: 1.3;
    }}
    .live-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: {T['card_bg']};
        border: 1px solid {T['border']};
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 700;
        color: {T['accent']};
    }}
    @keyframes dotPulse {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.45); opacity: 0.45; }}
    }}
    .live-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {T['accent']};
        box-shadow: 0 0 10px {T['accent']};
        animation: dotPulse 1.6s ease-in-out infinite;
    }}

    .input-intro-banner {{
        background: {T['card_bg']};
        border: 1px solid {T['border']};
        border-left: 5px solid {T['accent']};
        border-radius: 14px;
        padding: 13px 16px;
        color: {T['text_sub']};
        font-size: 13.5px;
        line-height: 1.5;
        margin-bottom: 12px;
    }}
    .form-section-header {{
        background: {T['card_bg']};
        border: 1px solid {T['border_soft']};
        border-left: 5px solid {T['accent']};
        border-radius: 12px;
        padding: 11px 16px;
        margin-top: 14px;
        margin-bottom: 10px;
    }}
    .form-section-title {{
        font-size: 15.5px;
        font-weight: 800;
        color: {T['accent']};
        margin-bottom: 2px;
    }}
    .form-section-sub {{
        font-size: 12.5px;
        color: {T['text_muted']};
    }}

    .device-box-label {{
        background: {T['card_bg']};
        border: 1px solid {T['border_soft']};
        border-radius: 12px;
        padding: 9px 14px;
        margin-top: 8px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 6px;
    }}
    .device-box-name {{
        font-size: 14.5px;
        font-weight: 800;
        color: {T['text_main']};
    }}
    .device-box-watt {{
        font-size: 12px;
        font-weight: 700;
        color: {T['accent']};
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid {T['border']};
        padding: 2px 9px;
        border-radius: 999px;
    }}

    .sticky-preview-bar {{
        position: fixed;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        width: calc(100% - 24px);
        max-width: 960px;
        z-index: 999;
        background: {T['card_solid']};
        border: 1.5px solid {T['accent']};
        border-radius: 14px;
        padding: 10px 16px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.42);
    }}

    .summary-tldr-box {{
        background: {T['card_bg']};
        border: 1.5px solid {T['accent']};
        border-left: 6px solid {T['accent']};
        border-radius: 16px;
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 14px;
    }}
    .summary-tldr-title {{
        font-size: 16px;
        font-weight: 800;
        color: {T['accent']};
        margin-bottom: 8px;
    }}
    .summary-tldr-item {{
        font-size: 13.8px;
        color: {T['text_main']};
        line-height: 1.6;
        margin-bottom: 5px;
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-top: 12px;
        margin-bottom: 16px;
    }}
    .kpi-card {{
        position: relative;
        overflow: hidden;
        background: {T['card_bg']};
        border: 1px solid {T['border_soft']};
        border-radius: 16px;
        padding: 15px 17px;
        color: {T['text_main']};
    }}
    .kpi-card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: {T['btn_grad']};
    }}
    .kpi-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }}
    .kpi-label {{
        font-size: 11px;
        color: {T['text_muted']};
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}
    .kpi-icon {{
        font-size: 15px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid {T['border_soft']};
        border-radius: 8px;
        padding: 2px 7px;
    }}
    .kpi-value {{
        font-size: 22px;
        font-weight: 800;
        color: {T['text_main']};
    }}
    .kpi-sub {{
        font-size: 12px;
        color: {T['text_muted']};
        margin-top: 3px;
    }}

    .glass-card {{
        background: {T['card_bg']};
        border: 1px solid {T['border_soft']};
        border-radius: 16px;
        padding: 18px 20px;
        color: {T['text_main']};
        margin-bottom: 14px;
    }}
    .feature-title {{
        font-size: 15.5px;
        font-weight: 800;
        color: {T['accent']};
        margin-bottom: 10px;
    }}

    @keyframes geserProgress {{
        0% {{ background-position: 0% 50%; }}
        100% {{ background-position: 100% 50%; }}
    }}
    .progress-track {{
        width: 100%;
        height: 14px;
        background-color: rgba(148, 163, 184, 0.22);
        border-radius: 999px;
        overflow: hidden;
        margin: 10px 0;
        border: 1px solid {T['border_soft']};
        padding: 2px;
    }}
    .progress-fill-anim {{
        height: 100%;
        border-radius: 999px;
        background-size: 200% 200%;
        animation: geserProgress 2.2s linear infinite;
        transition: width 0.55s ease;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {T['card_bg']};
        padding: 6px;
        border-radius: 14px;
        border: 1px solid {T['border_soft']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 9px 16px;
        font-weight: 700;
        font-size: 13px;
        color: {T['text_sub']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {T['btn_grad']} !important;
        color: #ffffff !important;
    }}

    .diag-card {{
        background: {'rgba(255, 241, 242, 0.96)' if T['is_light'] else 'linear-gradient(145deg, rgba(38, 14, 21, 0.9), rgba(24, 12, 18, 0.94))'};
        border: 1px solid rgba(244, 63, 94, 0.45);
        border-left: 5px solid #f43f5e;
        border-radius: 14px;
        padding: 15px 18px;
        color: {T['text_main']};
        margin-bottom: 12px;
    }}
    .diag-header {{
        font-size: 15px;
        font-weight: 800;
        color: {'#be123c' if T['is_light'] else '#fda4af'};
        margin-bottom: 7px;
    }}
    .diag-row {{
        font-size: 13.5px;
        color: {T['text_sub']};
        margin-bottom: 5px;
        line-height: 1.55;
    }}

    .explain-box {{
        background: {T['card_bg']};
        border: 1px solid {T['border_soft']};
        border-left: 4px solid {T['accent']};
        border-radius: 14px;
        padding: 15px 18px;
        color: {T['text_sub']};
        margin-top: 10px;
        margin-bottom: 14px;
    }}
    .explain-title {{
        font-size: 14.5px;
        font-weight: 800;
        color: {T['accent']};
        margin-bottom: 7px;
    }}
    .explain-item {{
        font-size: 13.5px;
        margin-bottom: 5px;
        line-height: 1.6;
        color: {T['text_sub']};
    }}

    .footer-card {{
        margin-top: 26px;
        padding: 20px 24px;
        background: linear-gradient(135deg, #0b132b 0%, #0f172a 60%, #172554 100%);
        border: 1px solid rgba(59, 130, 246, 0.45);
        border-radius: 18px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        color: #e2e8f0;
    }}
    .footer-brand {{
        display: flex;
        align-items: center;
        gap: 16px;
        flex: 1 1 300px;
    }}
    .wa-btn {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: #ffffff !important;
        text-decoration: none;
        font-weight: 700;
        font-size: 13px;
        padding: 9px 17px;
        border-radius: 999px;
        border: 1px solid #4ade80;
        margin-top: 6px;
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding-top: 4.5rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }}
        .main-app-title {{
            font-size: 19px;
        }}
        .kpi-grid {{
            grid-template-columns: repeat(2, 1fr);
            gap: 9px;
        }}
        .kpi-value {{
            font-size: 17.5px;
        }}
    }}
</style>
""")


# ==========================================
# 1. MODEL AI: ISOLATION FOREST, REGRESSION & LOGIKA FUZZY
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


def hitung_skor_fuzzy_efisiensi(rasio_kwh_normal: float, rasio_mcb: float, jumlah_alat_boros: int) -> tuple:
    """
    Sistem Inferensi Logika Fuzzy (Mamdani-Sugeno terpadu) untuk menghasilkan
    Skor Efisiensi Rumah (0-100) beserta lencana gamifikasi.
    """
    mu_hemat = float(np.clip((1.15 - rasio_kwh_normal) / 0.35, 0.0, 1.0))
    mu_sedang = float(np.clip(1.0 - abs(rasio_kwh_normal - 1.15) / 0.35, 0.0, 1.0))
    mu_boros = float(np.clip((rasio_kwh_normal - 1.15) / 0.55, 0.0, 1.0))

    penalti_mcb = 12.0 if rasio_mcb > 1.0 else (4.0 if rasio_mcb > 0.85 else 0.0)
    penalti_alat = min(20.0, jumlah_alat_boros * 6.5)

    bobot_total = max(mu_hemat + mu_sedang + mu_boros, 1e-6)
    skor_dasar = ((mu_hemat * 95.0) + (mu_sedang * 76.0) + (mu_boros * 42.0)) / bobot_total
    skor_akhir = int(np.clip(round(skor_dasar - penalti_mcb - penalti_alat), 15, 99))

    if skor_akhir >= 85:
        return skor_akhir, "🏆 Lencana: Rumah Sangat Hemat Energi", "#10b981"
    elif skor_akhir >= 70:
        return skor_akhir, "✅ Lencana: Rumah Wajar & Terkendali", "#38bdf8"
    elif skor_akhir >= 50:
        return skor_akhir, "⚠️ Lencana: Agak Boros — Perlu Pengaturan", "#f59e0b"
    else:
        return skor_akhir, "🚨 Lencana: Boros Kritis — Wajib Diet Watt", "#f43f5e"


# ==========================================
# 2. HERO ANIMASI DASHBOARD
# ==========================================
def render_animasi_dashboard(is_boros, skor_fuzzy, lencana_fuzzy, warna_lencana, total_kwh_hari, total_watt, batas_va, nama_golongan, tarif_pln):
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
        sub_status = "Ada peralatan yang menyala jauh lebih lama dari kebiasaan wajar. Lihat ringkasan & cara hematnya di bawah."
        label_meter = "BOROS"
    else:
        warna_utama = "#10b981"
        warna_aksen = "#38bdf8"
        glow_rgba = "rgba(16, 185, 129, 0.18)"
        badge_bg = "rgba(16, 185, 129, 0.16)"
        judul_status = "STATUS AMAN: PEMAKAIAN LISTRIK WAJAR & EFISIEN"
        sub_status = "Seluruh peralatan elektronik di rumah Anda bekerja dalam batas durasi harian yang wajar."
        label_meter = "EFISIEN"

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
            50% {{ box-shadow: 0 14px 36px rgba(56, 189, 248, 0.22); }}
        }}
        @keyframes kedipJendela {{
            0%, 100% {{ fill: {warna_aksen}; opacity: 0.9; }}
            50% {{ fill: #fef08a; opacity: 0.35; }}
        }}
        .hero-glass-banner {{
            background: linear-gradient(135deg, #0f172a 0%, #131c31 60%, #0f172a 100%);
            border: 2px solid {warna_utama};
            border-radius: 18px;
            padding: 18px 22px;
            color: #f8fafc;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
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
            .judul-hero {{ font-size: 16.5px !important; }}
            .panel-svg {{ width: 100% !important; justify-content: center !important; }}
        }}
    </style>
    <div class="hero-glass-banner">
        <div style="flex: 1 1 320px;">
            <div style="display: inline-flex; align-items: center; gap: 7px; background-color: {badge_bg}; color: {warna_utama}; border: 1px solid {warna_utama}; padding: 4px 12px; border-radius: 999px; font-size: 11px; font-weight: 800;">
                PLN {nama_golongan.split('(')[0].strip()} • TARIF Rp {tarif_pln:,.2f}/kWh
            </div>
            <div class="judul-hero" style="font-size: 20px; font-weight: 800; margin-top: 9px; color: {warna_utama}; line-height: 1.25;">
                {judul_status}
            </div>
            <div style="font-size: 13px; color: #cbd5e1; margin-top: 5px; line-height: 1.5;">
                {sub_status}
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 11px;">
                <span style="background: rgba(30, 41, 59, 0.9); border: 1px solid {warna_lencana}; padding: 4px 11px; border-radius: 8px; font-size: 12px; color: {warna_lencana}; font-weight: 700;">
                    Skor Efisiensi: {skor_fuzzy}/100 ({lencana_fuzzy})
                </span>
                <span style="background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(148,163,184,0.28); padding: 4px 11px; border-radius: 8px; font-size: 12px; color: #cbd5e1;">
                    Beban Meteran: <b style="color: {warna_aksen};">{persen_mcb}%</b> dari {batas_va} VA
                </span>
            </div>
        </div>
        <div class="panel-svg" style="flex: 0 1 auto; display: flex; flex-wrap: wrap; align-items: center; gap: 10px;">
            <svg width="245" height="114" viewBox="0 0 265 124">
                <rect x="3" y="4" width="259" height="116" rx="14" fill="#070b14" stroke="rgba(148,163,184,0.28)" stroke-width="1.5" />
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
            <svg width="104" height="114" viewBox="0 0 108 118">
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
# BILAH NAVIGASI ATAS KIRI (MUNCUL SAAT MELIHAT OUTPUT)
# ==========================================
if st.session_state["sudah_hitung"]:
    with st.container():
        render_html('<span class="top-nav-anchor"></span>')
        col_nav_edit, col_nav_reset, _ = st.columns([1.3, 1.0, 3.0])
        with col_nav_edit:
            st.button(
                "⬅️ Edit Input",
                key="btn_edit_atas_kiri",
                on_click=cb_edit_input,
                help="Kembali ke halaman pengisian untuk mengubah data input"
            )
        with col_nav_reset:
            st.button(
                "🔄 Reset",
                key="btn_reset_atas_kiri",
                on_click=cb_reset_semua,
                help="Kosongkan semua data dari awal"
            )


# ==========================================
# HEADER UTAMA & PEMILIH TEMA WARNA INSTAN
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

col_tema, col_mode = st.columns([1.4, 1.4])
with col_tema:
    st.selectbox(
        "🎨 Suasana Tema Tampilan",
        options=list(TEMAS.keys()),
        key="tema_aktif"
    )
with col_mode:
    if st.session_state["sudah_hitung"]:
        st.selectbox(
            "👁️ Sudut Pandang Penjelasan",
            options=["👤 Mode Warga Awam", "🎓 Mode Teknis / Dosen"],
            key="mode_tampilan"
        )


# ==========================================
# HALAMAN 1: FORMULIR PENGISIAN RINGKAS DI HP
# ==========================================
if not st.session_state["sudah_hitung"]:
    render_html("""
    <div class="input-intro-banner">
        📋 <b>Cara Pakai Cepat (3 Langkah):</b><br>
        1️⃣ Pilih <b>Golongan Daya PLN</b> rumah Anda (atau tekan <b>Tombol Contoh</b> di bawah untuk demo cepat).<br>
        2️⃣ Centang <b>Peralatan yang Ada di Rumah</b> agar layar tetap ringkas.<br>
        3️⃣ Isi jumlah unit &amp; jam nyala, lalu tekan tombol <b>Hitung &amp; Analisa Sekarang</b> di bawah.
    </div>
    """)

    cp1, cp2, cp3, cp4 = st.columns(4)
    with cp1:
        st.button("🏠 Contoh Kost 900VA", on_click=cb_terapkan_preset, args=("kost",), use_container_width=True)
    with cp2:
        st.button("👨‍👩‍👧 Contoh Hemat 1300VA", on_click=cb_terapkan_preset, args=("hemat",), use_container_width=True)
    with cp3:
        st.button("🚨 Contoh Boros 1300VA", on_click=cb_terapkan_preset, args=("boros",), use_container_width=True)
    with cp4:
        st.button("🔄 Kosongkan (0)", on_click=cb_reset_semua, use_container_width=True)

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

    render_html("""
    <div class="form-section-header">
        <div class="form-section-title">2. Pilih Peralatan Elektronik di Rumah Anda</div>
        <div class="form-section-sub">Ketuk kotak di bawah untuk memilih alat yang dimiliki agar formulir di HP tetap ringkas.</div>
    </div>
    """)

    col_ms, col_all = st.columns([3.2, 1])
    with col_ms:
        alat_terpilih = st.multiselect(
            "Daftar Peralatan yang Dimiliki",
            options=DAFTAR_CHIP_ALAT,
            key="alat_terpilih",
            placeholder="Ketuk di sini untuk memilih: Kulkas, AC, Magicom, Lampu LED, dll..."
        )
    with col_all:
        st.markdown("<div style='height: 27px;'></div>", unsafe_allow_html=True)
        st.button("✅ Pilih Semua", on_click=cb_pilih_semua_alat, use_container_width=True)

    if "❄️ Kulkas" in alat_terpilih:
        watt_kulkas_prev = OPSI_KULKAS[st.session_state["pilihan_kulkas"]]
        render_html(f"""
        <div class="device-box-label">
            <span class="device-box-name">❄️ Kulkas (Lemari Es)</span>
            <span class="device-box-watt">{watt_kulkas_prev} W/unit</span>
        </div>
        """)
        st.selectbox("Tipe Kulkas", options=list(OPSI_KULKAS.keys()), key="pilihan_kulkas", label_visibility="collapsed")
        ck1, ck2 = st.columns([1, 1.8])
        u_kulkas = ck1.number_input("Jumlah Kulkas", min_value=0, max_value=5, step=1, key="unit_kulkas")
        ck2.slider("Nyala Kulkas (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, key="jam_kulkas", disabled=(u_kulkas == 0))
        qb1, qb2 = st.columns(2)
        qb1.button("⚡ Pintas: 1 Unit Nyala Seharian (24j)", key="qb_kulkas_24", on_click=cb_set_kebiasaan, args=({"unit_kulkas": max(1, u_kulkas), "jam_kulkas": 24.0},), use_container_width=True)
        qb2.button("🔄 Nolkan Kulkas", key="qb_kulkas_0", on_click=cb_set_kebiasaan, args=({"unit_kulkas": 0, "jam_kulkas": 0.0},), use_container_width=True)

    if "🌬️ AC" in alat_terpilih:
        watt_ac_prev = OPSI_AC[st.session_state["pilihan_ac"]]
        render_html(f"""
        <div class="device-box-label">
            <span class="device-box-name">🌬️ Air Conditioner (AC)</span>
            <span class="device-box-watt">{watt_ac_prev} W/unit</span>
        </div>
        """)
        st.selectbox("Kapasitas AC", options=list(OPSI_AC.keys()), key="pilihan_ac", label_visibility="collapsed")
        ca1, ca2 = st.columns([1, 1.8])
        u_ac = ca1.number_input("Jumlah AC", min_value=0, max_value=5, step=1, key="unit_ac")
        ca2.slider("Nyala AC (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, key="jam_ac", disabled=(u_ac == 0))
        qa1, qa2 = st.columns(2)
        qa1.button("🌙 Pintas: Malam Tidur (8j)", key="qb_ac_8", on_click=cb_set_kebiasaan, args=({"unit_ac": max(1, u_ac), "jam_ac": 8.0},), use_container_width=True)
        qa2.button("☀️ Pintas: Siang + Malam (15j)", key="qb_ac_15", on_click=cb_set_kebiasaan, args=({"unit_ac": max(1, u_ac), "jam_ac": 15.0},), use_container_width=True)

    if "🍚 Magicom" in alat_terpilih:
        render_html(f"""
        <div class="device-box-label">
            <span class="device-box-name">🍚 Magicom (Penanak Nasi)</span>
            <span class="device-box-watt">Masak {WATT_MAGICOM_COOK}W / Hangat {WATT_MAGICOM_WARM}W</span>
        </div>
        """)
        cm1, cm2 = st.columns([1, 1.8])
        u_mag = cm1.number_input("Jumlah Magicom", min_value=0, max_value=5, step=1, key="unit_magicom")
        cm2.slider("Masak Nasi (Kali/Hari)", min_value=0, max_value=4, step=1, key="masak_magicom", disabled=(u_mag == 0))
        st.slider("Lama Mode Menghangatkan (Jam/Hari)", min_value=0.0, max_value=24.0, step=1.0, key="warm_magicom", disabled=(u_mag == 0))
        qm1, qm2 = st.columns(2)
        qm1.button("🍚 Pintas: Masak 1x + Hangat 12j", key="qb_mag_1", on_click=cb_set_kebiasaan, args=({"unit_magicom": max(1, u_mag), "masak_magicom": 1, "warm_magicom": 12.0},), use_container_width=True)
        qm2.button("🔥 Pintas: Masak 3x + Hangat 20j", key="qb_mag_3", on_click=cb_set_kebiasaan, args=({"unit_magicom": max(1, u_mag), "masak_magicom": 3, "warm_magicom": 20.0},), use_container_width=True)

    if "🧺 Mesin Cuci" in alat_terpilih:
        render_html(f"""
        <div class="device-box-label">
            <span class="device-box-name">🧺 Mesin Cuci</span>
            <span class="device-box-watt">{WATT_MESIN_CUCI} W/unit</span>
        </div>
        """)
        cc1, cc2 = st.columns([1, 1.8])
        u_cuci = cc1.number_input("Jumlah Mesin Cuci", min_value=0, max_value=3, step=1, key="unit_cuci")
        cc2.slider("Total Jam Cuci / Minggu", min_value=0.0, max_value=21.0, step=0.5, key="jam_cuci_minggu", disabled=(u_cuci == 0))
        qc1, qc2 = st.columns(2)
        qc1.button("🧺 Pintas: 2x Seminggu (3j)", key="qb_cuci_3", on_click=cb_set_kebiasaan, args=({"unit_cuci": max(1, u_cuci), "jam_cuci_minggu": 3.0},), use_container_width=True)
        qc2.button("💦 Pintas: Hampir Tiap Hari (9j)", key="qb_cuci_9", on_click=cb_set_kebiasaan, args=({"unit_cuci": max(1, u_cuci), "jam_cuci_minggu": 9.0},), use_container_width=True)

    map_chip_lain = [
        ("💡 Lampu LED", "Lampu LED (Per Titik)", 0, "💡", 5, 10.0, "Malam (10j)"),
        ("🌀 Kipas Angin", "Kipas Angin", 1, "🌀", 1, 8.0, "Tidur Malam (8j)"),
        ("📺 TV LED", "TV LED", 2, "📺", 1, 5.0, "Sore-Malam (5j)"),
        ("💧 Pompa Air (Sanyo)", "Sanyo (Pompa Air Otomatis)", 3, "💧", 1, 1.5, "Normal (1.5j)"),
        ("🚿 Water Heater", "Water Heater", 4, "🚿", 1, 1.0, "Mandi Pagi/Sore (1j)"),
    ]

    for chip_lbl, nama_asli, idx_lain, ikon_l, def_u, def_j, txt_pintas in map_chip_lain:
        if chip_lbl in alat_terpilih:
            spec = PERANGKAT_LAIN[nama_asli]
            sat_w = "W/lampu" if nama_asli == "Lampu LED (Per Titik)" else "W/unit"
            render_html(f"""
            <div class="device-box-label">
                <span class="device-box-name">{ikon_l} {nama_asli}</span>
                <span class="device-box-watt">{spec['watt']} {sat_w}</span>
            </div>
            """)
            cu, cj = st.columns([1, 1.8])
            lbl_u = "Jumlah Lampu" if nama_asli == "Lampu LED (Per Titik)" else "Jumlah Unit"
            u_val = cu.number_input(lbl_u, min_value=0, max_value=spec["max_unit"], step=1, key=f"unit_{idx_lain}")
            cj.slider("Lama Nyala (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, key=f"jam_{idx_lain}", disabled=(u_val == 0))
            qp1, qp2 = st.columns(2)
            qp1.button(
                f"⚡ Pintas: {txt_pintas}",
                key=f"qp_norm_{idx_lain}",
                on_click=cb_set_kebiasaan,
                args=({f"unit_{idx_lain}": max(def_u, u_val), f"jam_{idx_lain}": def_j},),
                use_container_width=True
            )
            qp2.button(
                "🔄 Nolkan",
                key=f"qp_zero_{idx_lain}",
                on_click=cb_set_kebiasaan,
                args=({f"unit_{idx_lain}": 0, f"jam_{idx_lain}": 0.0},),
                use_container_width=True
            )

    if "🔌 Alat Tambahan Lainnya" in alat_terpilih:
        render_html("""
        <div class="form-section-header">
            <div class="form-section-title">3. Alat Tambahan Lainnya (Setrika, PC, Dispenser, dll.)</div>
            <div class="form-section-sub">Isi nama, daya Watt, jumlah unit, dan lama nyala per hari.</div>
        </div>
        """)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.text_input("Nama Alat Tambahan 1", placeholder="Contoh: Setrika Listrik", key="nama_c1")
            cw1, cu1 = st.columns(2)
            w_c1 = cw1.number_input("Daya Alat 1 (Watt)", min_value=0, max_value=5000, step=5, key="watt_c1")
            u_c1 = cu1.number_input("Unit Alat 1", min_value=0, max_value=10, step=1, key="unit_c1")
            st.slider("Lama Nyala Alat 1 (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, disabled=(u_c1 == 0 or w_c1 == 0), key="jam_c1")

        with col_c2:
            st.text_input("Nama Alat Tambahan 2", placeholder="Contoh: PC / Komputer", key="nama_c2")
            cw2, cu2 = st.columns(2)
            w_c2 = cw2.number_input("Daya Alat 2 (Watt)", min_value=0, max_value=5000, step=5, key="watt_c2")
            u_c2 = cu2.number_input("Unit Alat 2", min_value=0, max_value=10, step=1, key="unit_c2")
            st.slider("Lama Nyala Alat 2 (Jam/Hari)", min_value=0.0, max_value=24.0, step=0.5, disabled=(u_c2 == 0 or w_c2 == 0), key="jam_c2")

    # Hitung estimasi sementara (Live Preview)
    tarif_prev = GOLONGAN_PLN[pilihan_golongan]["tarif"] if pilihan_golongan in GOLONGAN_PLN else 1444.70
    watt_prev = OPSI_STANDBY_AWAM[pilihan_standby]
    kwh_prev = (watt_prev * 24.0) / 1000.0

    if "❄️ Kulkas" in alat_terpilih and st.session_state["unit_kulkas"] > 0:
        w_k = OPSI_KULKAS[st.session_state["pilihan_kulkas"]] * st.session_state["unit_kulkas"]
        watt_prev += w_k
        kwh_prev += (w_k * st.session_state["jam_kulkas"]) / 1000.0
    if "🌬️ AC" in alat_terpilih and st.session_state["unit_ac"] > 0:
        w_a = OPSI_AC[st.session_state["pilihan_ac"]] * st.session_state["unit_ac"]
        watt_prev += w_a
        kwh_prev += (w_a * st.session_state["jam_ac"]) / 1000.0
    if "🍚 Magicom" in alat_terpilih and st.session_state["unit_magicom"] > 0:
        u_m = st.session_state["unit_magicom"]
        watt_prev += u_m * WATT_MAGICOM_COOK
        kwh_prev += u_m * ((st.session_state["masak_magicom"] * JAM_PER_MASAK * WATT_MAGICOM_COOK) + (st.session_state["warm_magicom"] * WATT_MAGICOM_WARM)) / 1000.0
    if "🧺 Mesin Cuci" in alat_terpilih and st.session_state["unit_cuci"] > 0:
        u_c = st.session_state["unit_cuci"]
        watt_prev += u_c * WATT_MESIN_CUCI
        kwh_prev += (u_c * WATT_MESIN_CUCI * (st.session_state["jam_cuci_minggu"] / 7.0)) / 1000.0

    for chip_lbl, nama_asli, idx_lain, _, _, _, _ in map_chip_lain:
        if chip_lbl in alat_terpilih and st.session_state[f"unit_{idx_lain}"] > 0:
            w_l = PERANGKAT_LAIN[nama_asli]["watt"] * st.session_state[f"unit_{idx_lain}"]
            watt_prev += w_l
            kwh_prev += (w_l * st.session_state[f"jam_{idx_lain}"]) / 1000.0

    if "🔌 Alat Tambahan Lainnya" in alat_terpilih:
        if st.session_state["unit_c1"] > 0 and st.session_state["watt_c1"] > 0:
            w_1 = st.session_state["unit_c1"] * st.session_state["watt_c1"]
            watt_prev += w_1
            kwh_prev += (w_1 * st.session_state["jam_c1"]) / 1000.0
        if st.session_state["unit_c2"] > 0 and st.session_state["watt_c2"] > 0:
            w_2 = st.session_state["unit_c2"] * st.session_state["watt_c2"]
            watt_prev += w_2
            kwh_prev += (w_2 * st.session_state["jam_c2"]) / 1000.0

    rp_prev = kwh_prev * 30.0 * tarif_prev

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    golongan_ok = (pilihan_golongan != PLACEHOLDER_PLN)
    if st.button("🚀 Hitung & Analisa Sekarang", type="primary", use_container_width=True):
        if not golongan_ok:
            st.error("⚠️ Silakan pilih **Golongan Daya Listrik (PLN)** terlebih dahulu pada **Bagian 1** di atas.")
        elif kwh_prev <= 0:
            st.error("⚠️ Data peralatan masih kosong. Centang minimal **1 peralatan elektronik** pada **Bagian 2** dan isi unit serta jam nyalanya.")
        else:
            simpan_input_ke_memori()
            st.session_state["sudah_hitung"] = True
            st.session_state["perlu_scroll_atas"] = True
            st.rerun()

    if kwh_prev > 0:
        render_html(f"""
        <div class="sticky-preview-bar">
            <div style="font-size: 12.5px; color: {T['text_sub']};">
                ⚡ <b>Estimasi Sementara:</b> <span style="color:{T['accent']}; font-weight:800;">{watt_prev:,} W</span> • <b>{kwh_prev:.2f} kWh/hari</b>
            </div>
            <div style="font-size: 13.5px; font-weight: 800; color: {T['accent']};">
                ±Rp {rp_prev:,.0f} / bulan
            </div>
        </div>
        """)

else:
    # ==========================================
    # HALAMAN 2: OUTPUT LENGKAP, MUDAH DIPAHAMI & INTERAKTIF
    # ==========================================
    data_in = st.session_state["saved_inputs"]
    mode_dosen = (st.session_state["mode_tampilan"] == "🎓 Mode Teknis / Dosen")
    alat_aktif_chip = data_in.get("alat_terpilih", [])

    pilihan_golongan = data_in["pilihan_golongan"]
    target_anggaran = data_in["target_anggaran"]
    pilihan_standby = data_in["pilihan_standby"]
    watt_standby = OPSI_STANDBY_AWAM[pilihan_standby]

    pilihan_kulkas = data_in["pilihan_kulkas"]
    watt_kulkas = OPSI_KULKAS[pilihan_kulkas]
    unit_kulkas = data_in["unit_kulkas"] if "❄️ Kulkas" in alat_aktif_chip else 0
    jam_kulkas = data_in["jam_kulkas"] if unit_kulkas > 0 else 0.0

    pilihan_ac = data_in["pilihan_ac"]
    watt_ac = OPSI_AC[pilihan_ac]
    unit_ac = data_in["unit_ac"] if "🌬️ AC" in alat_aktif_chip else 0
    jam_ac = data_in["jam_ac"] if unit_ac > 0 else 0.0

    unit_magicom = data_in["unit_magicom"] if "🍚 Magicom" in alat_aktif_chip else 0
    masak_kali = data_in["masak_magicom"] if unit_magicom > 0 else 0
    warm_jam = data_in["warm_magicom"] if unit_magicom > 0 else 0.0

    unit_cuci = data_in["unit_cuci"] if "🧺 Mesin Cuci" in alat_aktif_chip else 0
    jam_cuci_minggu = data_in["jam_cuci_minggu"] if unit_cuci > 0 else 0.0

    chip_lain_map = {
        "Lampu LED (Per Titik)": "💡 Lampu LED",
        "Kipas Angin": "🌀 Kipas Angin",
        "TV LED": "📺 TV LED",
        "Sanyo (Pompa Air Otomatis)": "💧 Pompa Air (Sanyo)",
        "Water Heater": "🚿 Water Heater",
    }

    input_unit_lain = {}
    input_jam_lain = {}
    for idx_read, nama_read in enumerate(PERANGKAT_LAIN.keys()):
        chip_cek = chip_lain_map[nama_read]
        u_val = data_in[f"unit_{idx_read}"] if chip_cek in alat_aktif_chip else 0
        j_val = data_in[f"jam_{idx_read}"] if u_val > 0 else 0.0
        input_unit_lain[nama_read] = u_val
        input_jam_lain[nama_read] = j_val

    aktif_custom = "🔌 Alat Tambahan Lainnya" in alat_aktif_chip
    nama_c1 = data_in["nama_c1"].strip() or "Alat Tambahan 1"
    watt_c1 = data_in["watt_c1"] if aktif_custom else 0
    unit_c1 = data_in["unit_c1"] if aktif_custom else 0
    jam_c1 = data_in["jam_c1"] if (unit_c1 > 0 and watt_c1 > 0) else 0.0

    nama_c2 = data_in["nama_c2"].strip() or "Alat Tambahan 2"
    if nama_c2 == nama_c1:
        nama_c2 = f"{nama_c2} (2)"
    watt_c2 = data_in["watt_c2"] if aktif_custom else 0
    unit_c2 = data_in["unit_c2"] if aktif_custom else 0
    jam_c2 = data_in["jam_c2"] if (unit_c2 > 0 and watt_c2 > 0) else 0.0

    batas_va = GOLONGAN_PLN[pilihan_golongan]["va"]
    TARIF_PLN = GOLONGAN_PLN[pilihan_golongan]["tarif"]

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
            "penyebab": "Pintu kulkas sering dibuka-tutup, karet pintu renggang, atau makanan panas langsung dimasukkan.",
            "solusi": "Dinginkan makanan sebelum masuk kulkas, pastikan pintu rapat, dan beri jarak 10 cm dari dinding."
        },
        pilihan_ac: {
            "penyebab": "AC menyala terlalu lama atau suhu disetel terlalu dingin (di bawah 22°C) sehingga kompresor bekerja penuh terus.",
            "solusi": "Atur suhu ideal <b>24°C–25°C</b>, pakai fitur <i>Timer/Sleep</i> saat dini hari, dan bersihkan filter rutin."
        },
        "Lampu LED (Per Titik)": {
            "penyebab": "Banyak lampu dibiarkan menyala pada siang hari atau di ruangan kosong.",
            "solusi": "Matikan lampu ruangan kosong, buka tirai di siang hari, dan pakai lampu otomatis untuk teras."
        },
        "Kipas Angin": {
            "penyebab": "Kipas berputar terus meski kamar kosong atau baling-baling berdebu tebal.",
            "solusi": "Matikan kipas saat keluar ruangan dan bersihkan debu pada baling-baling secara berkala."
        },
        "Sanyo (Pompa Air Otomatis)": {
            "penyebab": "Pompa hidup-mati sendiri akibat keran/pipa bocor halus atau pelampung toren rusak.",
            "solusi": "Cek keran menetes dan perbaiki pelampung otomatis toren agar pompa tidak bekerja terus."
        },
        "TV LED": {
            "penyebab": "TV menyala seharian tanpa ditonton atau terus dalam mode <i>standby</i> malam hari.",
            "solusi": "Matikan tombol utama TV setelah selesai menonton dan aktifkan fitur <i>Auto Sleep</i>."
        },
        "Water Heater": {
            "penyebab": "Pemanas air dibiarkan menyala sepanjang hari meski tidak sedang dipakai mandi.",
            "solusi": "Nyalakan hanya <b>15–20 menit sebelum mandi</b>, lalu matikan saklarnya segera."
        },
        "Magicom (Penanak Nasi)": {
            "penyebab": "Memasak nasi (daya 380 W) dilakukan berkali-kali dalam sehari.",
            "solusi": "Masak nasi sekaligus 1 kali di pagi hari untuk seharian, dan cabut steker bila nasi habis."
        },
        "Mesin Cuci": {
            "penyebab": "Mencuci pakaian sedikit demi sedikit hampir setiap hari.",
            "solusi": "Kumpulkan cucian dan cuci sekaligus 1–2 kali seminggu sesuai kapasitas tabung."
        }
    }

    DEFAULT_DIAG_KUSTOM = {
        "penyebab": "Durasi pemakaian alat tambahan ini terlalu lama dalam sehari.",
        "solusi": "Gunakan seperlunya dan cabut steker setelah selesai dipakai."
    }

    rincian_list = []
    total_kwh_hari = 0.0
    total_kwh_normal_patokan = 0.0
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
            total_kwh_normal_patokan += (daya_total_alat * spec["normal_jam"]) / 1000.0

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
                    "judul": f"{nama} ({unit} {satuan_txt}) — Menyala {jam:.1f} jam/hari (Wajar {spec['normal_jam']} jam/hari)",
                    "dampak": f"Menambah biaya listrik sekitar <b>+Rp {selisih_biaya:,.0f}/bulan</b> (atau <b>+Rp {selisih_biaya * 12:,.0f}/tahun</b>).",
                    "penyebab": info_diag["penyebab"],
                    "solusi": info_diag["solusi"],
                    "z_score": round(z_score, 2)
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
        total_kwh_normal_patokan += unit_magicom * ((JAM_PER_MASAK * WATT_MAGICOM_COOK) + (14.0 * WATT_MAGICOM_WARM)) / 1000.0

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
            "dampak": f"Menambah biaya listrik sekitar <b>+Rp {hemat_masak:,.0f}/bulan</b>.",
            "penyebab": info_diag["penyebab"],
            "solusi": info_diag["solusi"],
            "z_score": 2.5
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
        total_kwh_normal_patokan += (daya_puncak_cuci * 0.5) / 1000.0

    kwh_cuci_harian = (daya_puncak_cuci * jam_cuci_harian) / 1000.0
    biaya_cuci_bulan = (kwh_cuci_harian * 30.0) * TARIF_PLN
    total_kwh_hari += kwh_cuci_harian

    if unit_cuci > 0 and jam_cuci_minggu >= 8.0:
        hemat_cuci = (((jam_cuci_minggu - 3.5) / 7.0) * daya_puncak_cuci / 1000.0) * 30.0 * TARIF_PLN
        potensi_hemat_rp_total += hemat_cuci
        info_diag = DETAIL_DIAGNOSIS["Mesin Cuci"]
        diagnosis_boros_list.append({
            "judul": f"Mesin Cuci ({unit_cuci} unit) — Dipakai {jam_cuci_minggu:.1f} jam/minggu (Normal 2–4 jam/minggu)",
            "dampak": f"Menambah biaya listrik sekitar <b>+Rp {hemat_cuci:,.0f}/bulan</b>.",
            "penyebab": info_diag["penyebab"],
            "solusi": info_diag["solusi"],
            "z_score": 2.4
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
        total_kwh_normal_patokan += (25.0 * 24.0) / 1000.0

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
                    "judul": f"{nama} ({input_unit_umum[nama]} unit) — Menyala {input_jam_umum[nama]:.1f} jam/hari (Normal {spec['normal_jam']} jam/hari)",
                    "dampak": f"Berkontribusi menaikkan pengeluaran sekitar <b>+Rp {selisih_biaya:,.0f}/bulan</b>.",
                    "penyebab": info_diag["penyebab"],
                    "solusi": info_diag["solusi"],
                    "z_score": 1.8
                })

    df_rincian = pd.DataFrame(rincian_list)
    total_biaya_murni = df_rincian["Biaya Listrik/Bulan (Rp)"].sum()
    if total_biaya_murni > 0:
        df_rincian["Porsi Beban (%)"] = (df_rincian["Biaya Listrik/Bulan (Rp)"] / total_biaya_murni * 100).round(1)
    else:
        df_rincian["Porsi Beban (%)"] = 0.0

    prediksi_tagihan_murni = max(0.0, float(model_regresi.predict([[total_kwh_hari]])[0])) if total_kwh_hari > 0 else 0.0
    total_kwh_bulan = total_kwh_hari * 30.0
    potensi_hemat_tahun = potensi_hemat_rp_total * 12.0

    rasio_kwh_norm = total_kwh_hari / max(total_kwh_normal_patokan, 0.5)
    rasio_mcb = total_watt_terpasang / max(batas_va, 1)
    skor_fuzzy, lencana_fuzzy, warna_lencana = hitung_skor_fuzzy_efisiensi(rasio_kwh_norm, rasio_mcb, len(diagnosis_boros_list))

    render_animasi_dashboard(is_boros, skor_fuzzy, lencana_fuzzy, warna_lencana, total_kwh_hari, total_watt_terpasang, batas_va, pilihan_golongan, TARIF_PLN)

    rasio_default_bersih = ((100000.0 - 2500.0) * 0.92) / 100000.0
    estimasi_belanja_awal = prediksi_tagihan_murni / rasio_default_bersih

    df_urut_biaya = df_rincian[df_rincian["Energi Harian (kWh)"] > 0].sort_values(by="Biaya Listrik/Bulan (Rp)", ascending=False)
    if len(df_urut_biaya) > 0:
        juara_boros = df_urut_biaya.iloc[0]["Peralatan Rumah"]
        porsi_juara = df_urut_biaya.iloc[0]["Porsi Beban (%)"]
        biaya_juara = df_urut_biaya.iloc[0]["Biaya Listrik/Bulan (Rp)"]
    else:
        juara_boros, porsi_juara, biaya_juara = "-", 0.0, 0.0

    if potensi_hemat_rp_total > 0:
        liter_bensin = potensi_hemat_rp_total / 10000.0
        gelas_es_teh = int(potensi_hemat_rp_total / 4000.0)
        poin_3_intinya = (
            f"3️⃣ <b>Potensi Uang Selamat:</b> Jika pemakaian berlebih dirapikan sesuai saran di bawah, Anda bisa hemat "
            f"<b>Rp {potensi_hemat_rp_total:,.0f}/bulan</b> (setara <b>±{liter_bensin:.1f} liter Pertalite</b> atau <b>{gelas_es_teh} gelas es teh</b> tiap bulan!)."
        )
    else:
        poin_3_intinya = (
            "3️⃣ <b>Kabar Baik:</b> Durasi nyala alat di rumah Anda sudah terkendali. Tetap cabut steker yang tidak terpakai agar makin irit."
        )

    render_html(f"""
    <div class="summary-tldr-box">
        <div class="summary-tldr-title">💡 Intinya Begini... (Ringkasan Cepat 10 Detik)</div>
        <div class="summary-tldr-item">1️⃣ <b>Estimasi Keluar Uang:</b> Belanja token / tagihan listrik rumah Anda diperkirakan sekitar <b>Rp {estimasi_belanja_awal:,.0f}/bulan</b> (listrik murni Rp {prediksi_tagihan_murni:,.0f}).</div>
        <div class="summary-tldr-item">2️⃣ <b>Penyedot Biaya Terbesar:</b> Paling banyak tersedot oleh <b>{juara_boros}</b> sebesar <b>{porsi_juara:.1f}%</b> dari total tagihan (<b>Rp {biaya_juara:,.0f}/bulan</b>).</div>
        <div class="summary-tldr-item">{poin_3_intinya}</div>
    </div>
    """)

    if total_watt_terpasang > batas_va:
        alat_aktif_daya.sort(key=lambda item: item[1], reverse=True)
        top_alat_str = [f"**{n} ({w:,} W)**" for n, w in alat_aktif_daya[:4]]
        daftar_alat_warning = ", ".join(top_alat_str[:-1]) + ", dan " + top_alat_str[-1] if len(top_alat_str) > 1 else top_alat_str[0]
        st.warning(
            f"**Peringatan Beban Maksimal Meteran ({batas_va} VA):** Total daya semua alat aktif mencapai "
            f"**{total_watt_terpasang:,} Watt** (melebihi **{batas_va} VA**). Hindari menyalakan {daftar_alat_warning} bersamaan agar listrik tidak *jeglek*."
        )

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
                <span class="kpi-label">Skor Efisiensi Rumah</span>
                <span class="kpi-icon">🏅</span>
            </div>
            <div class="kpi-value" style="color: {warna_lencana};">{skor_fuzzy} / 100</div>
            <div class="kpi-sub">{lencana_fuzzy.replace('Lencana: ', '')}</div>
        </div>
        <div class="kpi-card" style="border-color: {'rgba(244,63,94,0.6)' if is_boros else 'rgba(16,185,129,0.6)'};">
            <div class="kpi-top">
                <span class="kpi-label">Estimasi Biaya / Bulan</span>
                <span class="kpi-icon">💰</span>
            </div>
            <div class="kpi-value" style="color: {'#f43f5e' if is_boros else '#10b981'};">Rp {estimasi_belanja_awal:,.0f}</div>
            <div class="kpi-sub">Listrik murni: Rp {prediksi_tagihan_murni:,.0f}</div>
        </div>
    </div>
    """)

    tab_ringkasan, tab_grafik, tab_tabel = st.tabs([
        "⚡ Token, Simulasi Hemat & Saklar MCB",
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
            persen_ppj = tk2.slider("Pajak Daerah / PPJ (%)", min_value=3, max_value=10, value=8, step=1)

            biaya_admin_bank = 2500.0
            saldo_setelah_admin = max(0.0, nominal_token - biaya_admin_bank)
            rupiah_bersih_token = saldo_setelah_admin * (1.0 - (persen_ppj / 100.0))
            kwh_didapat = rupiah_bersih_token / TARIF_PLN

            if total_kwh_hari > 0:
                estimasi_hari_token = kwh_didapat / total_kwh_hari
                teks_bertahan = f"<span style='color:{T['accent']}; font-size:20px; font-weight:800;'>±{estimasi_hari_token:.1f} Hari</span>"
                persen_sisa_visual = min(100.0, (estimasi_hari_token / 30.0) * 100.0)
            else:
                estimasi_hari_token = 0.0
                teks_bertahan = f"<span style='color:{T['accent']}; font-weight:800;'>Tidak terpakai</span>"
                persen_sisa_visual = 100.0

            rasio_kotor_ke_bersih = rupiah_bersih_token / max(nominal_token, 1)
            estimasi_belanja_token_bulan = prediksi_tagihan_murni / max(rasio_kotor_ke_bersih, 0.5)

            render_html(f"""
            <div class="glass-card">
                <div class="feature-title">🔋 Ketahanan Pulsa / Token Listrik Prabayar</div>
                <div style="font-size:13.5px; color:{T['text_sub']}; line-height:1.6;">
                    • Saldo listrik bersih (potong Admin Rp 2.500 & PPJ {persen_ppj}%): <b>Rp {rupiah_bersih_token:,.0f}</b> (<b>{kwh_didapat:.2f} kWh</b>)<br>
                    • Token <b>Rp {nominal_token:,.0f}</b> diperkirakan habis dalam: {teks_bertahan}
                </div>
                <div class="progress-track">
                    <div class="progress-fill-anim" style="width:{persen_sisa_visual:.1f}%; background: {T['btn_grad']};"></div>
                </div>
                <div style="font-size:12px; color:{T['text_muted']};">
                    Setara pengeluaran beli token: <b>±Rp {estimasi_belanja_token_bulan:,.0f} / bulan</b>.
                </div>
            </div>
            """)

        with col_budget:
            if target_anggaran > 0:
                persen_anggaran = (estimasi_belanja_token_bulan / max(target_anggaran, 1)) * 100.0
                lebar_bar = min(persen_anggaran, 100.0)
                selisih_anggaran = target_anggaran - estimasi_belanja_token_bulan

                if selisih_anggaran >= 0:
                    warna_bar_budget = "linear-gradient(90deg, #059669, #34d399, #10b981)"
                    status_budget_txt = f"<span style='color:#10b981; font-weight:800;'>MASUK ANGGARAN</span> — Masih hemat <b>Rp {selisih_anggaran:,.0f}/bulan</b> dari batas maksimal Anda."
                else:
                    warna_bar_budget = "linear-gradient(90deg, #e11d48, #fb7185, #f43f5e)"
                    status_budget_txt = f"<span style='color:#f43f5e; font-weight:800;'>MELEBIHI ANGGARAN</span> — Lebih besar <b>+Rp {abs(selisih_anggaran):,.0f}/bulan</b> (<b>+Rp {abs(selisih_anggaran) * 12:,.0f}/tahun</b>)."

                render_html(f"""
                <div class="glass-card">
                    <div class="feature-title">🎯 Kesesuaian dengan Target Anggaran Bulanan</div>
                    <div style="display:flex; justify-content:space-between; font-size:13.5px; color:{T['text_sub']}; margin-top:8px;">
                        <span>Estimasi Belanja: <b>Rp {estimasi_belanja_token_bulan:,.0f}</b></span>
                        <span>Target Anda: <b>Rp {target_anggaran:,.0f}</b> ({persen_anggaran:.1f}%)</span>
                    </div>
                    <div class="progress-track" style="margin-top:16px; margin-bottom:16px;">
                        <div class="progress-fill-anim" style="width:{lebar_bar:.1f}%; background: {warna_bar_budget};"></div>
                    </div>
                    <div style="font-size:13.5px; color:{T['text_main']}; line-height:1.55;">
                        {status_budget_txt}
                    </div>
                </div>
                """)
            else:
                render_html(f"""
                <div class="glass-card">
                    <div class="feature-title">🎯 Target Anggaran Bulanan (Opsional)</div>
                    <div style="font-size:13.5px; color:{T['text_sub']}; line-height:1.6; margin-top:8px;">
                        Anda belum mengisi batas anggaran bulanan (nilai saat ini <b>Rp 0</b>).<br>
                        Jika ingin membandingkan apakah tagihan <b>±Rp {estimasi_belanja_token_bulan:,.0f}/bulan</b> masuk anggaran Anda, tekan tombol <b>⬅️ Edit Input</b> di pojok atas kiri.
                    </div>
                </div>
                """)

        col_whatif, col_mcb = st.columns(2)

        with col_whatif:
            st.markdown("#### 🎛️ Simulator \"What-If\" Penghematan")
            persen_pangkas = st.slider(
                "Simulasi: Jika Anda mengurangi lama nyala alat sebesar (%)",
                min_value=0,
                max_value=50,
                value=20,
                step=5,
                help="Geser untuk melihat langsung berapa Rupiah yang dihemat dan berapa hari token listrik bertambah awet."
            )
            kwh_baru = total_kwh_hari * (1.0 - persen_pangkas / 100.0)
            biaya_baru_token = estimasi_belanja_token_bulan * (1.0 - persen_pangkas / 100.0)
            hemat_whatif_bulan = estimasi_belanja_token_bulan - biaya_baru_token
            hari_token_baru = (kwh_didapat / kwh_baru) if kwh_baru > 0 else 0.0
            selisih_hari_token = max(0.0, hari_token_baru - estimasi_hari_token)

            render_html(f"""
            <div class="glass-card">
                <div style="font-size:13.5px; color:{T['text_sub']}; line-height:1.65;">
                    • Pengeluaran turun dari <b>Rp {estimasi_belanja_token_bulan:,.0f}</b> menjadi <b style="color:#10b981; font-size:16px;">Rp {biaya_baru_token:,.0f}/bulan</b>.<br>
                    • Uang yang berhasil dihemat: <b style="color:{T['accent']};">+Rp {hemat_whatif_bulan:,.0f}/bulan</b> (<b>+Rp {hemat_whatif_bulan * 12:,.0f}/tahun</b>).<br>
                    • Token <b>Rp {nominal_token:,.0f}</b> menjadi lebih awet <b>+{selisih_hari_token:.1f} hari</b> (total bertahan <b>±{hari_token_baru:.1f} hari</b>).
                </div>
            </div>
            """)

        with col_mcb:
            st.markdown(f"#### 🔌 Simulasi Saklar MCB ({batas_va} VA)")
            alat_mcb_opsi = [f"{n} ({w} W)" for n, w in alat_aktif_daya if w > 0]
            alat_mcb_map = {f"{n} ({w} W)": w for n, w in alat_aktif_daya if w > 0}
            dipilih_serentak = st.multiselect(
                "Centang alat yang dinyalakan BERSAMAAN:",
                options=alat_mcb_opsi,
                default=alat_mcb_opsi[:3],
                key="simulasi_mcb_serentak"
            )
            watt_serentak = sum(alat_mcb_map[item] for item in dipilih_serentak)
            is_jeglek = watt_serentak > batas_va

            if is_jeglek:
                render_html(f"""
                <div class="glass-card" style="border: 2px solid #f43f5e;">
                    <div style="font-size:16px; font-weight:800; color:#f43f5e;">⚡ CETREK / JEGLEK! (Listrik Turun)</div>
                    <div style="font-size:13.5px; color:{T['text_sub']}; margin-top:6px; line-height:1.55;">
                        Beban serentak mencapai <b>{watt_serentak:,} Watt</b>, melampaui batas meteran <b>{batas_va} VA</b> sebesar <b>+{watt_serentak - batas_va:,} W</b>. Matikan salah satu alat berdaya besar!
                    </div>
                </div>
                """)
            else:
                sisa_va = batas_va - watt_serentak
                render_html(f"""
                <div class="glass-card" style="border: 2px solid #10b981;">
                    <div style="font-size:16px; font-weight:800; color:#10b981;">✅ SAKLAR MCB AMAN (ON)</div>
                    <div style="font-size:13.5px; color:{T['text_sub']}; margin-top:6px; line-height:1.55;">
                        Beban serentak <b>{watt_serentak:,} Watt</b> dari <b>{batas_va} VA</b>. Masih ada sisa kapasitas aman sebesar <b>{sisa_va:,} Watt</b>.
                    </div>
                </div>
                """)

        if is_boros or len(diagnosis_boros_list) > 0:
            st.markdown("#### 🚨 Daftar Pemakaian Berlebih (Boros) & Cara Mengatasinya")
            for item in diagnosis_boros_list:
                info_z = f" • <i>(Skor Z: {item['z_score']}σ)</i>" if mode_dosen else ""
                render_html(f"""
                <div class="diag-card">
                    <div class="diag-header">⚠️ {item['judul']}{info_z}</div>
                    <div class="diag-row">• <b>Dampak Pengeluaran:</b> {item['dampak']}</div>
                    <div class="diag-row">• <b>Penyebab Utama:</b> {item['penyebab']}</div>
                    <div class="diag-row" style="color: #10b981;">• <b>Cara Mengatasi:</b> {item['solusi']}</div>
                </div>
                """)

            if potensi_hemat_rp_total > 0:
                st.success(
                    f"**Total Uang yang Bisa Dihemat:** Dengan menerapkan langkah di atas, pengeluaran listrik rumah Anda berkurang sekitar "
                    f"**Rp {potensi_hemat_rp_total:,.0f}/bulan** atau **Rp {potensi_hemat_tahun:,.0f}/tahun**!"
                )
        else:
            solusi_terbesar = DETAIL_DIAGNOSIS.get(juara_boros, DEFAULT_DIAG_KUSTOM)["solusi"]
            render_html(f"""
            <div class="explain-box" style="border-left-color: #10b981;">
                <div class="explain-title" style="color: #10b981;">✅ Pemakaian Sudah Wajar — Tips Supaya Lebih Hemat Lagi</div>
                <div class="explain-item">• Tidak ditemukan peralatan yang menyala secara berlebihan di luar kewajaran harian.</div>
                <div class="explain-item">• Peralatan yang paling banyak menyerap biaya saat ini adalah <b>{juara_boros}</b> (<b>{porsi_juara:.1f}%</b> dari total tagihan).</div>
                <div class="explain-item">• <b>Saran Penghematan:</b> {solusi_terbesar}</div>
            </div>
            """)

    with tab_grafik:
        fig1, ax1 = plt.subplots(figsize=(10.5, 3.6))
        fig1.patch.set_facecolor(T["chart_fig"])
        ax1.set_facecolor(T["chart_ax"])

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

            ax1.bar(x - lebar / 2, kwh_normal, width=lebar, label="Batas Wajar Harian (kWh)", color="#64748b")
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
                    color=T["chart_text"]
                )

            ax1.set_title("1. Perbandingan Pemakaian Listrik Harian per Alat (kWh)", fontsize=11.5, fontweight="bold", color=T["chart_text"], pad=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(nama_alat, rotation=15, ha="right", fontsize=9, color=T["chart_text"])
            ax1.set_ylabel("Energi (kWh / hari)", fontsize=9, color=T["chart_text"])
            ax1.tick_params(axis="y", colors=T["chart_text"], labelsize=8.5)
            ax1.set_ylim(0, max(max(kwh_aktual), max(kwh_normal)) * 1.24 + 0.2)
            leg1 = ax1.legend(fontsize=8.5, loc="upper right", facecolor=T["chart_fig"], edgecolor=T["chart_grid"])
            for text in leg1.get_texts():
                text.set_color(T["chart_text"])
            ax1.grid(axis="y", color=T["chart_grid"], alpha=0.5, linestyle="--")
        else:
            ax1.text(0.5, 0.5, "Belum ada peralatan yang aktif", ha="center", va="center", color=T["chart_text"])

        for spine in ["top", "right"]:
            ax1.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax1.spines[spine].set_color(T["chart_grid"])

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
                f"Terdapat batang berwarna <b>merah</b> pada <b>{', '.join(alat_merah)}</b> karena lama nyalanya melewati batas wajar."
                if len(alat_merah) > 0 else
                "Semua batang pemakaian Anda berwarna <b>hijau</b>, artinya tidak ada peralatan yang dipakai secara berlebihan."
            )

            render_html(f"""
            <div class="explain-box">
                <div class="explain-title">📖 Cara Membaca Grafik 1</div>
                <div class="explain-item">• <b>Arti Warna Batang:</b> Batang <b>abu-abu</b> adalah patokan normal, batang <b>hijau</b> berarti aman, dan batang <b>merah</b> menandakan alat tersebut dipakai terlalu lama (boros).</div>
                <div class="explain-item">• <b>Kesimpulan Rumah Anda:</b> Pemakaian energi terbesar berasal dari <b>{nama_max_g1}</b> sebesar <b>{nilai_max_g1:.2f} kWh/hari</b>. {ket_merah}</div>
            </div>
            """)

        fig2, ax2 = plt.subplots(figsize=(10.5, 3.6))
        fig2.patch.set_facecolor(T["chart_fig"])
        ax2.set_facecolor(T["chart_ax"])

        ax2.scatter(hist_kwh, hist_tagihan / 1000.0, color="#38bdf8", alpha=0.65, s=38, label="Data Pemakaian Harian Normal (120 Hari)")

        x_max_plot = max(hist_kwh.max(), total_kwh_hari, 1.0) * 1.06
        x_line = np.linspace(0, x_max_plot, 100)
        y_line = model_regresi.predict(x_line.reshape(-1, 1)) / 1000.0
        ax2.plot(x_line, y_line, color="#f59e0b", linewidth=2.4, label="Garis Prediksi Kenaikan Biaya")

        ax2.scatter(
            [total_kwh_hari],
            [prediksi_tagihan_murni / 1000.0],
            color="#f43f5e" if is_boros else "#10b981",
            s=220,
            marker="*",
            edgecolors=T["chart_text"],
            linewidths=1.0,
            zorder=5,
            label=f"Posisi Rumah Anda ({total_kwh_hari:.2f} kWh → Rp {prediksi_tagihan_murni:,.0f} murni)"
        )

        ax2.set_title(f"2. Grafik Kenaikan Biaya Listrik Bulanan — {pilihan_golongan.split('(')[0].strip()}", fontsize=11.5, fontweight="bold", color=T["chart_text"], pad=12)
        ax2.set_xlabel("Total Pemakaian Listrik Harian (kWh / hari)", fontsize=9, color=T["chart_text"])
        ax2.set_ylabel("Biaya Listrik Murni (Ribu Rp)", fontsize=9, color=T["chart_text"])
        ax2.tick_params(colors=T["chart_text"], labelsize=8.5)
        leg2 = ax2.legend(fontsize=8.5, loc="upper left", facecolor=T["chart_fig"], edgecolor=T["chart_grid"])
        for text in leg2.get_texts():
            text.set_color(T["chart_text"])
        ax2.grid(True, color=T["chart_grid"], alpha=0.5, linestyle="--")

        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax2.spines[spine].set_color(T["chart_grid"])

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        render_html(f"""
        <div class="explain-box">
            <div class="explain-title">📖 Cara Membaca Grafik 2</div>
            <div class="explain-item">• <b>Arti Simbol Grafik:</b> Semakin ke kanan posisi <b>ikon bintang</b>, semakin besar konsumsi kWh harian Anda dan semakin tinggi tagihan rupiahnya mengikuti <b>garis miring oranye</b>.</div>
            <div class="explain-item">• <b>Rincian Angka Anda:</b> Total pemakaian <b>{total_kwh_hari:.2f} kWh/hari</b> (<b>{total_kwh_bulan:.1f} kWh/bulan</b>) dikali tarif PLN <b>Rp {TARIF_PLN:,.2f}/kWh</b> menghasilkan biaya listrik murni <b>Rp {prediksi_tagihan_murni:,.0f}</b> (atau <b>±Rp {estimasi_belanja_token_bulan:,.0f}/bulan</b> termasuk PPJ &amp; admin token).</div>
        </div>
        """)

    with tab_tabel:
        st.markdown("#### 📋 Rincian Perhitungan Daya, Biaya & Porsi Beban Tiap Alat")
        df_tampil = df_rincian[df_rincian["Jumlah"] > 0] if len(df_rincian[df_rincian["Jumlah"] > 0]) > 0 else df_rincian
        st.dataframe(
            df_tampil,
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

        df_export = df_tampil.copy()
        baris_total = pd.DataFrame([{
            "Peralatan Rumah": f"TOTAL KESELURUHAN ({pilihan_golongan.split('(')[0].strip()})",
            "Jumlah": int(df_tampil["Jumlah"].sum()),
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
            mime="text/csv"
        )

    if mode_dosen:
        with st.expander("🎓 Dokumentasi Akademik & Arsitektur Kecerdasan Buatan (Presentasi Dosen)", expanded=True):
            st.markdown(f"""
            **1. Ringkasan Inferensi Model Saat Ini**
            * **Status Isolation Forest:** `{'Anomali / Boros (-1)' if is_boros else 'Normal / Wajar (+1)'}` dengan *Decision Score* = `{indeks_wajar:.4f}`
            * **Skor Efisiensi Logika Fuzzy:** `{skor_fuzzy} / 100` (`{lencana_fuzzy}`) berdasarkan fungsi keanggotaan rasio konsumsi terhadap patokan normal (`{rasio_kwh_norm:.2f}x`) dan rasio beban MCB (`{rasio_mcb * 100:.1f}%`).
            * **Persamaan Linear Regression:** $\\hat{{y}} = {model_regresi.intercept_:.2f} + ({model_regresi.coef_[0]:.2f} \\times x)$

            **2. Rumus Fisika Daya & Ketahanan Token Prabayar**
            """)
            st.latex(r"E_{\text{harian}} = \frac{P \times t}{1.000} \quad (\text{kWh/hari}), \qquad \text{Biaya}_{\text{murni}} = \left(\sum E_{\text{harian}}\right) \times 30 \times T_{\text{PLN}}")
            st.latex(r"E_{\text{token}} = \frac{(N_{\text{token}} - \text{Admin}) \times \left(1 - \frac{p_{\text{PPJ}}}{100}\right)}{T_{\text{PLN}}}, \qquad Z = \frac{t - \mu}{\sigma}")


# ==========================================
# FOOTER HAK CIPTA, LOGO UNIVERSITAS SEMARANG (USM) & KONTAK WHATSAPP
# ==========================================
logo_usm_src = get_usm_logo_data_uri()

render_html(f"""
<div class="footer-card">
    <div class="footer-brand">
        <img src="{logo_usm_src}" alt="Logo Universitas Semarang (USM)" style="width: 80px; height: auto; border-radius: 12px; border: 1.5px solid #facc15; background-color: #000000; padding: 3px; flex-shrink: 0;">
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

# ==========================================
# EKSEKUSI AUTO-SCROLL KE PALING ATAS SETELAH HALAMAN SELESAI DIMUAT
# ==========================================
if st.session_state.get("perlu_scroll_atas", False):
    scroll_ke_atas()
    st.session_state["perlu_scroll_atas"] = False