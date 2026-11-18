import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# ==========================================
# KONFIGURASI HALAMAN & CSS RESPONSIF PADAT
# ==========================================
st.set_page_config(
    page_title="Smart Home AI - Energy Monitor & Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_html(raw_html: str):
    """Membersihkan indentasi agar HTML tidak terbaca sebagai blok kode Markdown."""
    clean_html = "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)


render_html("""
<style>
    .block-container {
        padding-top: 3.2rem !important;
        padding-bottom: 1.8rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1320px;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
        gap: 12px;
        margin-top: 12px;
        margin-bottom: 14px;
    }
    .kpi-card {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 16px;
        color: #f8fafc;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .kpi-label {
        font-size: 11.5px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .kpi-value {
        font-size: 20px;
        font-weight: 800;
        margin-top: 4px;
        color: #f8fafc;
    }
    .kpi-sub {
        font-size: 11.5px;
        color: #64748b;
        margin-top: 2px;
    }
    .feature-card {
        background: linear-gradient(145deg, #0f172a, #172033);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 18px;
        color: #f8fafc;
        margin-bottom: 12px;
    }
    .feature-title {
        font-size: 14.5px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 8px;
    }
    .progress-track {
        width: 100%;
        height: 10px;
        background-color: #1e293b;
        border-radius: 999px;
        overflow: hidden;
        margin: 8px 0;
        border: 1px solid #334155;
    }
    .diag-card {
        background: #180d10;
        border: 1px solid #ef4444;
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 14px 18px;
        color: #f8fafc;
        margin-bottom: 10px;
    }
    .diag-header {
        font-size: 15px;
        font-weight: 800;
        color: #f87171;
        margin-bottom: 6px;
    }
    .diag-row {
        font-size: 13.5px;
        color: #e2e8f0;
        margin-bottom: 5px;
        line-height: 1.45;
    }
    .explain-box {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        color: #e2e8f0;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    .explain-title {
        font-size: 14.5px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 8px;
    }
    .explain-item {
        font-size: 13.5px;
        margin-bottom: 6px;
        line-height: 1.5;
        color: #cbd5e1;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-top: 2.8rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        .kpi-value {
            font-size: 17px;
        }
    }
</style>
""")

# Daftar Golongan Daya Listrik PLN & Tarif Resmi (Rp/kWh)
GOLONGAN_PLN = {
    "450 VA (Subsidi - Rp 415/kWh)": {"va": 450, "tarif": 415.0},
    "900 VA (Subsidi - Rp 605/kWh)": {"va": 900, "tarif": 605.0},
    "900 VA (Non-Subsidi - Rp 1.352/kWh)": {"va": 900, "tarif": 1352.0},
    "1300 VA (Rumah Tangga - Rp 1.444,70/kWh)": {"va": 1300, "tarif": 1444.70},
    "2200 VA (Rumah Tangga - Rp 1.444,70/kWh)": {"va": 2200, "tarif": 1444.70},
    "3500 - 5500 VA (R-2 - Rp 1.699,53/kWh)": {"va": 3500, "tarif": 1699.53},
}

# Pilihan Kapasitas AC
OPSI_AC = {
    "AC 1/2 PK (400 Watt)": 400,
    "AC 1 PK (850 Watt)": 850,
}

# Spesifikasi Daya Magicom (2 Siklus Kerja)
WATT_MAGICOM_COOK = 350   # Daya saat memasak nasi
WATT_MAGICOM_WARM = 45    # Daya saat menghangatkan nasi
JAM_PER_MASAK = 0.75      # ±45 menit per siklus memasak

# Spesifikasi Mesin Cuci (Berbasis Jadwal Mingguan)
WATT_MESIN_CUCI = 450
JAM_PER_SESI_CUCI = 1.5   # ±1,5 jam per 1x sesi cuci + bilas + peras


# ==========================================
# 1. PEMBANGKIT DATASET HISTORIS & PELATIHAN MODEL AI
# ==========================================
@st.cache_resource
def latih_model_ai(tarif_pln, konfig_umum_tuple, unit_magicom, unit_cuci):
    rng = np.random.default_rng(42)
    jumlah_hari = 120

    data_fitur = {}
    total_kwh_harian = np.zeros(jumlah_hari, dtype=float)

    for nama, watt, unit, normal_jam, std_jam in konfig_umum_tuple:
        if unit == 0:
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

        sampel_warm = rng.normal(loc=16.5 * unit_magicom, scale=4.2 * np.sqrt(unit_magicom), size=jumlah_hari)
        sampel_warm = np.clip(sampel_warm, 0.0, 24.0 * unit_magicom)

    data_fitur["Magicom_Cook"] = sampel_cook
    data_fitur["Magicom_Warm"] = sampel_warm
    total_kwh_harian += ((sampel_cook * WATT_MAGICOM_COOK) + (sampel_warm * WATT_MAGICOM_WARM)) / 1000.0

    if unit_cuci == 0:
        sampel_cuci = np.zeros(jumlah_hari, dtype=float)
    else:
        normal_jam_cuci_harian = (1.0 * JAM_PER_SESI_CUCI) / 7.0
        sampel_cuci = rng.normal(
            loc=normal_jam_cuci_harian * unit_cuci,
            scale=0.08 * np.sqrt(unit_cuci),
            size=jumlah_hari
        )
        sampel_cuci = np.clip(sampel_cuci, 0.0, JAM_PER_SESI_CUCI * unit_cuci)

    data_fitur["Mesin_Cuci"] = sampel_cuci
    total_kwh_harian += (sampel_cuci * WATT_MESIN_CUCI) / 1000.0

    df_hist = pd.DataFrame(data_fitur)

    tagihan_bulanan = (total_kwh_harian * 30.0 * tarif_pln) + rng.normal(0, 8000, size=jumlah_hari)
    tagihan_bulanan = np.clip(tagihan_bulanan, 0.0, None)

    model_anomali = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model_anomali.fit(df_hist.to_numpy())

    model_regresi = LinearRegression()
    model_regresi.fit(total_kwh_harian.reshape(-1, 1), tagihan_bulanan)

    return df_hist, total_kwh_harian, tagihan_bulanan, model_anomali, model_regresi


# ==========================================
# 2. KOMPONEN ANIMASI BANNER & GAUGE
# ==========================================
def render_animasi_dashboard(is_boros, skor_anomali, total_kwh_hari, total_watt, batas_va, nama_golongan, tarif_pln):
    rasio_beban = float(np.clip(total_kwh_hari / 28.0, 0.08, 1.0))
    persen_mcb = int(np.clip((total_watt / max(batas_va, 1)) * 100, 0, 999))
    keliling_ring = 226.2
    offset_ring = keliling_ring * (1.0 - rasio_beban)

    durasi_aliran = 0.42 if is_boros else max(0.85, 2.2 - rasio_beban * 1.2)
    durasi_denyut = 0.65 if is_boros else 1.8

    if is_boros:
        warna_utama = "#ef4444"
        warna_aksen = "#f59e0b"
        badge_bg = "rgba(239, 68, 68, 0.18)"
        judul_status = "PERINGATAN: TERDETEKSI ANOMALI KEBOROSAN!"
        sub_status = "Pola pemakaian menyimpang dari kebiasaan normal. Lihat diagnosis & solusi lengkap di bawah."
        label_meter = "ANOMALI"
    else:
        warna_utama = "#10b981"
        warna_aksen = "#38bdf8"
        badge_bg = "rgba(16, 185, 129, 0.18)"
        judul_status = "STATUS NORMAL: PEMAKAIAN LISTRIK EFISIEN"
        sub_status = "Konsumsi energi terkendali dalam batas wajar sesuai pola harian rumah tangga."
        label_meter = "EFISIEN"

    render_html(f"""
    <style>
        @keyframes aliranListrik {{
            from {{ stroke-dashoffset: 36; }}
            to {{ stroke-dashoffset: 0; }}
        }}
        @keyframes denyutRadar {{
            0% {{ transform: scale(0.9); opacity: 0.9; }}
            70% {{ transform: scale(1.65); opacity: 0; }}
            100% {{ transform: scale(1.65); opacity: 0; }}
        }}
        @keyframes gelombangDaya {{
            0% {{ stroke-dashoffset: 120; }}
            100% {{ stroke-dashoffset: 0; }}
        }}
        @keyframes kilauBadge {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.65; }}
        }}
        .smart-banner {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: radial-gradient(circle at top right, #1e293b 0%, #0f172a 70%);
            border: 2px solid {warna_utama};
            border-radius: 14px;
            padding: 18px 22px;
            color: #f8fafc;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
            margin-bottom: 6px;
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
            animation: gelombangDaya {durasi_aliran * 1.5:.2f}s linear infinite;
        }}
        .badge-anim {{
            animation: kilauBadge {durasi_denyut * 1.6:.2f}s ease-in-out infinite;
        }}
        @media (max-width: 640px) {{
            .judul-hero {{ font-size: 17px !important; }}
            .panel-svg {{ width: 100% !important; justify-content: center !important; }}
        }}
    </style>
    <div class="smart-banner">
        <div style="flex: 1 1 300px;">
            <div class="badge-anim" style="display: inline-flex; align-items: center; gap: 7px; background-color: {badge_bg}; color: {warna_utama}; border: 1px solid {warna_utama}; padding: 3px 11px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.4px;">
                <span style="width: 8px; height: 8px; background: {warna_utama}; border-radius: 50%; display: inline-block;"></span>
                PLN {nama_golongan.split('(')[0].strip()} • TARIF Rp {tarif_pln:,.2f}/kWh
            </div>
            <div class="judul-hero" style="font-size: 20px; font-weight: 800; margin-top: 9px; color: {warna_utama}; line-height: 1.25;">
                {judul_status}
            </div>
            <div style="font-size: 13px; color: #cbd5e1; margin-top: 6px; line-height: 1.4;">
                {sub_status}
            </div>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">
                Skor Isolation Forest: <b style="color: #f8fafc;">{skor_anomali:.4f}</b> &nbsp;|&nbsp; Rasio Beban Puncak MCB: <b style="color: {warna_aksen};">{persen_mcb}%</b>
            </div>
        </div>
        <div class="panel-svg" style="flex: 0 1 auto; display: flex; flex-wrap: wrap; align-items: center; gap: 10px;">
            <svg width="235" height="110" viewBox="0 0 265 122">
                <rect x="3" y="4" width="259" height="114" rx="12" fill="#090e1a" stroke="#334155" stroke-width="1.5" />
                <circle cx="42" cy="61" r="22" fill="{badge_bg}" stroke="{warna_utama}" stroke-width="2" />
                <polygon points="44,46 34,62 42,62 39,76 52,59 43,59" fill="{warna_aksen}" />
                <text x="42" y="99" text-anchor="middle" fill="#94a3b8" font-size="9" font-weight="700">METER PLN</text>
                <line x1="66" y1="53" x2="144" y2="53" stroke="#1e293b" stroke-width="4" stroke-linecap="round" />
                <line class="kabel-arus" x1="66" y1="53" x2="144" y2="53" stroke="{warna_utama}" stroke-width="3.5" stroke-linecap="round" />
                <line x1="66" y1="69" x2="144" y2="69" stroke="#1e293b" stroke-width="4" stroke-linecap="round" />
                <line class="kabel-arus" x1="66" y1="69" x2="144" y2="69" stroke="{warna_aksen}" stroke-width="3" stroke-linecap="round" />
                <rect x="78" y="23" width="54" height="16" rx="8" fill="{badge_bg}" stroke="{warna_utama}" stroke-width="1" />
                <text x="105" y="34" text-anchor="middle" fill="{warna_utama}" font-size="8.5" font-weight="800">{label_meter}</text>
                <polygon points="150,51 204,19 252,51" fill="#1e293b" stroke="{warna_utama}" stroke-width="2.3" stroke-linejoin="round" />
                <rect x="160" y="51" width="82" height="49" rx="4" fill="#1e293b" stroke="{warna_utama}" stroke-width="2" />
                <rect x="171" y="60" width="60" height="25" rx="4" fill="#060911" stroke="#475569" stroke-width="1" />
                <polyline class="gelombang-meter" points="175,72 184,72 189,64 195,81 201,65 207,79 213,72 227,72" fill="none" stroke="{warna_utama}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />
                <circle class="gelombang-radar" cx="204" cy="16" r="6" fill="none" stroke="{warna_utama}" stroke-width="2" />
                <circle cx="204" cy="16" r="4" fill="{warna_utama}" />
                <text x="201" y="112" text-anchor="middle" fill="#94a3b8" font-size="9" font-weight="700">SMART HOME</text>
            </svg>
            <svg width="98" height="110" viewBox="0 0 102 114">
                <rect x="2" y="4" width="98" height="106" rx="12" fill="#090e1a" stroke="#334155" stroke-width="1.5" />
                <circle cx="51" cy="52" r="36" fill="none" stroke="#1e293b" stroke-width="8" />
                <circle cx="51" cy="52" r="36" fill="none" stroke="{warna_utama}" stroke-width="8" stroke-dasharray="{keliling_ring:.1f}" stroke-dashoffset="{offset_ring:.1f}" stroke-linecap="round" transform="rotate(-90 51 52)" />
                <text x="51" y="50" text-anchor="middle" fill="#f8fafc" font-size="13" font-weight="800">{total_kwh_hari:.1f}</text>
                <text x="51" y="63" text-anchor="middle" fill="#94a3b8" font-size="8.5" font-weight="600">kWh/hari</text>
                <text x="51" y="99" text-anchor="middle" fill="{warna_aksen}" font-size="8.5" font-weight="700">INDEKS BEBAN</text>
            </svg>
        </div>
    </div>
    """)


# ==========================================
# 3. PANEL KONFIGURASI RUMAH & INPUT PERANGKAT (SIDEBAR)
# ==========================================
render_html("""
<div style="background: linear-gradient(135deg, #0f172a, #1e293b); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 12px 14px; color: #e2e8f0; font-size: 12.5px; line-height: 1.45; margin-bottom: 12px;">
    <div style="font-weight: 700; color: #38bdf8; margin-bottom: 4px; font-size: 13px;">📋 Panduan Pengisian Data</div>
    Silakan sesuaikan golongan daya PLN, target anggaran, jumlah unit, dan durasi nyala perangkat elektronik sesuai kondisi di rumah Anda untuk mendapatkan hasil deteksi dan prediksi biaya yang akurat.
</div>
""")

st.sidebar.header("1. Spesifikasi Listrik & Anggaran")

pilihan_golongan = st.sidebar.selectbox(
    "Golongan Daya Listrik (PLN)",
    options=list(GOLONGAN_PLN.keys()),
    index=3  # Default: 1300 VA
)
batas_va = GOLONGAN_PLN[pilihan_golongan]["va"]
TARIF_PLN = GOLONGAN_PLN[pilihan_golongan]["tarif"]

target_anggaran = st.sidebar.number_input(
    "Target Anggaran Listrik / Bulan (Rp)",
    min_value=25000,
    max_value=50000000,
    value=400000,
    step=25000,
    help="Masukkan batas maksimal biaya listrik bulanan yang Anda inginkan."
)

st.sidebar.divider()
st.sidebar.header("2. Jumlah Unit & Durasi Nyala")

PERANGKAT_LAIN = {
    "Lampu LED (Per Titik)": {"watt": 12, "normal_jam": 8.0, "std_jam": 1.5, "default_unit": 6, "max_unit": 30},
    "Kipas Angin": {"watt": 55, "normal_jam": 8.0, "std_jam": 1.8, "default_unit": 2, "max_unit": 10},
    "Sanyo (Pompa Air Otomatis)": {"watt": 200, "normal_jam": 1.5, "std_jam": 0.4, "default_unit": 1, "max_unit": 3},
    "Kulkas 2 Pintu": {"watt": 150, "normal_jam": 22.0, "std_jam": 1.0, "default_unit": 1, "max_unit": 5},
    "TV LED": {"watt": 100, "normal_jam": 5.0, "std_jam": 1.2, "default_unit": 1, "max_unit": 5},
    "Water Heater": {"watt": 1200, "normal_jam": 1.0, "std_jam": 0.3, "default_unit": 0, "max_unit": 3},
}

# Inisialisasi default state
if "unit_ac" not in st.session_state:
    st.session_state["unit_ac"] = 1
if "jam_ac" not in st.session_state:
    st.session_state["jam_ac"] = 7.5

if "unit_magicom" not in st.session_state:
    st.session_state["unit_magicom"] = 1
if "masak_magicom" not in st.session_state:
    st.session_state["masak_magicom"] = 1
if "warm_magicom" not in st.session_state:
    st.session_state["warm_magicom"] = 14.0

if "unit_cuci" not in st.session_state:
    st.session_state["unit_cuci"] = 1
if "freq_cuci" not in st.session_state:
    st.session_state["freq_cuci"] = 1

for idx_init, (nama_init, spec_init) in enumerate(PERANGKAT_LAIN.items()):
    if f"unit_{idx_init}" not in st.session_state:
        st.session_state[f"unit_{idx_init}"] = spec_init["default_unit"]
    if f"jam_{idx_init}" not in st.session_state:
        st.session_state[f"jam_{idx_init}"] = float(spec_init["normal_jam"])

# Tombol preset skenario cepat
st.sidebar.markdown("**Skenario Uji Cepat:**")
col_p1, col_p2, col_p3 = st.sidebar.columns(3)

if col_p1.button("Normal"):
    st.session_state["jam_ac"] = 7.5
    for idx_k, (_, spec_p) in enumerate(PERANGKAT_LAIN.items()):
        st.session_state[f"jam_{idx_k}"] = float(spec_p["normal_jam"])
    st.session_state["masak_magicom"] = 1
    st.session_state["warm_magicom"] = 14.0
    st.session_state["freq_cuci"] = 1

if col_p2.button("Warm 24J"):
    st.session_state["jam_ac"] = 7.5
    for idx_k, (_, spec_p) in enumerate(PERANGKAT_LAIN.items()):
        st.session_state[f"jam_{idx_k}"] = float(spec_p["normal_jam"])
    st.session_state["masak_magicom"] = 1
    st.session_state["warm_magicom"] = 24.0
    st.session_state["freq_cuci"] = 1

if col_p3.button("Boros"):
    st.session_state["jam_ac"] = 16.0
    for idx_k, (nama_p, spec_p) in enumerate(PERANGKAT_LAIN.items()):
        if nama_p == "Sanyo (Pompa Air Otomatis)":
            st.session_state[f"jam_{idx_k}"] = 6.5
        elif nama_p == "Kipas Angin":
            st.session_state[f"jam_{idx_k}"] = 20.0
        elif nama_p == "Lampu LED (Per Titik)":
            st.session_state[f"jam_{idx_k}"] = 21.0
        else:
            st.session_state[f"jam_{idx_k}"] = min(24.0, float(spec_p["normal_jam"]) * 2.0)
    st.session_state["masak_magicom"] = 3
    st.session_state["warm_magicom"] = 24.0
    st.session_state["freq_cuci"] = 5

st.sidebar.divider()

# --- INPUT KHUSUS AC (KAPASITAS 1/2 PK ATAU 1 PK LANGSUNG DI KOLOM AC) ---
st.sidebar.markdown("**Air Conditioner (AC)**")
pilihan_ac = st.sidebar.selectbox(
    "Pilih Kapasitas AC",
    options=list(OPSI_AC.keys()),
    index=0,
    label_visibility="collapsed"
)
watt_ac = OPSI_AC[pilihan_ac]
st.sidebar.caption(f"Daya dipilih: `{watt_ac} W/unit`")

ca1, ca2 = st.sidebar.columns([1, 1.7])
unit_ac = ca1.number_input("Unit AC", min_value=0, max_value=5, step=1, key="unit_ac", label_visibility="collapsed")
jam_ac_raw = ca2.slider("Jam AC", min_value=0.0, max_value=24.0, step=0.5, key="jam_ac", disabled=(unit_ac == 0), label_visibility="collapsed")
ca1.caption(f"{unit_ac} unit")
ca2.caption(f"{jam_ac_raw:.1f} jam/hari" if unit_ac > 0 else "Tidak ada")
jam_ac = jam_ac_raw if unit_ac > 0 else 0.0

st.sidebar.divider()

# --- INPUT MAGICOM ---
st.sidebar.markdown(f"**Magicom** — `Cook {WATT_MAGICOM_COOK}W / Warm {WATT_MAGICOM_WARM}W`")
cm1, cm2 = st.sidebar.columns([1, 1.7])
unit_magicom = cm1.number_input("Unit Magicom", min_value=0, max_value=5, step=1, key="unit_magicom", label_visibility="collapsed")
masak_kali = cm2.slider("Frekuensi Masak/Hari", min_value=0, max_value=4, step=1, key="masak_magicom", disabled=(unit_magicom == 0), label_visibility="collapsed")
cm1.caption(f"{unit_magicom} unit")
cm2.caption(f"Masak: {masak_kali}x/hari ({masak_kali * JAM_PER_MASAK:.2f} jam)" if unit_magicom > 0 else "Tidak ada")

warm_jam = st.sidebar.slider("Durasi Mode Warm Magicom (Jam/Hari)", min_value=0.0, max_value=24.0, step=1.0, key="warm_magicom", disabled=(unit_magicom == 0))
if unit_magicom > 0:
    if warm_jam >= 23.0:
        st.sidebar.caption("Status: Nyala 24 jam penuh (nasi belum habis sampai pagi)")
    elif warm_jam == 0.0:
        st.sidebar.caption("Status: Langsung dicabut setelah nasi matang")
    else:
        st.sidebar.caption(f"Status: Warm {warm_jam:.0f} jam (dicabut setelah nasi habis)")

st.sidebar.divider()

# --- INPUT MESIN CUCI ---
st.sidebar.markdown(f"**Mesin Cuci** — `{WATT_MESIN_CUCI} W/unit`")
cc1, cc2 = st.sidebar.columns([1, 1.7])
unit_cuci = cc1.number_input("Unit Mesin Cuci", min_value=0, max_value=3, step=1, key="unit_cuci", label_visibility="collapsed")
freq_cuci = cc2.slider("Frekuensi Cuci/Minggu", min_value=0, max_value=7, step=1, key="freq_cuci", disabled=(unit_cuci == 0), label_visibility="collapsed")
cc1.caption(f"{unit_cuci} unit")
cc2.caption(f"{freq_cuci}x/minggu (±{freq_cuci * JAM_PER_SESI_CUCI:.1f} jam/mgg)" if unit_cuci > 0 else "Tidak ada")

st.sidebar.divider()

# --- INPUT PERANGKAT HARIAN LAINNYA ---
input_unit_lain = {}
input_jam_lain = {}

for idx, (nama, spec) in enumerate(PERANGKAT_LAIN.items()):
    label_satuan = "W/lampu" if nama == "Lampu LED (Per Titik)" else "W/unit"
    st.sidebar.markdown(f"**{nama}** — `{spec['watt']} {label_satuan}`")
    c_unit, c_jam = st.sidebar.columns([1, 1.7])

    unit_val = c_unit.number_input("Unit", min_value=0, max_value=spec["max_unit"], step=1, key=f"unit_{idx}", label_visibility="collapsed")
    jam_val = c_jam.slider("Jam/Hari", min_value=0.0, max_value=24.0, step=0.5, key=f"jam_{idx}", disabled=(unit_val == 0), label_visibility="collapsed")

    c_unit.caption(f"{unit_val} lampu" if nama == "Lampu LED (Per Titik)" else f"{unit_val} unit")
    if unit_val == 0:
        c_jam.caption("Tidak ada")
    elif nama == "Sanyo (Pompa Air Otomatis)":
        c_jam.caption(f"Otomatis: {jam_val:.1f} jam/hari")
    elif nama == "Lampu LED (Per Titik)":
        c_jam.caption(f"Rata-rata: {jam_val:.1f} jam/hari")
    else:
        c_jam.caption(f"{jam_val:.1f} jam/hari")

    input_unit_lain[nama] = unit_val
    input_jam_lain[nama] = jam_val if unit_val > 0 else 0.0

st.sidebar.divider()

# --- FITUR 1: INPUT PERANGKAT TAMBAHAN KUSTOM (2 SLOT BEBAS) ---
st.sidebar.header("3. Perangkat Tambahan Kustom")
st.sidebar.caption("Isi jika ada alat lain di rumah (Setrika, PC, Dispenser, dll). Biarkan 0 unit jika tidak ada.")

# Slot Kustom 1
raw_nama_c1 = st.sidebar.text_input("Nama Alat Kustom 1", value="Setrika Listrik", key="nama_c1")
nama_c1 = raw_nama_c1.strip() or "Perangkat Kustom 1"
cw1, cu1 = st.sidebar.columns(2)
watt_c1 = cw1.number_input("Daya (Watt)", min_value=5, max_value=5000, value=350, step=5, key="watt_c1")
unit_c1 = cu1.number_input("Jumlah Unit", min_value=0, max_value=10, value=0, step=1, key="unit_c1")
jam_c1_raw = st.sidebar.slider("Durasi Nyala Alat 1 (Jam/Hari)", min_value=0.0, max_value=24.0, value=1.0, step=0.5, disabled=(unit_c1 == 0), key="jam_c1")
jam_c1 = jam_c1_raw if unit_c1 > 0 else 0.0

st.sidebar.markdown("---")

# Slot Kustom 2
raw_nama_c2 = st.sidebar.text_input("Nama Alat Kustom 2", value="PC / Komputer", key="nama_c2")
nama_c2 = raw_nama_c2.strip() or "Perangkat Kustom 2"
if nama_c2 == nama_c1:
    nama_c2 = f"{nama_c2} (2)"
cw2, cu2 = st.sidebar.columns(2)
watt_c2 = cw2.number_input("Daya (Watt) ", min_value=5, max_value=5000, value=250, step=5, key="watt_c2")
unit_c2 = cu2.number_input("Jumlah Unit ", min_value=0, max_value=10, value=0, step=1, key="unit_c2")
jam_c2_raw = st.sidebar.slider("Durasi Nyala Alat 2 (Jam/Hari)", min_value=0.0, max_value=24.0, value=4.0, step=0.5, disabled=(unit_c2 == 0), key="jam_c2")
jam_c2 = jam_c2_raw if unit_c2 > 0 else 0.0


# ==========================================
# 4. GABUNGKAN KONFIGURASI & LATIH MODEL AI
# ==========================================
PERANGKAT_UMUM = {
    pilihan_ac: {"watt": watt_ac, "normal_jam": 7.5, "std_jam": 1.2, "is_custom": False}
}
for k, v in PERANGKAT_LAIN.items():
    PERANGKAT_UMUM[k] = {"watt": v["watt"], "normal_jam": v["normal_jam"], "std_jam": v["std_jam"], "is_custom": False}

# Masukkan slot kustom ke struktur model agar terdeteksi AI secara otomatis
PERANGKAT_UMUM[nama_c1] = {"watt": watt_c1, "normal_jam": 2.5, "std_jam": 1.0, "is_custom": True}
PERANGKAT_UMUM[nama_c2] = {"watt": watt_c2, "normal_jam": 4.0, "std_jam": 1.5, "is_custom": True}

input_unit_umum = {pilihan_ac: unit_ac}
input_unit_umum.update(input_unit_lain)
input_unit_umum[nama_c1] = unit_c1
input_unit_umum[nama_c2] = unit_c2

input_jam_umum = {pilihan_ac: jam_ac}
input_jam_umum.update(input_jam_lain)
input_jam_umum[nama_c1] = jam_c1
input_jam_umum[nama_c2] = jam_c2

konfig_umum_tuple = tuple(
    (nama, spec["watt"], input_unit_umum[nama], spec["normal_jam"], spec["std_jam"])
    for nama, spec in PERANGKAT_UMUM.items()
)

df_hist, hist_kwh, hist_tagihan, model_anomali, model_regresi = latih_model_ai(
    TARIF_PLN, konfig_umum_tuple, unit_magicom, unit_cuci
)

jam_cook_aktual = (masak_kali * JAM_PER_MASAK) if unit_magicom > 0 else 0.0
jam_warm_aktual = warm_jam if unit_magicom > 0 else 0.0
jam_cuci_harian = ((freq_cuci * JAM_PER_SESI_CUCI) / 7.0) if unit_cuci > 0 else 0.0

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
    skor_anomali = 0.1
else:
    status_pred = model_anomali.predict(fitur_input)[0]
    skor_anomali = model_anomali.decision_function(fitur_input)[0]

is_boros = (status_pred == -1)


# ==========================================
# 5. PERHITUNGAN ENERGI, DIAGNOSIS & SOLUSI
# ==========================================
DETAIL_DIAGNOSIS = {
    pilihan_ac: {
        "penyebab": "Kompresor AC bekerja terlalu lama dalam sehari atau suhu ruangan disetel terlalu rendah sehingga mesin pendingin terus menarik daya penuh tanpa jeda.",
        "solusi": "Atur suhu ideal di <b>24°C–25°C</b>, aktifkan fitur <i>Timer/Sleep</i> saat dini hari, pastikan pintu dan jendela tertutup rapat, serta bersihkan filter AC secara rutin."
    },
    "Lampu LED (Per Titik)": {
        "penyebab": "Banyak titik lampu dibiarkan menyala pada siang hari atau tetap hidup di ruangan yang sedang kosong.",
        "solusi": "Matikan lampu ruangan yang tidak dipakai, manfaatkan cahaya matahari lewat jendela di siang hari, dan gunakan sensor otomatis untuk lampu teras."
    },
    "Kipas Angin": {
        "penyebab": "Kipas angin dibiarkan berputar terus-menerus meski tidak ada orang di ruangan, atau debu menumpuk pada baling-baling sehingga putaran motor menjadi berat.",
        "solusi": "Matikan kipas saat meninggalkan ruangan, gunakan tingkat kecepatan rendah–sedang, dan bersihkan debu baling-baling secara berkala."
    },
    "Sanyo (Pompa Air Otomatis)": {
        "penyebab": "Pompa air otomatis terlalu sering hidup-mati (<i>short cycling</i>) akibat kebocoran halus pada pipa/keran air atau kerusakan saklar otomatis/pelampung toren (<i>radar</i>).",
        "solusi": "Periksa seluruh keran air dan sambungan pipa dari kebocoran menetes, serta cek kondisi pelampung otomatis toren agar pompa hanya menyala saat air benar-benar habis."
    },
    "Kulkas 2 Pintu": {
        "penyebab": "Pintu kulkas terlalu sering dibuka-tutup, karet pintu mulai renggang, atau makanan panas langsung dimasukkan sehingga kompresor bekerja ekstra.",
        "solusi": "Tunggu makanan panas hingga bersuhu ruang sebelum dimasukkan, pastikan pintu kulkas tertutup rapat, dan beri jarak minimal 10 cm dari dinding belakang."
    },
    "TV LED": {
        "penyebab": "TV dibiarkan menyala seharian tanpa ditonton atau terus berada dalam mode <i>standby</i> saat malam hari.",
        "solusi": "Matikan TV melalui tombol daya utama atau cabut steker setelah selesai menonton, dan aktifkan fitur <i>Auto Sleep</i>."
    },
    "Water Heater": {
        "penyebab": "Pemanas air dibiarkan menyala terus-menerus sepanjang hari untuk menjaga suhu air tetap panas meski tidak sedang dipakai mandi.",
        "solusi": "Nyalakan pemanas air hanya <b>15–20 menit sebelum mandi</b>, lalu matikan kembali saklarnya setelah selesai digunakan."
    },
    "Magicom (Cook + Warm)": {
        "penyebab": "Frekuensi memasak nasi (mode <i>Cook</i> 350 W) dilakukan berkali-kali dalam sehari sehingga menarik lonjakan daya besar berulang.",
        "solusi": "Masak nasi sekaligus 1 kali di pagi hari untuk kebutuhan seharian, dan segera cabut steker Magicom bila nasi sudah habis di malam hari."
    },
    "Mesin Cuci (Mingguan)": {
        "penyebab": "Mencuci pakaian sedikit demi sedikit hampir setiap hari membuat motor penggerak dan pompa air bekerja jauh lebih sering.",
        "solusi": "Kumpulkan pakaian kotor dan cuci sekaligus 1–2 kali seminggu sesuai kapasitas maksimal tabung mesin cuci."
    }
}

DEFAULT_DIAG_KUSTOM = {
    "penyebab": "Durasi pemakaian perangkat kustom ini berlangsung terlalu lama dalam sehari sehingga akumulasi kWh hariannya melonjak.",
    "solusi": "Batasi penggunaan perangkat hanya saat benar-benar dibutuhkan dan matikan/cabut dari stopkontak setelah selesai dipakai."
}

rincian_list = []
total_kwh_hari = 0.0
total_watt_terpasang = 0
alat_aktif_daya = []
diagnosis_boros_list = []
potensi_hemat_rp_total = 0.0

# 1. Hitung Perangkat Umum + Kustom
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

    if unit > 0:
        z_score = (jam - spec["normal_jam"]) / spec["std_jam"]
        if z_score > 2.2:
            selisih_biaya = ((jam - spec["normal_jam"]) * daya_total_alat / 1000.0) * 30.0 * TARIF_PLN
            potensi_hemat_rp_total += selisih_biaya
            satuan_txt = "lampu" if nama == "Lampu LED (Per Titik)" else "unit"
            info_diag = DETAIL_DIAGNOSIS.get(nama, DEFAULT_DIAG_KUSTOM)

            diagnosis_boros_list.append({
                "judul": f"{nama} ({unit} {satuan_txt}) — Menyala {jam:.1f} jam/hari (Normalnya {spec['normal_jam']} jam/hari)",
                "dampak": f"Menambah beban tagihan sekitar <b>+Rp {selisih_biaya:,.0f}/bulan</b> (atau <b>+Rp {selisih_biaya * 12:,.0f}/tahun</b>).",
                "penyebab": info_diag["penyebab"],
                "solusi": info_diag["solusi"]
            })

    rincian_list.append({
        "Perangkat Rumah": nama,
        "Jumlah Unit": unit,
        "Daya Operasi": f"{spec['watt']} W",
        "Beban Puncak (W)": daya_total_alat,
        "Durasi Pakai": f"{jam:.1f} jam/hari",
        "Energi Harian (kWh)": round(kwh_hari, 2),
        "Estimasi Biaya/Bulan (Rp)": round(biaya_bulan, 0)
    })

# 2. Hitung Magicom
daya_puncak_magicom = unit_magicom * WATT_MAGICOM_COOK
total_watt_terpasang += daya_puncak_magicom
if unit_magicom > 0:
    alat_aktif_daya.append(("Magicom (mode Cook)", daya_puncak_magicom))

kwh_cook = (unit_magicom * WATT_MAGICOM_COOK * jam_cook_aktual) / 1000.0
kwh_warm = (unit_magicom * WATT_MAGICOM_WARM * jam_warm_aktual) / 1000.0
kwh_magicom_total = kwh_cook + kwh_warm
biaya_magicom_bulan = (kwh_magicom_total * 30.0) * TARIF_PLN
total_kwh_hari += kwh_magicom_total

if unit_magicom > 0 and masak_kali >= 3:
    hemat_masak = ((masak_kali - 1) * JAM_PER_MASAK * unit_magicom * WATT_MAGICOM_COOK / 1000.0) * 30.0 * TARIF_PLN
    potensi_hemat_rp_total += hemat_masak
    info_diag = DETAIL_DIAGNOSIS["Magicom (Cook + Warm)"]
    diagnosis_boros_list.append({
        "judul": f"Magicom ({unit_magicom} unit) — Memasak nasi {masak_kali}x/hari (Normalnya 1x/hari)",
        "dampak": f"Menambah beban tagihan sekitar <b>+Rp {hemat_masak:,.0f}/bulan</b> (atau <b>+Rp {hemat_masak * 12:,.0f}/tahun</b>).",
        "penyebab": info_diag["penyebab"],
        "solusi": info_diag["solusi"]
    })

rincian_list.append({
    "Perangkat Rumah": "Magicom (Cook + Warm)",
    "Jumlah Unit": unit_magicom,
    "Daya Operasi": f"{WATT_MAGICOM_COOK}W / {WATT_MAGICOM_WARM}W",
    "Beban Puncak (W)": daya_puncak_magicom,
    "Durasi Pakai": f"Cook {masak_kali}x + Warm {jam_warm_aktual:.0f}j",
    "Energi Harian (kWh)": round(kwh_magicom_total, 2),
    "Estimasi Biaya/Bulan (Rp)": round(biaya_magicom_bulan, 0)
})

# 3. Hitung Mesin Cuci
daya_puncak_cuci = unit_cuci * WATT_MESIN_CUCI
total_watt_terpasang += daya_puncak_cuci
if unit_cuci > 0:
    alat_aktif_daya.append(("Mesin Cuci", daya_puncak_cuci))

kwh_cuci_harian = (daya_puncak_cuci * jam_cuci_harian) / 1000.0
biaya_cuci_bulan = (kwh_cuci_harian * 30.0) * TARIF_PLN
total_kwh_hari += kwh_cuci_harian

if unit_cuci > 0 and freq_cuci >= 4:
    hemat_cuci = (((freq_cuci - 1) * JAM_PER_SESI_CUCI / 7.0) * daya_puncak_cuci / 1000.0) * 30.0 * TARIF_PLN
    potensi_hemat_rp_total += hemat_cuci
    info_diag = DETAIL_DIAGNOSIS["Mesin Cuci (Mingguan)"]
    diagnosis_boros_list.append({
        "judul": f"Mesin Cuci ({unit_cuci} unit) — Dipakai {freq_cuci}x/minggu (Normalnya 1x/minggu)",
        "dampak": f"Menambah beban tagihan sekitar <b>+Rp {hemat_cuci:,.0f}/bulan</b> (atau <b>+Rp {hemat_cuci * 12:,.0f}/tahun</b>).",
        "penyebab": info_diag["penyebab"],
        "solusi": info_diag["solusi"]
    })

rincian_list.append({
    "Perangkat Rumah": "Mesin Cuci (Mingguan)",
    "Jumlah Unit": unit_cuci,
    "Daya Operasi": f"{WATT_MESIN_CUCI} W",
    "Beban Puncak (W)": daya_puncak_cuci,
    "Durasi Pakai": f"{freq_cuci}x/minggu ({jam_cuci_harian:.2f} j/hr)",
    "Energi Harian (kWh)": round(kwh_cuci_harian, 2),
    "Estimasi Biaya/Bulan (Rp)": round(biaya_cuci_bulan, 0)
})

# Pengaman: jika AI mendeteksi anomali gabungan namun belum melewati Z-score tunggal
if is_boros and len(diagnosis_boros_list) == 0:
    for nama, spec in PERANGKAT_UMUM.items():
        if input_unit_umum[nama] > 0 and input_jam_umum[nama] > spec["normal_jam"] * 1.25:
            selisih_biaya = ((input_jam_umum[nama] - spec["normal_jam"]) * input_unit_umum[nama] * spec["watt"] / 1000.0) * 30.0 * TARIF_PLN
            potensi_hemat_rp_total += selisih_biaya
            info_diag = DETAIL_DIAGNOSIS.get(nama, DEFAULT_DIAG_KUSTOM)
            diagnosis_boros_list.append({
                "judul": f"{nama} ({input_unit_umum[nama]} unit) — Menyala {input_jam_umum[nama]:.1f} jam/hari (Normalnya {spec['normal_jam']} jam/hari)",
                "dampak": f"Berkontribusi menaikkan tagihan sekitar <b>+Rp {selisih_biaya:,.0f}/bulan</b> (atau <b>+Rp {selisih_biaya * 12:,.0f}/tahun</b>).",
                "penyebab": info_diag["penyebab"],
                "solusi": info_diag["solusi"]
            })

df_rincian = pd.DataFrame(rincian_list)

total_biaya_mentah = df_rincian["Estimasi Biaya/Bulan (Rp)"].sum()
if total_biaya_mentah > 0:
    df_rincian["Porsi Beban (%)"] = (df_rincian["Estimasi Biaya/Bulan (Rp)"] / total_biaya_mentah * 100).round(1)
else:
    df_rincian["Porsi Beban (%)"] = 0.0

if total_kwh_hari == 0:
    prediksi_tagihan_rp = 0.0
else:
    prediksi_tagihan_rp = float(model_regresi.predict([[total_kwh_hari]])[0])
    prediksi_tagihan_rp = max(0.0, prediksi_tagihan_rp)

total_kwh_bulan = total_kwh_hari * 30.0
potensi_hemat_tahun = potensi_hemat_rp_total * 12.0


# ==========================================
# 6. TAMPILAN DASBOR UTAMA (PADAT & RESPONSIF)
# ==========================================
st.markdown("### ⚡ Smart Home AI: Monitoring Energi & Deteksi Keborosan")

render_animasi_dashboard(is_boros, skor_anomali, total_kwh_hari, total_watt_terpasang, batas_va, pilihan_golongan, TARIF_PLN)

# Peringatan Kapasitas Daya MCB Dinamis (Hanya menyebut alat yang benar-benar ada/aktif)
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
        f"**Peringatan Beban Puncak MCB ({batas_va} VA):** Total beban puncak dari seluruh perangkat aktif di rumah Anda mencapai "
        f"**{total_watt_terpasang:,} Watt** (melebihi kapasitas **{batas_va} VA**). {kalimat_hindari}"
    )

# 4 Kartu KPI Padat & Responsif
render_html(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Beban Terpasang</div>
        <div class="kpi-value">{total_watt_terpasang:,} W</div>
        <div class="kpi-sub">Kapasitas PLN: {batas_va} VA</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Konsumsi Harian</div>
        <div class="kpi-value">{total_kwh_hari:.2f} kWh</div>
        <div class="kpi-sub">Rata-rata per 24 jam</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Estimasi Bulanan</div>
        <div class="kpi-value">{total_kwh_bulan:.1f} kWh</div>
        <div class="kpi-sub">Akumulasi 30 hari</div>
    </div>
    <div class="kpi-card" style="border-color: {'#ef4444' if is_boros else '#10b981'};">
        <div class="kpi-label">Prediksi Tagihan AI</div>
        <div class="kpi-value" style="color: {'#f87171' if is_boros else '#34d399'};">Rp {prediksi_tagihan_rp:,.0f}</div>
        <div class="kpi-sub">Tarif Rp {TARIF_PLN:,.2f}/kWh</div>
    </div>
</div>
""")

# ==========================================
# 6B. FITUR 2 & 3: TARGET ANGGARAN BULANAN & KALKULATOR TOKEN PRABAYAR
# ==========================================
col_budget, col_token = st.columns(2)

with col_budget:
    persen_anggaran = (prediksi_tagihan_rp / max(target_anggaran, 1)) * 100.0
    lebar_bar = min(persen_anggaran, 100.0)
    selisih_anggaran = target_anggaran - prediksi_tagihan_rp

    if selisih_anggaran >= 0:
        warna_bar_budget = "#10b981"
        status_budget_txt = f"<span style='color:#34d399; font-weight:700;'>AMAN</span> — Masih tersisa ruang anggaran <b>Rp {selisih_anggaran:,.0f}/bulan</b> dari target maksimal."
    else:
        warna_bar_budget = "#ef4444"
        status_budget_txt = f"<span style='color:#f87171; font-weight:700;'>MELEBIHI TARGET</span> — Membengkak <b>+Rp {abs(selisih_anggaran):,.0f}/bulan</b> (atau <b>+Rp {abs(selisih_anggaran) * 12:,.0f}/tahun</b>)."

    render_html(f"""
    <div class="feature-card">
        <div class="feature-title">🎯 Monitoring Target Anggaran Bulanan</div>
        <div style="display:flex; justify-content:space-between; font-size:13px; color:#cbd5e1;">
            <span>Prediksi: <b>Rp {prediksi_tagihan_rp:,.0f}</b></span>
            <span>Target: <b>Rp {target_anggaran:,.0f}</b> ({persen_anggaran:.1f}%)</span>
        </div>
        <div class="progress-track">
            <div style="width:{lebar_bar:.1f}%; height:100%; background:{warna_bar_budget}; border-radius:999px;"></div>
        </div>
        <div style="font-size:12.5px; color:#e2e8f0; margin-top:6px; line-height:1.45;">
            {status_budget_txt}
        </div>
    </div>
    """)

with col_token:
    tk1, tk2 = st.columns([1.4, 1])
    nominal_token = tk1.selectbox(
        "Nominal Token Prabayar",
        options=[50000, 100000, 200000, 500000, 1000000],
        index=1,
        format_func=lambda x: f"Token Rp {x:,.0f}"
    )
    persen_ppj = tk2.slider("PPJ Daerah (%)", min_value=3, max_value=10, value=5, step=1)

    rupiah_bersih_token = nominal_token * (1.0 - (persen_ppj / 100.0))
    kwh_didapat = rupiah_bersih_token / TARIF_PLN

    if total_kwh_hari > 0:
        estimasi_hari_token = kwh_didapat / total_kwh_hari
        teks_bertahan = f"<b style='color:#38bdf8; font-size:16px;'>±{estimasi_hari_token:.1f} Hari</b>"
    else:
        teks_bertahan = "<b style='color:#38bdf8;'>Tidak terpakai (0 kWh/hari)</b>"

    render_html(f"""
    <div class="feature-card" style="padding: 12px 16px;">
        <div style="font-size:12.5px; color:#cbd5e1; line-height:1.5;">
            • Listrik bersih didapat (setelah PPJ {persen_ppj}%): <b>{kwh_didapat:.2f} kWh</b><br>
            • Estimasi ketahanan pulsa token: {teks_bertahan}
        </div>
    </div>
    """)


# ==========================================
# 7. PANEL DIAGNOSIS PENYEBAB, SOLUSI & PROYEKSI HEMAT TAHUNAN
# ==========================================
if is_boros or len(diagnosis_boros_list) > 0:
    st.markdown("#### 🚨 Diagnosis Perangkat Penyebab Keborosan & Cara Mengatasinya")

    for item in diagnosis_boros_list:
        render_html(f"""
        <div class="diag-card">
            <div class="diag-header">⚠️ {item['judul']}</div>
            <div class="diag-row">• <b>Dampak Biaya:</b> {item['dampak']}</div>
            <div class="diag-row">• <b>Penyebab Terdeteksi:</b> {item['penyebab']}</div>
            <div class="diag-row" style="color: #86efac;">• <b>Cara Mengatasi:</b> {item['solusi']}</div>
        </div>
        """)

    if potensi_hemat_rp_total > 0:
        st.success(
            f"**Total Potensi Penghematan (Bulanan & Tahunan):** Dengan menerapkan langkah perbaikan di atas dan mengembalikan durasi alat ke batas normal, "
            f"Anda berpotensi menghemat biaya listrik sekitar **Rp {potensi_hemat_rp_total:,.0f}/bulan** atau **Rp {potensi_hemat_tahun:,.0f}/tahun**!"
        )
else:
    df_aktif_sort = df_rincian[df_rincian["Jumlah Unit"] > 0].sort_values(by="Energi Harian (kWh)", ascending=False)
    alat_terbesar = df_aktif_sort.iloc[0]["Perangkat Rumah"] if len(df_aktif_sort) > 0 else "-"
    porsi_terbesar = df_aktif_sort.iloc[0]["Porsi Beban (%)"] if len(df_aktif_sort) > 0 else 0.0
    solusi_terbesar = DETAIL_DIAGNOSIS.get(alat_terbesar, DEFAULT_DIAG_KUSTOM)["solusi"]

    render_html(f"""
    <div class="explain-box" style="border-left-color: #10b981;">
        <div class="explain-title" style="color: #34d399;">✅ Status Pemakaian Efisien — Saran Optimasi Tambahan</div>
        <div class="explain-item">• Sistem AI tidak menemukan pemakaian perangkat yang boros atau menyimpang dari kebiasaan normal.</div>
        <div class="explain-item">• Kontributor daya terbesar di rumah Anda saat ini adalah <b>{alat_terbesar}</b> (menyumbang <b>{porsi_terbesar:.1f}%</b> dari total tagihan).</div>
        <div class="explain-item">• <b>Tips Menghemat Lebih Lanjut:</b> {solusi_terbesar}</div>
    </div>
    """)

if unit_magicom > 0 and jam_warm_aktual >= 23.0:
    biaya_warm_24 = kwh_warm * 30.0 * TARIF_PLN
    st.info(
        f"**Catatan Siklus Magicom 24 Jam:** Meskipun Magicom menyala 24 jam penuh, daya pada mode *Warm* hanya **{WATT_MAGICOM_WARM} W** "
        f"(sekitar **Rp {biaya_warm_24:,.0f}/bulan**). Namun, apabila nasi sudah habis pada malam hari, mencabut steker Magicom akan langsung menghentikan beban *Warm* tersebut."
    )

st.divider()


# ==========================================
# 8. VISUALISASI GRAFIK ATAS - BAWAH + PENJELASAN DETAIL
# ==========================================
st.subheader("📊 Visualisasi Analisis Konsumsi & Prediksi Regresi")

# --- GRAFIK 1 (ATAS) ---
fig1, ax1 = plt.subplots(figsize=(10.5, 3.5))
df_aktif = df_rincian[df_rincian["Jumlah Unit"] > 0]

if len(df_aktif) > 0:
    nama_alat = [f"{r['Perangkat Rumah']} ({r['Jumlah Unit']}x)" for _, r in df_aktif.iterrows()]
    kwh_aktual = df_aktif["Energi Harian (kWh)"].to_numpy()

    kwh_normal_list = []
    for _, r in df_aktif.iterrows():
        nama_row = r["Perangkat Rumah"]
        u_row = r["Jumlah Unit"]
        if nama_row == "Magicom (Cook + Warm)":
            norm_val = u_row * ((JAM_PER_MASAK * WATT_MAGICOM_COOK) + (16.5 * WATT_MAGICOM_WARM)) / 1000.0
        elif nama_row == "Mesin Cuci (Mingguan)":
            norm_val = u_row * (((1.0 * JAM_PER_SESI_CUCI) / 7.0) * WATT_MESIN_CUCI) / 1000.0
        else:
            norm_val = (PERANGKAT_UMUM[nama_row]["normal_jam"] * PERANGKAT_UMUM[nama_row]["watt"] * u_row) / 1000.0
        kwh_normal_list.append(norm_val)

    kwh_normal = np.array(kwh_normal_list)
    x = np.arange(len(nama_alat))
    lebar = 0.35

    ax1.bar(x - lebar / 2, kwh_normal, width=lebar, label="Batas Normal (kWh/hari)", color="#64748b")
    warna_bar = ["#ef4444" if akt > norm * 1.35 else "#10b981" for akt, norm in zip(kwh_aktual, kwh_normal)]
    bars_akt = ax1.bar(x + lebar / 2, kwh_aktual, width=lebar, label="Pemakaian Aktual (kWh/hari)", color=warna_bar)

    for bar in bars_akt:
        tinggi = bar.get_height()
        ax1.annotate(
            f"{tinggi:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, tinggi),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#1e293b"
        )

    ax1.set_title("1. Perbandingan Konsumsi Energi Harian per Perangkat (kWh)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nama_alat, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Energi (kWh / hari)", fontsize=9)
    ax1.set_ylim(0, max(max(kwh_aktual), max(kwh_normal)) * 1.22 + 0.2)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
else:
    ax1.text(0.5, 0.5, "Tidak ada unit perangkat yang aktif", ha="center", va="center")

plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

if len(df_aktif) > 0:
    baris_max_kwh = df_aktif.sort_values(by="Energi Harian (kWh)", ascending=False).iloc[0]
    nama_max_g1 = baris_max_kwh["Perangkat Rumah"]
    nilai_max_g1 = baris_max_kwh["Energi Harian (kWh)"]
    alat_merah = [
        r["Perangkat Rumah"]
        for (_, r), akt, norm in zip(df_aktif.iterrows(), kwh_aktual, kwh_normal)
        if akt > norm * 1.35
    ]
    ket_merah = (
        f"Terdapat batang berwarna <b>merah</b> pada <b>{', '.join(alat_merah)}</b> karena durasi nyalanya melampaui batas kewajaran harian."
        if len(alat_merah) > 0 else
        "Seluruh batang pemakaian saat ini berwarna <b>hijau</b>, yang berarti tidak ada alat yang melebihi batas wajar hariannya."
    )

    render_html(f"""
    <div class="explain-box">
        <div class="explain-title">📖 Cara Membaca Grafik 1 (Perbandingan Konsumsi Harian per Alat)</div>
        <div class="explain-item">• <b>Apa arti warna batang grafik?</b> Batang <b>abu-abu</b> menunjukkan batas pemakaian normal rumah tangga, batang <b>hijau</b> berarti pemakaian Anda masih aman/efisien, sedangkan batang <b>merah</b> menandakan pemakaian boros yang jauh melebihi batas normal.</div>
        <div class="explain-item">• <b>Mengapa hal ini terjadi pada simulasi Anda?</b> Nilai konsumsi (kWh) dihitung dari <i>Daya Alat (Watt) × Lama Menyala (Jam) ÷ 1.000</i>. Saat ini, konsumsi energi harian tertinggi berasal dari <b>{nama_max_g1}</b> sebesar <b>{nilai_max_g1:.2f} kWh/hari</b>. {ket_merah}</div>
    </div>
    """)

# --- GRAFIK 2 (BAWAH) ---
fig2, ax2 = plt.subplots(figsize=(10.5, 3.5))
ax2.scatter(hist_kwh, hist_tagihan / 1000.0, color="#38bdf8", alpha=0.65, s=36, label="Dataset Historis Normal (120 Hari)")

x_max_plot = max(hist_kwh.max(), total_kwh_hari, 1.0) * 1.06
x_line = np.linspace(0, x_max_plot, 100)
y_line = model_regresi.predict(x_line.reshape(-1, 1)) / 1000.0
ax2.plot(x_line, y_line, color="#f59e0b", linewidth=2.2, label="Garis Prediksi Linear Regression")

ax2.scatter(
    [total_kwh_hari],
    [prediksi_tagihan_rp / 1000.0],
    color="#ef4444" if is_boros else "#10b981",
    s=180,
    marker="*",
    edgecolors="#0f172a",
    linewidths=1.0,
    zorder=5,
    label=f"Posisi Saat Ini ({total_kwh_hari:.2f} kWh → Rp {prediksi_tagihan_rp:,.0f})"
)

ax2.set_title(f"2. Proyeksi Tagihan Listrik Bulanan — {pilihan_golongan.split('(')[0].strip()} (Linear Regression)", fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Total Konsumsi Energi Harian (kWh / hari)", fontsize=9)
ax2.set_ylabel("Estimasi Tagihan (Ribu Rp)", fontsize=9)
ax2.legend(fontsize=8.5, loc="upper left")
ax2.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

rata_kwh_hist = float(hist_kwh.mean())
posisi_bintang = (
    f"bergeser jauh ke sebelah kanan kumpulan titik biru (di atas rata-rata normal {rata_kwh_hist:.2f} kWh/hari) sehingga memicu kenaikan tagihan"
    if total_kwh_hari > rata_kwh_hist * 1.25 else
    f"berada tepat di dalam area kumpulan titik biru normal (sekitar {rata_kwh_hist:.2f} kWh/hari)"
)
render_html(f"""
<div class="explain-box">
    <div class="explain-title">📖 Cara Membaca Grafik 2 (Proyeksi Tagihan Bulanan dengan AI Linear Regression)</div>
    <div class="explain-item">• <b>Apa arti titik biru, garis oranye, dan ikon bintang?</b> Kumpulan <b>titik biru</b> adalah riwayat pemakaian listrik normal selama 120 hari terakhir. <b>Garis oranye miring</b> adalah rumus kecerdasan buatan (<i>Linear Regression</i>) yang memprediksi kenaikan rupiah seiring bertambahnya kWh. Sedangkan <b>ikon bintang</b> menunjukkan posisi rumah Anda saat ini.</div>
    <div class="explain-item">• <b>Mengapa prediksi tagihan Anda menjadi Rp {prediksi_tagihan_rp:,.0f}/bulan?</b> Karena total pemakaian seluruh alat Anda mencapai <b>{total_kwh_hari:.2f} kWh/hari</b> (atau <b>{total_kwh_bulan:.1f} kWh/bulan</b>) dikalikan tarif PLN golongan <b>{pilihan_golongan.split('(')[0].strip()}</b> sebesar <b>Rp {TARIF_PLN:,.2f}/kWh</b>. Pada grafik, posisi bintang Anda {posisi_bintang}.</div>
</div>
""")

st.divider()


# ==========================================
# 9. TABEL RINCIAN PERHITUNGAN + UNDUH CSV + PENJELASAN DETAIL
# ==========================================
st.subheader("📋 Rincian Perhitungan Daya, Biaya & Porsi Beban")
st.dataframe(
    df_rincian,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Estimasi Biaya/Bulan (Rp)": st.column_config.NumberColumn(format="Rp %d"),
        "Porsi Beban (%)": st.column_config.ProgressColumn(
            min_value=0,
            max_value=100,
            format="%.1f%%"
        ),
    }
)

# --- FITUR 4: TOMBOL UNDUH LAPORAN AUDIT ENERGI (CSV) ---
df_export = df_rincian.copy()
baris_total = pd.DataFrame([{
    "Perangkat Rumah": f"TOTAL KESELURUHAN ({pilihan_golongan.split('(')[0].strip()})",
    "Jumlah Unit": int(df_rincian["Jumlah Unit"].sum()),
    "Daya Operasi": "-",
    "Beban Puncak (W)": int(total_watt_terpasang),
    "Durasi Pakai": "30 Hari / Bulan",
    "Energi Harian (kWh)": round(total_kwh_hari, 2),
    "Estimasi Biaya/Bulan (Rp)": round(total_biaya_mentah, 0),
    "Porsi Beban (%)": 100.0 if total_biaya_mentah > 0 else 0.0
}])
df_export = pd.concat([df_export, baris_total], ignore_index=True)
csv_bytes = ("\ufeff" + df_export.to_csv(index=False)).encode("utf-8")

st.download_button(
    label="📥 Unduh Laporan Audit Energi (Excel / CSV)",
    data=csv_bytes,
    file_name="Laporan_Audit_Energi_SmartHome.csv",
    mime="text/csv",
    help="Simpan tabel rincian daya dan estimasi tagihan ini ke file CSV untuk dibuka di Microsoft Excel."
)

# Penjelasan Detail Tabel Rincian
df_top2 = df_rincian[df_rincian["Jumlah Unit"] > 0].sort_values(by="Porsi Beban (%)", ascending=False).head(2)
if len(df_top2) >= 2:
    p1 = df_top2.iloc[0]
    p2 = df_top2.iloc[1]
    ringkasan_porsi = (
        f"Dua penyumbang tagihan terbesar di rumah Anda adalah <b>{p1['Perangkat Rumah']}</b> "
        f"(<b>{p1['Porsi Beban (%)']:.1f}%</b> atau sekitar <b>Rp {p1['Estimasi Biaya/Bulan (Rp)']:,.0f}/bulan</b>) "
        f"dan <b>{p2['Perangkat Rumah']}</b> (<b>{p2['Porsi Beban (%)']:.1f}%</b> atau sekitar <b>Rp {p2['Estimasi Biaya/Bulan (Rp)']:,.0f}/bulan</b>)."
    )
elif len(df_top2) == 1:
    p1 = df_top2.iloc[0]
    ringkasan_porsi = (
        f"Satu-satunya penyumbang beban listrik aktif saat ini adalah <b>{p1['Perangkat Rumah']}</b> "
        f"(<b>{p1['Porsi Beban (%)']:.1f}%</b> atau sekitar <b>Rp {p1['Estimasi Biaya/Bulan (Rp)']:,.0f}/bulan</b>)."
    )
else:
    ringkasan_porsi = "Belum ada perangkat elektronik yang aktif (jumlah unit masih 0)."

render_html(f"""
<div class="explain-box">
    <div class="explain-title">📖 Panduan Membaca Tabel & Kesimpulan Rincian Beban Listrik</div>
    <div class="explain-item">• <b>Arti Kolom Penting:</b> <i>Beban Puncak (W)</i> adalah daya listrik yang ditarik jika seluruh unit alat tersebut menyala bersamaan. <i>Energi Harian (kWh)</i> adalah jumlah listrik yang terpakai dalam sehari. <i>Porsi Beban (%)</i> menunjukkan seberapa besar persentase biaya alat tersebut terhadap total tagihan bulanan Anda.</div>
    <div class="explain-item">• <b>Mengapa alat berdaya kecil bisa mahal, atau sebaliknya?</b> Biaya bulanan tidak hanya ditentukan oleh besarnya Watt, tetapi juga <b>lama waktu menyala</b>. Alat berdaya kecil seperti Kulkas (150 W) tetap memakan biaya cukup besar karena menyala hampir 24 jam penuh, sedangkan alat berdaya besar seperti AC atau Water Heater langsung melonjak biayanya walau hanya dipakai beberapa jam.</div>
    <div class="explain-item">• <b>Kesimpulan Rumah Anda:</b> {ringkasan_porsi}</div>
</div>
""")


# ==========================================
# 10. FITUR 5: KOTAK LIPAT METODOLOGI & RUMUS TEKNIS (UNTUK PRESENTASI / DOSEN)
# ==========================================
with st.expander("🔬 Metodologi Teknis, Rumus Fisika & Arsitektur Model AI (Klik untuk Membuka)"):
    st.markdown("""
    **1. Konversi Daya Listrik ke Energi (kWh) dan Biaya Tagihan PLN**
    Setiap perangkat elektronik dengan daya aktif $P$ (Watt) yang beroperasi selama durasi $t$ (jam/hari) mengonsumsi energi harian sebesar:
    """)
    st.latex(r"E_{\text{harian}} = \frac{P \times t}{1.000} \quad (\text{kWh/hari})")

    st.markdown("""
    Akumulasi biaya pemakaian bulanan (30 hari) dihitung berdasarkan tarif dasar listrik golongan PLN ($T_{\\text{PLN}}$):
    """)
    st.latex(r"\text{Biaya}_{\text{bulanan}} = \left(\sum E_{\text{harian}}\right) \times 30 \times T_{\text{PLN}}")

    st.markdown("""
    **2. Perhitungan Ketahanan Token Listrik Prabayar**
    Pembelian token prabayar senilai $N$ rupiah dikenakan potongan Pajak Penerangan Jalan (PPJ) daerah sebesar $p\\%$, sehingga energi bersih ($E_{\\text{token}}$) dan estimasi hari bertahan ($D_{\\text{token}}$) dirumuskan sebagai:
    """)
    st.latex(r"E_{\text{token}} = \frac{N \times \left(1 - \frac{p}{100}\right)}{T_{\text{PLN}}}, \qquad D_{\text{token}} = \frac{E_{\text{token}}}{\sum E_{\text{harian}}}")

    st.markdown("""
    **3. Deteksi Anomali Keborosan (*Isolation Forest* & *Z-Score*)**
    * **Isolation Forest (Unsupervised Machine Learning):** Model dilatih menggunakan 120 data historis pola konsumsi harian rumah tangga (`contamination = 0.05`). Kombinasi durasi nyala perangkat yang menyimpang jauh dari distribusi normal akan diisolasi lebih cepat pada pohon keputusan (*decision tree*) dan menghasilkan skor anomali negatif (`status = -1`).
    * **Diagnosis Perangkat Spesifik (*Z-Score*):** Untuk mengetahui perangkat mana yang memicu anomali, sistem menghitung simpangan baku relatif tiap alat ($Z = \\frac{t_{\\text{aktual}} - \\mu_{\\text{normal}}}{\\sigma}$) dan menandai perangkat dengan nilai $Z > 2{,}2$.

    **4. Proyeksi Tagihan (*Linear Regression*)**
    Model regresi linear memetakan hubungan antara total konsumsi energi harian ($x$) terhadap estimasi tagihan bulanan ($\\hat{y}$) untuk memproyeksikan tren kenaikan biaya:
    """)
    st.latex(r"\hat{y} = \beta_0 + \beta_1 x")