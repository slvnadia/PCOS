# =============================================================================
# PCOS Early Detection — Streamlit App
# Fix: model butuh 41 fitur, scaler hanya untuk 10 fitur top PFI
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="PCOS Early Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD MODEL, SCALER, DAN DAFTAR FITUR
# =============================================================================
@st.cache_resource
def load_artifacts():
    model         = joblib.load("best_model.pkl")   # Pipeline: SMOTE → RF (41 fitur)
    scaler        = joblib.load("scaler.pkl")        # StandardScaler (10 fitur top PFI)
    feature_names = joblib.load("feature_names.pkl") # list 41 nama fitur (urutan training)
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()

    # Nama 10 fitur yang di-scale (urutan sesuai scaler)
    SCALER_FEATURES = list(scaler.feature_names_in_)
    model_loaded    = True

except Exception as e:
    model_loaded = False
    load_error   = str(e)

# =============================================================================
# HEADER
# =============================================================================
st.title("Sistem Skrining Dini PCOS")
st.markdown("""
**Polycystic Ovary Syndrome (PCOS) Early Detection System**  
Berbasis algoritma *Random Forest* yang dilatih menggunakan dataset klinis
dari 10 rumah sakit di Kerala, India.

> **Perhatian:** Sistem ini merupakan *proof of concept* untuk keperluan
penelitian. **Bukan pengganti diagnosis medis profesional.**
""")
st.divider()

# =============================================================================
# SIDEBAR — INFORMASI MODEL
# =============================================================================
with st.sidebar:
    st.header("Informasi Model")
    st.markdown("""
**Algoritma:** Random Forest
**Validasi:** Stratified 5-Fold CV + SMOTE

| Metrik | Nilai CV |
|--------|----------|
| Accuracy | 0,9039 |
| Recall | 0,8190 |
| F1-Score | 0,8478 |
| AUC-ROC | 0,9575 |

---
**Fitur Input:**
Top 10 berdasarkan
*Permutation Feature Importance*
""")
    st.divider()
    st.markdown("""
**Panduan Pengisian**
- Isi semua kolom dengan data klinis pasien
- Klik tombol **Prediksi** untuk melihat hasil
- Klik **Reset** untuk mengosongkan form
""")

# =============================================================================
# FORM INPUT — KOSONG SAAT PERTAMA DIBUKA
# =============================================================================
st.subheader("Data Klinis Pasien")
st.markdown(
    "Masukkan nilai parameter klinis berikut "
    "berdasarkan hasil pemeriksaan pasien:"
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Morfologi Ovarium**")
    follicle_r = st.number_input(
        "Jumlah Folikel Ovarium Kanan — Follicle No. (R)",
        min_value=0, max_value=50,
        value=None, step=1,
        placeholder="Masukkan angka...",
        help="Jumlah folikel antral pada ovarium kanan (hasil USG)"
    )
    follicle_l = st.number_input(
        "Jumlah Folikel Ovarium Kiri — Follicle No. (L)",
        min_value=0, max_value=50,
        value=None, step=1,
        placeholder="Masukkan angka...",
        help="Jumlah folikel antral pada ovarium kiri (hasil USG)"
    )

    st.markdown("**Parameter Hormonal**")
    lh = st.number_input(
        "Kadar LH (mIU/mL) — Luteinizing Hormone",
        min_value=0.0, max_value=100.0,
        value=None, step=0.1,
        placeholder="Masukkan angka...",
        help="Luteinizing Hormone dari hasil laboratorium"
    )
    amh = st.number_input(
        "Kadar AMH (ng/mL) — Anti-Müllerian Hormone",
        min_value=0.0, max_value=20.0,
        value=None, step=0.1,
        placeholder="Masukkan angka...",
        help="Anti-Müllerian Hormone dari hasil laboratorium"
    )

    st.markdown("**Siklus Menstruasi**")
    cycle = st.selectbox(
        "Keteraturan Siklus — Cycle (R/I)",
        options=[None, 2, 4],
        format_func=lambda x: (
            "— Pilih —" if x is None
            else ("Tidak Teratur (I)" if x == 2 else "Teratur (R)")
        ),
        help="R = Regular (nilai 4), I = Irregular (nilai 2) — sesuai encoding dataset"
    )

with col2:
    st.markdown("**Gejala Hiperandrogenisme**")
    hair_growth = st.selectbox(
        "Pertumbuhan Rambut Berlebih — Hair Growth (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None else ("Ya" if x == 1 else "Tidak"),
        help="Hirsutisme: pertumbuhan rambut dengan pola maskulin"
    )
    skin_darkening = st.selectbox(
        "Penggelapan Kulit — Skin Darkening (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None else ("Ya" if x == 1 else "Tidak"),
        help="Acanthosis nigricans pada lipatan kulit"
    )
    pimples = st.selectbox(
        "Jerawat Parah — Pimples (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None else ("Ya" if x == 1 else "Tidak"),
        help="Jerawat akibat kelebihan hormon androgen"
    )

    st.markdown("**Parameter Metabolik**")
    weight_gain = st.selectbox(
        "Penambahan Berat Badan — Weight Gain (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None else ("Ya" if x == 1 else "Tidak"),
        help="Penambahan berat badan yang tidak wajar"
    )

    st.markdown("**Gaya Hidup**")
    fast_food = st.selectbox(
        "Konsumsi Makanan Cepat Saji — Fast Food (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None else ("Ya" if x == 1 else "Tidak"),
        help="Kebiasaan konsumsi makanan cepat saji secara rutin"
    )

# =============================================================================
# TOMBOL PREDIKSI DAN RESET
# =============================================================================
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    predict_btn = st.button("Prediksi Risiko PCOS", type="primary",
                            use_container_width=True)
with col_btn2:
    reset_btn = st.button("Reset", type="secondary", use_container_width=True)

if reset_btn:
    st.rerun()

# =============================================================================
# VALIDASI INPUT
# =============================================================================
def validate_inputs():
    missing = []
    checks = [
        (follicle_r,     "Jumlah Folikel Ovarium Kanan"),
        (follicle_l,     "Jumlah Folikel Ovarium Kiri"),
        (lh,             "Kadar LH"),
        (amh,            "Kadar AMH"),
        (cycle,          "Keteraturan Siklus Menstruasi"),
        (hair_growth,    "Pertumbuhan Rambut Berlebih"),
        (skin_darkening, "Penggelapan Kulit"),
        (pimples,        "Jerawat Parah"),
        (weight_gain,    "Penambahan Berat Badan"),
        (fast_food,      "Konsumsi Makanan Cepat Saji"),
    ]
    for val, label in checks:
        if val is None:
            missing.append(label)
    return missing

# =============================================================================
# PROSES PREDIKSI
# =============================================================================
if predict_btn:
    if not model_loaded:
        st.error(f"File model tidak ditemukan: {load_error}")
        st.stop()

    missing_fields = validate_inputs()
    if missing_fields:
        st.warning(
            "Mohon lengkapi semua kolom berikut sebelum prediksi:\n\n"
            + "\n".join([f"- {f}" for f in missing_fields])
        )
        st.stop()

    try:
        # ── Langkah 1: input 10 fitur dari pengguna (nilai raw) ──────────────
        user_input_10 = {
            "Follicle No. (R)"    : float(follicle_r),
            "hair growth(Y/N)"    : float(hair_growth),
            "Follicle No. (L)"    : float(follicle_l),
            "Weight gain(Y/N)"    : float(weight_gain),
            "Cycle(R/I)"          : float(cycle),
            "Skin darkening (Y/N)": float(skin_darkening),
            "LH(mIU/mL)"          : float(lh),
            "AMH(ng/mL)"          : float(amh),
            "Fast food (Y/N)"     : float(fast_food),
            "Pimples(Y/N)"        : float(pimples),
        }

        # ── Langkah 2: scale 10 fitur input menggunakan scaler ───────────────
        #    Buat DataFrame dengan urutan kolom SESUAI scaler.feature_names_in_
        df_10 = pd.DataFrame(
            [[user_input_10[f] for f in SCALER_FEATURES]],
            columns=SCALER_FEATURES
        )
        scaled_10 = scaler.transform(df_10)  # hasil: array shape (1, 10)

        # ── Langkah 3: buat array 41 fitur, semua diisi 0 sebagai baseline ──
        #    Posisi 10 fitur yang sudah di-scale diisi dengan nilai scaled
        input_41 = np.zeros((1, len(feature_names)))
        df_41    = pd.DataFrame(input_41, columns=feature_names)

        for i, feat in enumerate(SCALER_FEATURES):
            if feat in df_41.columns:
                df_41[feat] = scaled_10[0, i]

        # ── Langkah 4: prediksi ───────────────────────────────────────────────
        prediction      = model.predict(df_41)[0]
        prediction_prob = model.predict_proba(df_41)[0]
        prob_negative   = prediction_prob[0]
        prob_positive   = prediction_prob[1]

        # ── Tampilkan hasil ───────────────────────────────────────────────────
        st.subheader("Hasil Prediksi")

        if prediction == 1:
            st.error(
                f"###RISIKO PCOS TERDETEKSI\n"
                f"Model memprediksi pasien **positif PCOS** "
                f"dengan probabilitas **{prob_positive:.1%}**"
            )
        else:
            st.success(
                f"###RISIKO PCOS TIDAK TERDETEKSI\n"
                f"Model memprediksi pasien **non-PCOS** "
                f"dengan probabilitas **{prob_negative:.1%}**"
            )

        st.divider()

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("**Probabilitas Prediksi**")
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.barh(
                ["Non-PCOS", "PCOS"],
                [prob_negative, prob_positive],
                color=["#5B9BD5", "#E07B54"],
                edgecolor="white", height=0.5
            )
            for val, y in zip([prob_negative, prob_positive], [0, 1]):
                ax.text(val + 0.01, y, f"{val:.1%}",
                        va="center", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 1.25)
            ax.axvline(x=0.5, color="gray", linewidth=1,
                       linestyle="--", alpha=0.5, label="Batas 50%")
            ax.set_xlabel("Probabilitas")
            ax.set_title("Distribusi Probabilitas")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_r2:
            st.markdown("**Ringkasan Input Pasien**")
            summary = {
                "Folikel Kanan"      : follicle_r,
                "Folikel Kiri"       : follicle_l,
                "LH (mIU/mL)"        : lh,
                "AMH (ng/mL)"        : amh,
                "Siklus"             : "Teratur (R)" if cycle == 4
                                       else "Tidak Teratur (I)",
                "Pertumbuhan Rambut" : "Ya" if hair_growth == 1 else "Tidak",
                "Penggelapan Kulit"  : "Ya" if skin_darkening == 1 else "Tidak",
                "Jerawat"            : "Ya" if pimples == 1 else "Tidak",
                "Kenaikan BB"        : "Ya" if weight_gain == 1 else "Tidak",
                "Fast Food"          : "Ya" if fast_food == 1 else "Tidak",
            }
            st.dataframe(
                pd.DataFrame(summary.items(), columns=["Parameter", "Nilai"]),
                use_container_width=True,
                hide_index=True
            )

        st.divider()
        st.info(
            "**Catatan Klinis:** Hasil prediksi ini dihasilkan oleh model "
            "komputasional dan **tidak dapat digunakan sebagai diagnosis definitif**. "
            "Konfirmasi diagnosis PCOS harus dilakukan oleh tenaga medis berlisensi "
            "melalui pemeriksaan klinis lengkap mengacu pada Kriteria Rotterdam (2003)."
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.code(str(e))