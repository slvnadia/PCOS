# =============================================================================
# PCOS Early Detection — Streamlit Prototype
# Perbaikan: input 41 fitur lengkap + form kosong saat pertama dibuka
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
# LOAD MODEL DAN FITUR
# =============================================================================
@st.cache_resource
def load_artifacts():
    model         = joblib.load("best_model.pkl")
    scaler        = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    model_loaded = True
except FileNotFoundError as e:
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

>  **Perhatian:** Sistem ini merupakan *proof of concept* untuk keperluan
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
** Panduan Pengisian**
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

# Gunakan session_state untuk melacak status form
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "reset" not in st.session_state:
    st.session_state.reset = False

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Morfologi Ovarium**")
    follicle_r = st.number_input(
        "Jumlah Folikel Ovarium Kanan — Follicle No. (R)",
        min_value=0, max_value=50,
        value=None,                  # ← kosong saat pertama dibuka
        step=1,
        placeholder="Masukkan angka...",
        help="Jumlah folikel pada ovarium kanan (hasil USG)"
    )
    follicle_l = st.number_input(
        "Jumlah Folikel Ovarium Kiri — Follicle No. (L)",
        min_value=0, max_value=50,
        value=None,
        step=1,
        placeholder="Masukkan angka...",
        help="Jumlah folikel pada ovarium kiri (hasil USG)"
    )

    st.markdown("**Parameter Hormonal**")
    lh = st.number_input(
        "Kadar LH (mIU/mL)",
        min_value=0.0, max_value=100.0,
        value=None,
        step=0.1,
        placeholder="Masukkan angka...",
        help="Luteinizing Hormone dari hasil laboratorium"
    )
    amh = st.number_input(
        "Kadar AMH (ng/mL)",
        min_value=0.0, max_value=20.0,
        value=None,
        step=0.1,
        placeholder="Masukkan angka...",
        help="Anti-Müllerian Hormone dari hasil laboratorium"
    )

    st.markdown("**Siklus Menstruasi**")
    cycle = st.selectbox(
        "Keteraturan Siklus — Cycle (R/I)",
        options=[None, 1, 2],
        format_func=lambda x: "— Pilih —" if x is None
                              else ("Teratur (R)" if x == 1
                                    else "Tidak Teratur (I)"),
        help="R = Regular, I = Irregular"
    )

with col2:
    st.markdown("**Gejala Hiperandrogenisme**")
    hair_growth = st.selectbox(
        "Pertumbuhan Rambut Berlebih — Hair Growth (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None
                              else ("Ya" if x == 1 else "Tidak"),
        help="Hirsutisme: pertumbuhan rambut dengan pola maskulin"
    )
    skin_darkening = st.selectbox(
        "Penggelapan Kulit — Skin Darkening (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None
                              else ("Ya" if x == 1 else "Tidak"),
        help="Acanthosis nigricans pada lipatan kulit"
    )
    pimples = st.selectbox(
        "Jerawat Parah — Pimples (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None
                              else ("Ya" if x == 1 else "Tidak"),
        help="Jerawat akibat kelebihan hormon androgen"
    )

    st.markdown("**Parameter Metabolik**")
    weight_gain = st.selectbox(
        "Penambahan Berat Badan — Weight Gain (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None
                              else ("Ya" if x == 1 else "Tidak"),
        help="Penambahan berat badan yang tidak wajar"
    )

    st.markdown("**Gaya Hidup**")
    fast_food = st.selectbox(
        "Konsumsi Makanan Cepat Saji — Fast Food (Y/N)",
        options=[None, 1, 0],
        format_func=lambda x: "— Pilih —" if x is None
                              else ("Ya" if x == 1 else "Tidak"),
        help="Kebiasaan konsumsi makanan cepat saji secara rutin"
    )

# =============================================================================
# TOMBOL PREDIKSI DAN RESET
# =============================================================================
col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    predict_btn = st.button(
        "Prediksi Risiko PCOS",
        type="primary",
        use_container_width=True
    )
with col_btn2:
    reset_btn = st.button(
        "Reset",
        type="secondary",
        use_container_width=True
    )

if reset_btn:
    st.rerun()

# =============================================================================
# VALIDASI INPUT
# =============================================================================
def validate_inputs():
    missing = []
    if follicle_r is None: missing.append("Jumlah Folikel Ovarium Kanan")
    if follicle_l is None: missing.append("Jumlah Folikel Ovarium Kiri")
    if lh         is None: missing.append("Kadar LH")
    if amh        is None: missing.append("Kadar AMH")
    if cycle      is None: missing.append("Keteraturan Siklus Menstruasi")
    if hair_growth    is None: missing.append("Pertumbuhan Rambut Berlebih")
    if skin_darkening is None: missing.append("Penggelapan Kulit")
    if pimples        is None: missing.append("Jerawat Parah")
    if weight_gain    is None: missing.append("Penambahan Berat Badan")
    if fast_food      is None: missing.append("Konsumsi Makanan Cepat Saji")
    return missing

# =============================================================================
# PROSES PREDIKSI
# =============================================================================
if predict_btn:
    if not model_loaded:
        st.error(f"File model tidak ditemukan: {load_error}")

    else:
        # ── Validasi kelengkapan input ────────────────────────────────────────
        missing_fields = validate_inputs()
        if missing_fields:
            st.warning(
                "Mohon lengkapi semua kolom berikut sebelum prediksi:\n\n"
                + "\n".join([f"- {f}" for f in missing_fields])
            )

        else:
            # ── Buat input lengkap 41 fitur dengan nilai median (default 0) ──
            # Nilai default = 0 untuk semua fitur yang tidak diinput pengguna
            # (fitur non-top-10 dianggap memiliki pengaruh minimal)
            input_full = pd.DataFrame(
                np.zeros((1, len(feature_names))),
                columns=feature_names
            )

            # ── Isi 10 fitur dari input pengguna ─────────────────────────────
            top10_values = {
                "Follicle No. (R)"    : follicle_r,
                "hair growth(Y/N)"    : hair_growth,
                "Follicle No. (L)"    : follicle_l,
                "Weight gain(Y/N)"    : weight_gain,
                "Cycle(R/I)"          : cycle,
                "Skin darkening (Y/N)": skin_darkening,
                "LH(mIU/mL)"          : lh,
                "AMH(ng/mL)"          : amh,
                "Fast food (Y/N)"     : fast_food,
                "Pimples(Y/N)"        : pimples,
            }

            for col, val in top10_values.items():
                if col in input_full.columns:
                    input_full[col] = val
                else:
                    st.warning(f"Kolom '{col}' tidak ditemukan dalam model.")

            # ── Prediksi ──────────────────────────────────────────────────────
            try:
                prediction      = model.predict(input_full)[0]
                prediction_prob = model.predict_proba(input_full)[0]
                prob_negative   = prediction_prob[0]
                prob_positive   = prediction_prob[1]

                # ── Tampilkan hasil ───────────────────────────────────────────
                st.subheader("Hasil Prediksi")

                if prediction == 1:
                    st.error(
                        f"### RISIKO PCOS TERDETEKSI\n"
                        f"Model memprediksi pasien **positif PCOS** "
                        f"dengan probabilitas **{prob_positive:.1%}**"
                    )
                else:
                    st.success(
                        f"### RISIKO PCOS TIDAK TERDETEKSI\n"
                        f"Model memprediksi pasien **non-PCOS** "
                        f"dengan probabilitas **{prob_negative:.1%}**"
                    )

                st.divider()

                col_r1, col_r2 = st.columns(2)

                with col_r1:
                    st.markdown("**Probabilitas Prediksi**")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.barh(
                        ["Non-PCOS", "PCOS"],
                        [prob_negative, prob_positive],
                        color=["#5B9BD5", "#E07B54"],
                        edgecolor="white", height=0.5
                    )
                    for val, y in zip(
                        [prob_negative, prob_positive], [0, 1]
                    ):
                        ax.text(
                            val + 0.01, y,
                            f"{val:.1%}",
                            va="center",
                            fontsize=11,
                            fontweight="bold"
                        )
                    ax.set_xlim(0, 1.2)
                    ax.axvline(
                        x=0.5, color="gray",
                        linewidth=1, linestyle="--", alpha=0.5
                    )
                    ax.set_xlabel("Probabilitas")
                    ax.set_title("Distribusi Probabilitas")
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
                        "Siklus"             : "Teratur"
                                               if cycle == 1
                                               else "Tidak Teratur",
                        "Pertumbuhan Rambut" : "Ya"
                                               if hair_growth else "Tidak",
                        "Penggelapan Kulit"  : "Ya"
                                               if skin_darkening else "Tidak",
                        "Jerawat"            : "Ya"
                                               if pimples else "Tidak",
                        "Kenaikan BB"        : "Ya"
                                               if weight_gain else "Tidak",
                        "Fast Food"          : "Ya"
                                               if fast_food else "Tidak",
                    }
                    st.dataframe(
                        pd.DataFrame(
                            summary.items(),
                            columns=["Parameter", "Nilai"]
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                st.divider()
                st.info(
                    "** Catatan Klinis:** Hasil prediksi ini dihasilkan "
                    "oleh model komputasional dan **tidak dapat digunakan "
                    "sebagai diagnosis definitif**. Konfirmasi diagnosis PCOS "
                    "harus dilakukan oleh tenaga medis berlisensi melalui "
                    "pemeriksaan klinis lengkap mengacu pada "
                    "Kriteria Rotterdam (2003)."
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.code(str(e))
