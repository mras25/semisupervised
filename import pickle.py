import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Klaster Mobil Bekas PakWheels", page_icon="🚗", layout="wide")

PKL = Path(__file__).parent / "model_bundle.pkl"


@st.cache_resource
def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if not PKL.exists():
    st.error("File `model_bundle.pkl` tidak ditemukan. Jalankan cell pickle di akhir notebook dulu, "
             "lalu taruh file-nya satu folder dengan app.py.")
    st.stop()

b = load_bundle(PKL)
features, scaler, models, data = b["features"], b["scaler"], b["models"], b["data"]


def nama_klaster(c):
    return "Noise / Outlier (-1)" if c == -1 else f"Klaster {c}"


st.title("🚗 Prediksi Klaster Mobil Bekas PakWheels")
st.caption("Klaster dibentuk dengan DBSCAN, lalu diprediksi memakai SVM / Decision Tree / Naive Bayes.")

tab1, tab2 = st.tabs(["🔮 Prediksi", "🧩 Profil Klaster"])

with tab1:
    model_name = st.sidebar.selectbox("Pilih model", list(models.keys()))

    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Tahun", 1961, 2026, 2018, step=1)
        price = st.number_input("Harga (PKR)", 150_000, 200_000_000, 3_500_000, step=50_000)
    with c2:
        mileage = st.number_input("Jarak tempuh (km)", 0, 1_000_000, 90_000, step=1_000)
        engine = st.number_input("Kapasitas mesin (cc)", 100, 8_000, 1300, step=50)

    if st.button("Prediksi klaster", type="primary"):
        x_new = pd.DataFrame([[year, price, mileage, engine]], columns=features)
        x_scaled = scaler.transform(x_new)
        model = models[model_name]
        pred = int(model.predict(x_scaled)[0])
        st.success(f"Hasil prediksi ({model_name}): **{nama_klaster(pred)}**")

        if hasattr(model, "predict_proba"):
            proba = pd.Series(model.predict_proba(x_scaled)[0],
                              index=[nama_klaster(c) for c in model.classes_])
            st.bar_chart(proba)

        st.write(f"Ringkasan data pada {nama_klaster(pred)}:")
        st.dataframe(data[data["cluster"] == pred][features].describe().loc[["mean", "min", "max"]].round(1),
                     use_container_width=True)

with tab2:
    cnt = data["cluster"].value_counts().sort_index()
    cnt.index = [nama_klaster(c) for c in cnt.index]
    st.subheader("Jumlah data per klaster")
    st.dataframe(cnt.rename("Jumlah data"), use_container_width=True)

    st.subheader("Rata-rata fitur per klaster")
    st.dataframe(data.groupby("cluster")[features].mean().round(1), use_container_width=True)

    st.subheader("Scatter plot")
    a, c = st.columns(2)
    fx = a.selectbox("Sumbu X", features, index=3)
    fy = c.selectbox("Sumbu Y", features, index=1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in sorted(data["cluster"].unique()):
        d = data[data["cluster"] == k]
        ax.scatter(d[fx], d[fy], s=8, alpha=0.6, label=nama_klaster(k))
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.legend()
    st.pyplot(fig)