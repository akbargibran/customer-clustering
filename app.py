
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Customer Clustering dengan K-Means")

st.write(
    "Aplikasi ini digunakan untuk mengelompokkan pelanggan "
    "berdasarkan pendapatan dan aktivitas pembelian."
)

uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    features = [
        'Income',
        'Recency',
        'MntWines',
        'MntMeatProducts',
        'NumWebPurchases',
        'NumStorePurchases',
        'NumCatalogPurchases'
    ]

    X_cluster = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("Hasil Clustering")

    st.write("Jumlah pelanggan pada setiap cluster:")

    cluster_counts = df['Cluster'].value_counts().sort_index()

    st.dataframe(cluster_counts)

    st.subheader("Data Pelanggan dengan Cluster")

    st.dataframe(df)

    st.subheader("Rata-rata Karakteristik Setiap Cluster")

    cluster_summary = df.groupby('Cluster')[features].mean()

    st.dataframe(cluster_summary)
