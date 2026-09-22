import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mobile Recommendation System",
    page_icon="📱",
    layout="wide"
)

# -----------------------------
# Load Data
# -----------------------------
product_df = pd.read_csv("mobile_product_segmentation.csv")
similarity_matrix = np.load("similarity_matrix.npy")

# -----------------------------
# Title
# -----------------------------
st.title("📱 Mobile Product Segmentation & Recommendation System")

st.write(
    "An intelligent mobile product analysis and recommendation "
    "system using Python and Machine Learning."
)

# -----------------------------
# Check Data
# -----------------------------
st.success("Data loaded successfully! 🎉")

st.write("Number of products:", len(product_df))
st.write("Similarity matrix shape:", similarity_matrix.shape)

# -----------------------------
# Mobile Selection
# -----------------------------

st.header("🔍 Find Similar Mobile Phones")

model_list = product_df["model"].tolist()

selected_model = st.selectbox(
    "Select a mobile model:",
    model_list
)

st.write("You selected:", selected_model)
# -----------------------------
# Selected Mobile Details
# -----------------------------

selected_product = product_df[
    product_df["model"] == selected_model
].iloc[0]

st.subheader("📱 Selected Mobile Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Brand",
        selected_product["brand"]
    )

with col2:
    st.metric(
        "Price (USD)",
        f"${selected_product['avg_price_usd']:.2f}"
    )

with col3:
    st.metric(
        "Average Rating",
        f"{selected_product['avg_rating']:.2f} ⭐"
    )

with col4:
    st.metric(
        "Specification Score",
        f"{selected_product['avg_specification_score']:.2f} / 5"
    )
st.write(
    f"**Product Cluster:** Cluster {int(selected_product['cluster'])}"
)
# -----------------------------
# Product Recommendations
# -----------------------------

st.subheader("🤖 Top 5 Similar Mobile Recommendations")
st.info(
    f"💡 Recommendations for **{selected_model}** are based on "
    "similarity in price, rating, battery, camera, performance, "
    "design, and display characteristics."
)

# Find the selected model's position in the product dataset
model_index = product_df.index[
    product_df["model"] == selected_model
][0]

# Get similarity scores for the selected model
similarity_scores = similarity_matrix[model_index]

# Sort products from highest to lowest similarity
similar_indices = similarity_scores.argsort()[::-1]

recommendations = []

for index in similar_indices:

    # Skip the selected mobile itself
    if index == model_index:
        continue

    recommendations.append({
        "Model": product_df.loc[index, "model"],
        "Brand": product_df.loc[index, "brand"],
        "Price (USD)": round(
            product_df.loc[index, "avg_price_usd"], 2
        ),
        "Rating": round(
            product_df.loc[index, "avg_rating"], 2
        ),
        "Specification Score": round(
            product_df.loc[index, "avg_specification_score"], 2
        ),
        "Similarity Score": round(
            similarity_scores[index], 3
        )
    })

    # Stop after 5 recommendations
    if len(recommendations) == 5:
        break

recommendation_df = pd.DataFrame(recommendations)

st.dataframe(
    recommendation_df,
    use_container_width=True,
    hide_index=True
)
# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("📱 Mobile Recommendation")

st.sidebar.markdown("""
### About the Project

This application uses:

- 🐍 Python
- 📊 Pandas
- 🤖 K-Means Clustering
- 📐 Cosine Similarity
- 🎯 Content-Based Recommendation
- 🎈 Streamlit

### Project Workflow

**Data Cleaning → EDA → Feature Engineering →  
K-Means Clustering → Similarity Analysis →  
Mobile Recommendations**
""")

st.sidebar.info(
    "Recommendations are generated using product-level "
    "features such as price, rating and specifications."
)
# -----------------------------
# Specification Analysis
# -----------------------------

st.subheader("📊 Specification Analysis")

specification_data = pd.DataFrame({
    "Specification": [
        "Battery",
        "Camera",
        "Performance",
        "Design",
        "Display"
    ],
    "Score": [
        selected_product["avg_battery_rating"],
        selected_product["avg_camera_rating"],
        selected_product["avg_performance_rating"],
        selected_product["avg_design_rating"],
        selected_product["avg_display_rating"]
    ]
})

st.bar_chart(
    specification_data.set_index("Specification")
)
# -----------------------------
# Cluster Overview
# -----------------------------

st.subheader("🧩 Product Cluster Overview")

cluster_data = product_df.groupby("cluster").agg(
    Product_Count=("model", "count"),
    Average_Price=("avg_price_usd", "mean"),
    Average_Rating=("avg_rating", "mean"),
    Average_Specification=("avg_specification_score", "mean")
).reset_index()

cluster_data["cluster"] = cluster_data["cluster"].apply(
    lambda x: f"Cluster {int(x)}"
)

st.dataframe(
    cluster_data,
    use_container_width=True,
    hide_index=True
)