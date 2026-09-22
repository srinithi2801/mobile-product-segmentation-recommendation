# ============================================================
# MOBILE PRODUCT SEGMENTATION & RECOMMENDATION PIPELINE
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

DATA_PATH = r"C:\Users\12622\Desktop\guvi_projects\Mobile_Product_Segmentation_Recommendation\data\Mobile Reviews Sentiment null.csv"

df = pd.read_csv(DATA_PATH)

print("Raw data loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. DATA CLEANING
# ============================================================

def clean_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Remove duplicate records
    # --------------------------------------------------------

    df = df.drop_duplicates()

    # Remove duplicate review IDs
    df = df.drop_duplicates(subset="review_id")


    # --------------------------------------------------------
    # Clean local price column
    # --------------------------------------------------------

    df["price_local_numeric"] = (
        df["price_local"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)")[0]
    )

    df["price_local_numeric"] = pd.to_numeric(
        df["price_local_numeric"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Recover missing USD prices
    # using local price / exchange rate
    # --------------------------------------------------------

    missing_usd = (
        df["price_usd"].isna()
        & df["price_local_numeric"].notna()
    )

    calculated_usd = (
        df.loc[missing_usd, "price_local_numeric"]
        / df.loc[missing_usd, "exchange_rate_to_usd"]
    )

    df.loc[missing_usd, "price_usd"] = (
        calculated_usd.astype(float).round(2)
    )


    # --------------------------------------------------------
    # Fill remaining missing USD prices
    # using model median
    # --------------------------------------------------------

    model_median_price = (
        df.groupby("model")["price_usd"]
        .transform("median")
    )

    df["price_usd"] = df["price_usd"].fillna(
        model_median_price
    )


    # --------------------------------------------------------
    # Recover missing local prices
    # using USD price * exchange rate
    # --------------------------------------------------------

    missing_local = df["price_local_numeric"].isna()

    calculated_local = (
        df.loc[missing_local, "price_usd"]
        * df.loc[missing_local, "exchange_rate_to_usd"]
    )

    df.loc[missing_local, "price_local_numeric"] = (
        calculated_local.astype(float).round(2)
    )


    # Final fallback for local price
    df["price_local_numeric"] = (
        df["price_local_numeric"]
        .fillna(
            df["price_usd"]
            * df["exchange_rate_to_usd"]
        )
        .round(2)
    )


    # --------------------------------------------------------
    # Fill missing ratings using model median
    # --------------------------------------------------------

    model_median_rating = (
        df.groupby("model")["rating"]
        .transform("median")
    )

    df["rating"] = df["rating"].fillna(
        model_median_rating
    )


    # --------------------------------------------------------
    # Fill missing sentiment using rating
    # --------------------------------------------------------

    def sentiment_from_rating(rating):

        if rating <= 2:
            return "Negative"

        elif rating == 3:
            return "Neutral"

        else:
            return "Positive"


    missing_sentiment = df["sentiment"].isna()

    df.loc[missing_sentiment, "sentiment"] = (
        df.loc[missing_sentiment, "rating"]
        .apply(sentiment_from_rating)
    )


    # --------------------------------------------------------
    # Fill missing source
    # --------------------------------------------------------

    df["source"] = df["source"].fillna("Unknown")


    # --------------------------------------------------------
    # Clean text columns
    # --------------------------------------------------------

    text_columns = [
        "customer_name",
        "brand",
        "model",
        "country",
        "language",
        "sentiment",
        "source",
        "currency"
    ]

    for col in text_columns:

        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )


    # --------------------------------------------------------
    # Convert review date
    # --------------------------------------------------------

    df["review_date"] = pd.to_datetime(
        df["review_date"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Remove original price_local column
    # because cleaned numeric version is used
    # --------------------------------------------------------

    df = df.drop(
    columns=["price_local"]
    )


    return df


# Apply cleaning
df = clean_data(df)

print("\nData cleaning completed!")
print("Cleaned dataset shape:", df.shape)
print(
    "Remaining missing values:",
    df.isna().sum().sum()
)


# ============================================================
# 3. DATE FEATURE ENGINEERING
# ============================================================

df["review_year"] = (
    df["review_date"].dt.year
)

df["review_month"] = (
    df["review_date"].dt.month
)

df["review_month_name"] = (
    df["review_date"].dt.month_name()
)

df["review_day_of_week"] = (
    df["review_date"].dt.day_name()
)


# ============================================================
# 4. SPECIFICATION SCORE
# ============================================================

rating_columns = [
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

df["specification_score"] = (
    df[rating_columns].mean(axis=1)
)

print("\nFeature engineering completed!")

print(
    "Specification score mean:",
    round(
        df["specification_score"].mean(),
        2
    )
)


# ============================================================
# 5. CREATE PRODUCT-LEVEL DATASET
# ============================================================

product_df = df.groupby(
    ["brand", "model"],
    as_index=False
).agg(

    avg_price_usd=(
        "price_usd",
        "mean"
    ),

    avg_rating=(
        "rating",
        "mean"
    ),

    avg_battery_rating=(
        "battery_life_rating",
        "mean"
    ),

    avg_camera_rating=(
        "camera_rating",
        "mean"
    ),

    avg_performance_rating=(
        "performance_rating",
        "mean"
    ),

    avg_design_rating=(
        "design_rating",
        "mean"
    ),

    avg_display_rating=(
        "display_rating",
        "mean"
    ),

    avg_specification_score=(
        "specification_score",
        "mean"
    ),

    avg_helpful_votes=(
        "helpful_votes",
        "mean"
    ),

    review_count=(
        "review_id",
        "count"
    ),

    verified_purchase_rate=(
        "verified_purchase",
        "mean"
    )
)


print("\nProduct-level dataset created!")

print(
    "Number of products:",
    len(product_df)
)

print(
    "Product dataset shape:",
    product_df.shape
)


# ============================================================
# 6. MACHINE LEARNING FEATURES
# ============================================================

ml_features = [

    "avg_price_usd",

    "avg_rating",

    "avg_battery_rating",

    "avg_camera_rating",

    "avg_performance_rating",

    "avg_design_rating",

    "avg_display_rating",

    "avg_helpful_votes",

    "verified_purchase_rate"
]

X = product_df[
    ml_features
].copy()

print(
    "\nML feature matrix shape:",
    X.shape
)


# ============================================================
# 7. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=ml_features
)

print(
    "\nFeature scaling completed!"
)

print(
    "Scaled feature means:",
    X_scaled_df.mean()
    .round(2)
    .tolist()
)


# ============================================================
# 8. K-MEANS CLUSTERING
# ============================================================

# K = 2 was selected based on the
# Elbow and Silhouette analysis
# performed during the project.

optimal_k = 2

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

product_df["cluster"] = (
    kmeans.fit_predict(X_scaled)
)

print(
    "\nK-Means clustering completed!"
)

print(
    "Number of clusters:",
    product_df["cluster"].nunique()
)


# ============================================================
# 9. CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    product_df
    .groupby("cluster")
    .agg(

        product_count=(
            "model",
            "count"
        ),

        avg_price_usd=(
            "avg_price_usd",
            "mean"
        ),

        avg_rating=(
            "avg_rating",
            "mean"
        ),

        avg_specification_score=(
            "avg_specification_score",
            "mean"
        ),

        avg_helpful_votes=(
            "avg_helpful_votes",
            "mean"
        ),

        verified_purchase_rate=(
            "verified_purchase_rate",
            "mean"
        )
    )
    .reset_index()
)

print("\nCluster summary:")
print(cluster_summary)


# ============================================================
# 10. CONTENT-BASED RECOMMENDATION SYSTEM
# ============================================================

recommendation_features = [

    "avg_price_usd",

    "avg_rating",

    "avg_battery_rating",

    "avg_camera_rating",

    "avg_performance_rating",

    "avg_design_rating",

    "avg_display_rating"
]

recommendation_matrix = (
    product_df[
        recommendation_features
    ].copy()
)


# Scale recommendation features
recommendation_scaler = StandardScaler()

recommendation_scaled = (
    recommendation_scaler
    .fit_transform(
        recommendation_matrix
    )
)


# Calculate cosine similarity
similarity_matrix = (
    cosine_similarity(
        recommendation_scaled
    )
)


print(
    "\nRecommendation system created!"
)

print(
    "Similarity matrix shape:",
    similarity_matrix.shape
)


# ============================================================
# 11. RECOMMENDATION FUNCTION
# ============================================================

def recommend_mobile(
    model_name,
    top_n=5
):

    if model_name not in product_df[
        "model"
    ].values:

        return (
            f"Model '{model_name}' not found."
        )


    model_index = (
        product_df.index[
            product_df["model"]
            == model_name
        ][0]
    )


    similarity_scores = (
        similarity_matrix[
            model_index
        ]
    )


    similar_indices = (
        similarity_scores
        .argsort()[::-1]
    )


    recommendations = []


    for index in similar_indices:

        # Skip the selected product
        if index == model_index:
            continue


        recommendations.append({

            "model":
                product_df.loc[
                    index,
                    "model"
                ],

            "brand":
                product_df.loc[
                    index,
                    "brand"
                ],

            "avg_price_usd":
                round(
                    product_df.loc[
                        index,
                        "avg_price_usd"
                    ],
                    2
                ),

            "avg_rating":
                round(
                    product_df.loc[
                        index,
                        "avg_rating"
                    ],
                    2
                ),

            "specification_score":
                round(
                    product_df.loc[
                        index,
                        "avg_specification_score"
                    ],
                    2
                ),

            "similarity_score":
                round(
                    similarity_scores[index],
                    3
                ),

            "cluster":
                product_df.loc[
                    index,
                    "cluster"
                ]
        })


        if len(recommendations) == top_n:
            break


    return pd.DataFrame(
        recommendations
    )


# ============================================================
# 12. TEST RECOMMENDATION SYSTEM
# ============================================================

test_model = "OnePlus 12"

print(
    f"\nRecommendations for {test_model}:"
)

print(
    recommend_mobile(
        test_model,
        top_n=5
    )
)


# ============================================================
# 13. SAVE PRODUCT DATASET
# ============================================================

product_df.to_csv(
    "mobile_product_segmentation.csv",
    index=False
)


# ============================================================
# 14. SAVE SIMILARITY MATRIX
# ============================================================

np.save(
    "similarity_matrix.npy",
    similarity_matrix
)


# ============================================================
# 15. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "data/cleaned_mobile_reviews.csv",
    index=False
)


# ============================================================
# 16. FINAL PIPELINE STATUS
# ============================================================

print("\n" + "=" * 60)

print(
    "PIPELINE COMPLETED SUCCESSFULLY!"
)

print("=" * 60)

print(
    "Raw records:",
    len(df)
)

print(
    "Unique products:",
    len(product_df)
)

print(
    "Clusters:",
    product_df[
        "cluster"
    ].nunique()
)

print(
    "Similarity matrix:",
    similarity_matrix.shape
)

print(
    "Missing values:",
    df.isna().sum().sum()
)

print("\nFiles generated:")

print(
    "1. mobile_product_segmentation.csv"
)

print(
    "2. similarity_matrix.npy"
)

print(
    "3. data/cleaned_mobile_reviews.csv"
)

print("=" * 60)
