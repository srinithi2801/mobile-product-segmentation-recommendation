📱 Mobile Product Segmentation and Recommendation System Using Python and Machine Learning
📌 Project Overview

This project develops an end-to-end mobile product segmentation and recommendation system using Python and Machine Learning.

The system analyzes mobile phone review data, performs data cleaning and exploratory analysis, segments products using K-Means clustering, and recommends similar mobile phones using a content-based recommendation approach with cosine similarity.

A Streamlit web application was developed to allow users to select a mobile phone and view its product information, cluster assignment, specification analysis, and similar mobile recommendations.

🎯 Project Objectives
Clean and preprocess mobile review data.
Handle missing values and duplicate records.
Perform exploratory data analysis (EDA).
Engineer useful product-level features.
Aggregate review-level data into product-level information.
Segment mobile products using K-Means clustering.
Evaluate different cluster counts using Elbow and Silhouette methods.
Analyze product cluster characteristics.
Build a content-based mobile recommendation system.
Calculate product similarity using cosine similarity.
Develop an interactive Streamlit application.
Document the complete machine learning workflow.
📊 Dataset

The dataset contains mobile phone customer review information.

Dataset Size
50,000 review records
22 original columns
Review period: 2022–2025
22 unique brand-model combinations
Important Columns
Column	Description
review_id	Unique review identifier
customer_name	Customer name
age	Customer age
brand	Mobile phone brand
model	Mobile phone model
price_usd	Mobile price in USD
price_local	Mobile price in local currency
currency	Local currency
exchange_rate_to_usd	Exchange rate
rating	Overall review rating
sentiment	Review sentiment
country	Customer country
language	Review language
review_date	Date of review
verified_purchase	Whether the purchase was verified
battery_life_rating	Battery rating
camera_rating	Camera rating
performance_rating	Performance rating
design_rating	Design rating
display_rating	Display rating
helpful_votes	Number of helpful votes
source	Review source
🧹 Data Preprocessing
1. Missing Value Analysis

Missing values were identified in:

price_usd
price_local
rating
sentiment
source

The missing values were approximately 5% of the dataset for the affected columns.

2. Price Cleaning

The local price column contained text and currency formatting.

The values were converted into numeric form.

Missing USD prices were recovered using:

Price USD = Local Price / Exchange Rate

Remaining missing USD prices were handled using model-level median prices.

3. Rating Imputation

Missing ratings were filled using the median rating of the corresponding mobile model.

4. Sentiment Imputation

Missing sentiment values were inferred from the rating:

Rating 1–2 → Negative
Rating 3 → Neutral
Rating 4–5 → Positive
5. Categorical Data Cleaning

Text columns were cleaned by removing unnecessary whitespace.

Missing values in the source column were assigned:

Unknown

6. Date Processing

The review_date column was converted into datetime format.

Additional time-based features were created:

Review year
Review month
Review month name
Day of week
7. Duplicate Check
No duplicate full records were found.
No duplicate review_id values were found.
⚙️ Feature Engineering

A new feature called specification_score was created.

It represents the average of:

Battery rating
Camera rating
Performance rating
Design rating
Display rating

Formula:

Specification Score =
(Battery + Camera + Performance + Design + Display) / 5

📦 Product-Level Aggregation

The original dataset contains review-level records.

For machine learning, the data was aggregated at the brand-model level.

This produced:

22 unique mobile products

Product-level features included:

Average price
Average rating
Average battery rating
Average camera rating
Average performance rating
Average design rating
Average display rating
Average specification score
Average helpful votes
Review count
Verified purchase rate
📈 Exploratory Data Analysis

Several visualizations were created during EDA:

Average price by brand
Average rating by brand
Price vs. rating
Average specification score by brand
Individual specification ratings
Key Observations
Mobile prices showed noticeable variation across brands and products.
Overall product ratings were relatively close to one another.
Specification scores were also relatively similar across many products.
Price showed more variation than average ratings.
🤖 Product Segmentation Using K-Means

K-Means clustering was used to segment mobile products based on their product-level characteristics.

Machine Learning Features

The clustering model used:

avg_price_usd
avg_rating
avg_battery_rating
avg_camera_rating
avg_performance_rating
avg_design_rating
avg_display_rating
avg_helpful_votes
verified_purchase_rate

The derived avg_specification_score was excluded from clustering because it is calculated from the five individual specification ratings.

📏 Feature Scaling

Before applying K-Means, the numerical features were standardized using StandardScaler.

This prevents features with larger numerical ranges from dominating the clustering algorithm.

🔎 Choosing the Number of Clusters

Two evaluation techniques were used:

Elbow Method

K-Means inertia was calculated for different values of K.

Silhouette Score

Silhouette scores were calculated for:

K = 2 to 8

The highest silhouette score was obtained for K = 2.

The silhouette value was moderate, indicating that the product segments are present but not extremely separated.

Based on the evaluation, 2 clusters were selected for the final model.

🧩 Cluster Results
Cluster 0
5 products
Average price ≈ $771.94
Average rating ≈ 3.08
Average specification score ≈ 2.68
Cluster 1
17 products
Average price ≈ $719.95
Average rating ≈ 3.13
Average specification score ≈ 2.73

The clusters represent feature-based product groupings rather than simple brand categories.

Multiple brands appear across the clusters, indicating that the segmentation is based on the selected product characteristics.

📉 PCA Visualization

Principal Component Analysis (PCA) was used to project the standardized clustering features into two dimensions.

The PCA visualization provides a two-dimensional view of how the mobile products are distributed according to their feature patterns and cluster assignments.

🎯 Recommendation System

A content-based recommendation system was developed using cosine similarity.

The recommendation system compares mobile products based on:

Average price
Average rating
Battery rating
Camera rating
Performance rating
Design rating
Display rating

These features were standardized before calculating cosine similarity.

📐 Cosine Similarity

Cosine similarity measures the similarity between two product feature vectors.

The similarity matrix contains:

22 × 22 similarity values

because there are 22 mobile products.

The selected mobile itself is excluded from the recommendation results.

The system returns the Top 5 most similar mobile products.

Recommendation Process

Selected Mobile
↓
Find Product Feature Vector
↓
Calculate Similarity with All Products
↓
Sort Similarity Scores
↓
Remove Selected Product
↓
Return Top 5 Similar Products

The recommendation results include:

Model
Brand
Price
Average rating
Specification score
Similarity score
⚠️ Recommendation System Limitation

This project uses content-based recommendation rather than collaborative filtering.

The available dataset does not provide sufficient repeated user-product interaction history for a reliable collaborative filtering system.

Therefore, recommendations are based on product characteristics rather than user behavior.

🖥️ Streamlit Application

An interactive Streamlit application was developed for the project.

Application Features
📱 Mobile model selection
💰 Product price
⭐ Average rating
📊 Specification score
🧩 Product cluster
🤖 Top 5 similar mobile recommendations
📐 Similarity scores
📊 Specification analysis chart
🧩 Cluster overview
ℹ️ Recommendation methodology explanation
🛠️ Technologies Used
Programming Language
Python
Data Analysis
Pandas
NumPy
Visualization
Matplotlib
Seaborn
Streamlit charts
Machine Learning
Scikit-learn
StandardScaler
K-Means
Silhouette Score
PCA
Cosine Similarity
Application
Streamlit