import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt

# Simulated Explicit Ratings
ratings = pd.DataFrame([
    [5, 4, 0, 0, 3],
    [4, 5, 0, 2, 3],
    [0, 0, 5, 4, 0],
    [0, 0, 4, 5, 1],
    [4, 4, 0, 0, 5],
], columns=["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"])

print("Ratings Matrix:\n", ratings)

# Implicit Feedback (Watch Time)
implicit = pd.DataFrame([
    [300, 280, 0,   0,   240],
    [250, 320, 0,   100, 220],
    [0,   0,   290, 310, 0],
    [0,   0,   300, 350, 80],
    [220, 210, 0,   0,   270],
], columns=ratings.columns)

# Normalize to 0–5 scale
implicit_scaled = (implicit / implicit.to_numpy().max()) * 5
print("\nNormalized Feedback:\n", np.round(implicit_scaled, 1))

# Step 3: Collaborative Filtering via Matrix Factorization
svd = TruncatedSVD(n_components=2)
user_factors = svd.fit_transform(ratings)
item_factors = svd.components_

cf_reconstructed = np.dot(user_factors, item_factors)
cf_preds = pd.DataFrame(cf_reconstructed, columns=ratings.columns)
print("\nCollaborative Filtering Predictions:\n", cf_preds.round(2))

# Content-Based Filtering
movie_features = pd.DataFrame({
    "Genre_Comedy": [1, 1, 0, 0, 0],
    "Genre_Action": [0, 0, 1, 1, 0],
    "Mood_Upbeat": [1, 0, 0, 0, 1]
}, index=ratings.columns)

# Normalize ratings for user preference weighting
norm_ratings = ratings.div(ratings.max(axis=1), axis=0).fillna(0)

# User profile = weighted average of rated movie features
user_profile = norm_ratings.dot(movie_features)
user_profile = user_profile.div(user_profile.sum(axis=1), axis=0).fillna(0)

# Similarity scores between each user and each movie
cb_preds = cosine_similarity(user_profile, movie_features)
cb_preds_df = pd.DataFrame(cb_preds, index=ratings.index, columns=ratings.columns)

print("\nContent-Based Filtering Predictions:\n", cb_preds_df.round(2))

# Hybrid Recommender (Average)
hybrid_preds = (cf_preds + cb_preds_df) / 2
print("\nHybrid Recommendations:\n", hybrid_preds.round(2))

# Heatmap 
rounded_cb = cb_preds_df.round(2)

plt.figure(figsize=(10, 6))
sns.heatmap(
    rounded_cb,
    annot=True,
    cmap="viridis",            
    vmin=0,
    vmax=1,
    linewidths=0.5,
    linecolor='black',
    cbar_kws={"label": "Match Score (0 = weak, 1 = strong)"},
    xticklabels=rounded_cb.columns,
    yticklabels=[f"User {i}" for i in rounded_cb.index]
)

plt.title("User-Movie Match Scores (Content-Based Filtering)", fontsize=12)
plt.xlabel("Movies")
plt.ylabel("Users")
plt.tight_layout()
plt.show()



