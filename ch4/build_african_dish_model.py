import pandas as pd
import numpy as np
import faiss
import joblib
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your dataset 
df = pd.read_csv("african_dishes_final.csv")  
df = df.dropna(subset=["Description", "Country/Region"])
df["Country/Region"] = df["Country/Region"].str.strip()

# Map countries to regions 
region_map = {
    'Nigeria': 'West Africa', 'Ghana': 'West Africa', 'Benin': 'West Africa',
    'Cameroon': 'Central Africa', 'Côte d\'Ivoire': 'West Africa',
    'Morocco': 'North Africa', 'Libya': 'North Africa',
    'South Africa': 'Southern Africa', 'Zimbabwe': 'Southern Africa',
    'Tanzania': 'East Africa', 'Namibia': 'Southern Africa',
    'Botswana': 'Southern Africa', 'Zambia': 'Southern Africa',
    'Niger': 'West Africa', 'Democratic Republic of Congo': 'Central Africa'
}

def map_region(country_cell):
    countries = [c.strip() for c in str(country_cell).split(',')]
    for c in countries:
        if c in region_map:
            return region_map[c]
    return "Other"

df["Region"] = df["Country/Region"].apply(map_region)
df = df[df["Region"] != "Other"]

# Add Semantic Tags 
def tag_vegetarian(desc):
    return not any(meat in desc.lower() for meat in ['meat', 'beef', 'goat', 'chicken', 'fish'])

def tag_spicy(desc):
    return any(spice in desc.lower() for spice in ['spicy', 'pepper', 'chili', 'hot'])

df["Vegetarian"] = df["Description"].apply(tag_vegetarian)
df["Spicy"] = df["Description"].apply(tag_spicy)

# Generate Embeddings 
model = SentenceTransformer('all-MiniLM-L6-v2')
df["Embedding"] = model.encode(df["Description"].tolist()).tolist()

# Train a Classifier 
embedding_matrix = np.vstack(df["Embedding"].values)
X_train, X_test, y_train, y_test = train_test_split(
    embedding_matrix, df["Region"], test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate 
report = classification_report(y_test, clf.predict(X_test), output_dict=True)
print("Classification Report:")
print(json.dumps(report, indent=2))

# Create FAISS Index 
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Save Everything 
os.makedirs("african_dishes_model", exist_ok=True)

df.to_csv("african_dishes_model/processed_dishes.csv", index=False)
joblib.dump(clf, "african_dishes_model/region_classifier.pkl")
joblib.dump(model, "african_dishes_model/embedding_model.pkl")
faiss.write_index(index, "african_dishes_model/faiss_index.index")

with open("african_dishes_model/classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("All models and data saved in './african_dishes_model'")
