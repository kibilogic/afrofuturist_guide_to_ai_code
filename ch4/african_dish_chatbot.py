# Part 2 - Semantic chatbot
import pandas as pd
import faiss
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("african_dishes_model/processed_dishes.csv")
index = faiss.read_index("african_dishes_model/faiss_index.index")
model = joblib.load("african_dishes_model/embedding_model.pkl")

# Rule-based parser 
def parse_query(query):
    query_lower = query.lower()
    filters = {
        "spicy": "spicy" in query_lower,
        "vegetarian": "vegetarian" in query_lower,
        "region": None,
        "type": None,
        "dish_name": None
    }

    for region in df["Region"].unique():
        if region.lower() in query_lower:
            filters["region"] = region

    for word in ["soup", "stew", "snack", "dessert", "rice", "bread"]:
        if word in query_lower:
            filters["type"] = word

    for name in df["Name of Dish"].dropna().tolist():
        if name.lower() in query_lower:
            filters["dish_name"] = name
            break

    return filters

# Search logic 
def query_dishes(user_query, top_k=5):
    filters = parse_query(user_query)
    filtered_df = df.copy()

    if filters["spicy"]:
        filtered_df = filtered_df[filtered_df["Spicy"] == True]
    if filters["vegetarian"]:
        filtered_df = filtered_df[filtered_df["Vegetarian"] == True]
    if filters["region"]:
        filtered_df = filtered_df[filtered_df["Region"] == filters["region"]]
    if filters["type"]:
        filtered_df = filtered_df[filtered_df["Description"].str.lower().str.contains(filters["type"])]

    if filtered_df.empty:
        return ["No matches found."]

    if filters["dish_name"]:
        query_text = df[df["Name of Dish"] == filters["dish_name"]]["Description"].values[0]
    else:
        query_text = user_query

    query_embedding = model.encode([query_text])
    dish_embeddings = model.encode(filtered_df["Description"].tolist(), convert_to_numpy=True)

    temp_index = faiss.IndexFlatL2(dish_embeddings.shape[1])
    temp_index.add(dish_embeddings)

    D, I = temp_index.search(query_embedding, top_k)

    results = []
    for idx in I[0]:
        dish = filtered_df.iloc[idx]
        results.append(f"{dish['Name of Dish']} ({dish['Region']})\n→ {dish['Description']}\n")
    return results

# For Interaction  
import ipywidgets as widgets
from IPython.display import display

text = widgets.Text(
    value='',
    placeholder='e.g. spicy vegetarian dishes from West Africa',
    description='Query:',
    disabled=False
)

output = widgets.Output()

def handle_submit(sender):
    output.clear_output()
    with output:
        results = query_dishes(text.value)
        for r in results:
            print(r)

text.on_submit(handle_submit)
display(text, output)


