"""
Auto Complaint Categorization System
------------------------------------
✅ Inputs:
   - complaints.csv (columns: complaint_text)
   - categories.xlsx (columns: category, subcategory)

✅ Outputs:
   - matched_complaints.csv       → complaints mapped to existing categories
   - new_clusters.csv             → new complaint clusters (potential new categories)
   - new_cluster_summaries.csv    → auto-generated category names (via LLM)

🧠 Features:
   - Embedding-based category matching (cosine similarity ≥ 0.8)
   - HDBSCAN clustering for unmatched complaints
   - Optional: LLM summarization for new cluster names
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from tqdm import tqdm
from collections import defaultdict
import time

# === CONFIG ===
API_KEY = "YOUR_API_KEY"  # 🔑 Replace with your OpenAI API key
SIMILARITY_THRESHOLD = 0.8
EMBEDDING_MODEL = "text-embedding-3-large"
SUMMARIZE_NEW_CLUSTERS = True  # Set to False if you don’t want auto-naming

# === INIT ===
client = OpenAI(api_key=API_KEY)
tqdm.pandas()

# === STEP 1: LOAD DATA ===
complaints = pd.read_csv("complaints.csv")
categories = pd.read_excel("categories.xlsx")

if "complaint_text" not in complaints.columns:
    raise ValueError("complaints.csv must contain a 'complaint_text' column.")
if not {"category", "subcategory"}.issubset(categories.columns):
    raise ValueError("categories.xlsx must contain 'category' and 'subcategory' columns.")

categories["combined"] = categories["category"].astype(str) + " - " + categories["subcategory"].astype(str)

# === STEP 2: GENERATE EMBEDDINGS ===
def get_embedding(text):
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

print("\n🔹 Generating embeddings for categories...")
categories["embedding"] = categories["combined"].progress_apply(get_embedding)

print("\n🔹 Generating embeddings for complaints (this may take a while)...")
complaints["embedding"] = complaints["complaint_text"].progress_apply(get_embedding)

# Drop any rows where embedding failed
complaints = complaints.dropna(subset=["embedding"])
categories = categories.dropna(subset=["embedding"])

# === STEP 3: MATCH COMPLAINTS TO EXISTING CATEGORIES ===
print("\n🔹 Matching complaints to known categories...")
category_embeddings = np.stack(categories["embedding"].values)

matched_rows = []
unmatched_rows = []

for i, row in tqdm(complaints.iterrows(), total=len(complaints)):
    comp_emb = np.array(row["embedding"]).reshape(1, -1)
    sims = cosine_similarity(comp_emb, category_embeddings)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        matched_rows.append({
            "complaint_text": row["complaint_text"],
            "matched_category": categories.loc[best_idx, "category"],
            "matched_subcategory": categories.loc[best_idx, "subcategory"],
            "similarity": best_score
        })
    else:
        unmatched_rows.append(row)

matched_df = pd.DataFrame(matched_rows)
matched_df.to_csv("matched_complaints.csv", index=False)
print(f"✅ Matched {len(matched_df)} complaints to existing categories.")

# === STEP 4: CLUSTER UNMATCHED COMPLAINTS ===
if len(unmatched_rows) > 0:
    print(f"\n🔹 Clustering {len(unmatched_rows)} unmatched complaints...")
    unmatched_df = pd.DataFrame(unmatched_rows)
    X = np.stack(unmatched_df["embedding"].values)

    clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=10)
    unmatched_df["cluster"] = clusterer.fit_predict(X)

    unmatched_df.to_csv("new_clusters.csv", index=False)
    print(f"✅ Clustering complete: {unmatched_df['cluster'].nunique()} potential new categories found.")

    # === STEP 5: AUTO-NAME NEW CLUSTERS (Optional) ===
    if SUMMARIZE_NEW_CLUSTERS:
        print("\n🔹 Generating auto-summaries for new clusters...")
        cluster_groups = defaultdict(list)
        for _, row in unmatched_df.iterrows():
            if row["cluster"] != -1:  # ignore noise
                cluster_groups[row["cluster"]].append(row["complaint_text"])

        summaries = []
        for cid, texts in tqdm(cluster_groups.items()):
            sample_texts = "\n".join(texts[:5])  # top 5 examples per cluster
            prompt = f"""
You are an expert in customer complaint analysis.
Summarize the common theme or issue in the following complaints in 3–5 words.

Complaints:
{sample_texts}
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                summary = f"Error generating summary: {e}"

            summaries.append({"cluster_id": cid, "suggested_category": summary})
            time.sleep(1.5)  # rate limit safety

        summaries_df = pd.DataFrame(summaries)
        summaries_df.to_csv("new_cluster_summaries.csv", index=False)
        print("✅ Auto-named new categories saved to new_cluster_summaries.csv")

else:
    print("🎉 All complaints matched to existing categories. No new clusters needed.")