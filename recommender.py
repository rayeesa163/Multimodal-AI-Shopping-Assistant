import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load products
products = pd.read_csv("products.csv")

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_products(user_text, user_emotion, top_n=3):
    # Filter by emotion
    filtered = products[products['emotion_tag'] == user_emotion]

    # Encode product names and user query
    product_embeddings = model.encode(filtered['product_name'].tolist(), convert_to_tensor=True)
    user_embedding = model.encode(user_text, convert_to_tensor=True)

    # Compute similarity
    scores = util.cos_sim(user_embedding, product_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_n]

    # Show recommendations
    print("\n🛍️ Recommended for you:")
    for idx in top_indices:
        product = filtered.iloc[int(idx)]
        print(f"➡️ {product['product_name']} - ₹{product['price']}")

# 🧪 Test
if __name__ == "__main__":
    recommend_products("looking for stylish sneakers", "happy")
