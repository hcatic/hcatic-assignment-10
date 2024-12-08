from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import numpy as np
import io

app = Flask(__name__)

# Load image embeddings
df = pd.read_pickle('data/image_embeddings.pickle')

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
model = model.to(device)
model.eval()

# Helper function to calculate image embedding
def get_image_embedding(image_file):
    image = Image.open(image_file).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = F.normalize(model.encode_image(image_tensor))
    return embedding.cpu().numpy()

# Helper function to calculate text embedding
def get_text_embedding(text_query):
    text_tokens = tokenizer.tokenize([text_query]).to(device)
    with torch.no_grad():
        embedding = F.normalize(model.encode_text(text_tokens))
    return embedding.cpu().numpy()

# Compute cosine similarity
def compute_similarity(query_embedding, database_embeddings):
    similarities = database_embeddings @ query_embedding.T
    return similarities.squeeze()

@app.route('/coco_images_resized/<filename>')
def serve_coco_image(filename):
    return send_from_directory('coco_images_resized', filename)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        query_text = request.form.get("query_text")
        query_image = request.files.get("query_image")
        weight = float(request.form.get("weight", 0.5))
        query_type = request.form.get("query_type")

        text_embedding = None
        image_embedding = None

        # Process based on query type
        if query_type == "Image query" and query_image:
            image_embedding = get_image_embedding(query_image)
            combined_embedding = image_embedding
        elif query_type == "Text query" and query_text:
            text_embedding = get_text_embedding(query_text)
            combined_embedding = text_embedding
        elif query_type == "Hybrid query" and query_image and query_text:
            image_embedding = get_image_embedding(query_image)
            text_embedding = get_text_embedding(query_text)
            combined_embedding = weight * image_embedding + (1 - weight) * text_embedding
        else:
            combined_embedding = None

        # Compute similarity scores
        if combined_embedding is not None:
            embeddings = np.vstack(df["embedding"].values)
            similarities = compute_similarity(combined_embedding, embeddings)
            df["similarity"] = similarities
            results = df.nlargest(5, "similarity").to_dict(orient="records")

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)

