import json, pickle
import faiss
from sentence_transformers import SentenceTransformer

from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
def load_mcqs():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index(mcqs):
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [mcq["question"] + " " + " ".join(mcq["options"]) for mcq in mcqs]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(mcqs, f)

    print(f"Indexed {len(mcqs)} MCQs")

if __name__ == "__main__":
    mcqs = load_mcqs()
    build_index(mcqs)
