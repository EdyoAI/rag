import pickle, faiss
from sentence_transformers import SentenceTransformer
from config import *
from typing import List
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def load_index_and_meta():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

import numpy as np
def retrieve_similar(mcqs_meta, index, topic, exam, subject, top_k=TOP_K):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL_NAME)
    topic_expansion_map = {
        "history": [
            "ইতিহাস", "মুঘল সাম্রাজ্য", "ব্রিটিশ শাসন", "বঙ্গভঙ্গ", "ভারতের স্বাধীনতা আন্দোলন",
            "বাংলার নবজাগরণ", "ঔপনিবেশিক শাসন", "উনিশ শতকের ইতিহাস", "সুভাষ চন্দ্র বসু", "গান্ধীজির আন্দোলন"
        ],

    }

    expanded_queries = topic_expansion_map.get(topic.lower(), [topic])
    q_embs = model.encode(expanded_queries)
    seen = set()
    results = []

    for q_emb in q_embs:
        D, I = index.search(np.array([q_emb]), top_k)
        for idx in I[0]:
            if idx not in seen:
                mcq = mcqs_meta[idx]
                if mcq["exam_name"] == exam and mcq["topic_name"].lower() == subject.lower():
                    results.append(mcq)
                    seen.add(idx)

    return results

def build_prompt(mcqs: List[dict], n: int):
    prompt = f"""
Generate {n} new Bengali multiple-choice questions (MCQs) similar in topic and style to the examples provided below.

Each question must follow this exact JSON format:
{{
  "question": "প্রশ্ন",
  "options": ["option1", "option2", "option3", "option4"]
}}

❌ Do NOT include explanations, headings, notes, or wrap the response with ```json or any markdown formatting.
✅ Only return a clean JSON array like: [{{"question": "...", "options": [...] }}, ...]

Examples:"""
    prompt += "\n[\n"
    for i, mcq in enumerate(mcqs[:10]):
        json_obj = {
            "question": mcq["question"],
            "options": mcq["options"]
        }
        prompt += f"  {json.dumps(json_obj, ensure_ascii=False)},\n"
    prompt = prompt.rstrip(",\n") + "\n]\n"

    prompt += "\nNow generate the new questions in the same JSON array format only."
    return prompt

