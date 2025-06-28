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
'''
1
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

2
def retrieve_similar(mcqs_meta, index, topic, exam, subject, top_k=TOP_K):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL_NAME)

    topic_expansion_map = {
        "history": [
            "ইতিহাস", "মুঘল সাম্রাজ্য", "ব্রিটিশ শাসন", "বঙ্গভঙ্গ", "ভারতের স্বাধীনতা আন্দোলন",
            "বাংলার নবজাগরণ", "ঔপনিবেশিক শাসন", "উনিশ শতকের ইতিহাস", "সুভাষ চন্দ্র বসু", "গান্ধীজির আন্দোলন"
        ],
        "geography": [
            "ভূগোল", "নদীসমূহ", "পর্বতমালা", "মানচিত্র", "প্রাকৃতিক সম্পদ", "মৌসুমী বায়ু"
        ],
    }

    subject_expansion_map = {
        "current_affairs": [
            "কারেন্ট অ্যাফেয়ার্স", "চলতি ঘটনা", "বর্তমান ঘটনা", "সাম্প্রতিক বিষয়াবলি", "জাতীয় ও আন্তর্জাতিক খবর"
        ],
        "maths": [
            "গণিত", "পাটিগণিত", "অ্যালজেব্রা", "জ্যামিতি", "লঘুগুণ", "সমীকরণ", "প্রতিসাম্য"
        ],
    }

    expanded_topics = topic_expansion_map.get(topic.lower(), [topic])
    expanded_subjects = subject_expansion_map.get(subject.lower(), [subject])
    combined_queries = expanded_topics + expanded_subjects
    q_embs = model.encode(combined_queries)

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
'''
def retrieve_similar(mcqs_meta, index, topic, exam, subject, top_k=TOP_K):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL_NAME)

    topic_expansion_map = {
        "history": [
            "ইতিহাস", "মুঘল সাম্রাজ্য", "ব্রিটিশ শাসন", "বঙ্গভঙ্গ", "ভারতের স্বাধীনতা আন্দোলন",
            "বাংলার নবজাগরণ", "ঔপনিবেশিক শাসন", "উনিশ শতকের ইতিহাস", "সুভাষ চন্দ্র বসু", "গান্ধীজির আন্দোলন"
        ],
        "politics": [

            "রাজনীতি", "সংবিধান", "নির্বাচন", "রাষ্ট্রপতি", "প্রধানমন্ত্রী", "রাজনৈতিক দল", "আইনসভা", "সংসদ"
        ],
        "sports": [
            "খেলা", "ক্রিকেট", "ফুটবল", "অলিম্পিক", "ব্যাটমিন্টন", "হকি", "ক্রীড়া ব্যক্তিত্ব", "টুর্নামেন্ট"
        ],
        "algebra": [
            "বীজগণিত", "সমীকরণ", "সরল রৈখিক সমীকরণ", "দ্বিঘাত সমীকরণ", "বহুপদী", "ঘাত", "গুণনীয়ক", "রাশির গুণফল", "বিন্যাস", "সমবায়"
        ],
        "trigonometry": [
            "ত্রিকোণমিতি", "সাইন কোণ", "কোসাইন কোণ", "ট্যানজেন্ট", "ত্রিকোণমিতিক অনুপাত", "ত্রিকোণমিতিক সূত্র", "কোণ পরিমাপ", "সিন²θ + কোস²θ", "উচ্চতা ও দূরত্ব", "বৃত্তে কোণের প্রয়োগ"
        ]


       

    }

    subject_expansion_map = {
        "current_affairs": [
            "কারেন্ট অ্যাফেয়ার্স", "চলতি ঘটনা", "বর্তমান ঘটনা", "সাম্প্রতিক বিষয়াবলি", "জাতীয় ও আন্তর্জাতিক খবর"
        ],
        "maths": [
            "গণিত"
        ],
    }

    expanded_topics = topic_expansion_map.get(topic.lower(), [topic]) if topic else []
    expanded_subjects = subject_expansion_map.get(subject.lower(), [subject]) if subject else []
    combined_queries = expanded_topics + expanded_subjects

    results = []
    seen = set()

    if combined_queries:
        q_embs = model.encode(combined_queries)

        for q_emb in q_embs:
            D, I = index.search(np.array([q_emb]), top_k)
            for idx in I[0]:
                if idx not in seen:
                    mcq = mcqs_meta[idx]
                    if mcq["exam_name"].lower() == exam.lower():
                        if subject:
                            if mcq["topic_name"].lower() == subject.lower():
                                results.append(mcq)
                                seen.add(idx)
                        else:
                            results.append(mcq)
                            seen.add(idx)
    else:
        for mcq in mcqs_meta:
            if mcq["exam_name"].lower() == exam.lower():
                results.append(mcq)
                if len(results) >= top_k:
                    break

    return results


def build_prompt(mcqs: List[dict], n: int):
    prompt = f"""
Generate {n} new Bengali multiple-choice questions (MCQs) similar in topic and style to the examples provided below, and answers ,give the exact answer , dont give option number

Each question must follow this exact JSON format:
{{
  "question": "প্রশ্ন",
  "options": ["option1", "option2", "option3", "option4"],
  "answer":"<answer-dont just give option number>"
}}

❌ Do NOT include explanations, headings, notes, or wrap the response with ```json or any markdown formatting.
✅ Only return a clean JSON array like: [{{"question": "...", "options": [...], "answer": [...] }}, ...]

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
def clean_llm_response(raw_response: str) -> str:
    if raw_response.strip().startswith("```"):
        return raw_response.strip().strip("`").split("\n", 1)[1].rsplit("\n", 1)[0]
    return raw_response.strip()

'''

def build_prompt(mcqs: List[dict], n: int):
    prompt = f"""
Generate {n} new Bengali multiple-choice questions (MCQs) similar in topic and style to the examples provided below.

Each question must follow this **exact JSON format**:
{{
  "question": "প্রশ্ন",
  "options": ["option1", "option2", "option3", "option4"],
  "answer": 2  ← (This is the index (0–3) of the correct option)
}}

❌ Do NOT include explanations, headings, notes, or wrap the response with ```json or any markdown formatting.
✅ Only return a clean JSON array like:
[
  {{
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer": 2
  }},
  ...
]

Here are some examples:
"""
    prompt += "\n[\n"
    for i, mcq in enumerate(mcqs[:10]):
        json_obj = {
            "question": mcq["question"],
            "options": mcq["options"],
            "answer": mcq.get("answer", 0) 
        }
        prompt += f"  {json.dumps(json_obj, ensure_ascii=False)},\n"
    prompt = prompt.rstrip(",\n") + "\n]\n"

    prompt += "\nNow generate the new questions in the same JSON array format only."
    return prompt
'''