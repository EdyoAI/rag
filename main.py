from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from rag.retrieve_generate import load_index_and_meta, retrieve_similar, build_prompt
from rag.llm import generate_from_llm
from config import *

app = FastAPI(title="Bengali MCQ Generator")

class GenerateRequest(BaseModel):
    exam: str
    subject: str
    topic: str
    count: int = 6
@app.get("/generate")
def hi():
    return "Bengali MCQ Generator"
@app.post("/generate")
def generate_mcqs(req: GenerateRequest):
    try:
        print("loading index and metadata")
        index, meta = load_index_and_meta()

        print("retrieving similar questions")
        print("topic:", req.topic, " exam:", req.exam, " subject:", req.subject)

        matches = retrieve_similar(meta, index,  req.topic,req.exam,req.subject)
        print(matches)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during retrieval: {str(e)}")

    if not matches:
        raise HTTPException(status_code=404, detail="No matching data found.")

    try:
        prompt = build_prompt(matches, req.count)
        response = generate_from_llm(prompt)
        return {
            "generated_questions": response
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")
