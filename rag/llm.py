import os
import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_from_llm(prompt: str, model_name="models/gemini-2.0-flash"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

# client = OpenAI(
#     api_key=os.getenv("GROQ_API_KEY"),
#     base_url="https://api.groq.com/openai/v1"
# )

# def generate_from_llm(prompt: str, model_name="llama3-70b-8192"):
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7,
#         max_tokens=2048
#     )
#     return response.choices[0].message.content.strip()
