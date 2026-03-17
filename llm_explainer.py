# llm_explainer.py
import os
import json
from openai import OpenAI
from inference import recommend

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain(student_caps):
    """
    Generate an explanation for recommended tracks for a student
    using OpenAI LLM based on student capabilities and model predictions.
    """
    # أولاً نجيب التوصيات
    result = recommend(student_caps)

    system_prompt = (
        "You are an academic advisor helping a Computer Science student "
        "choose the best specialization track.\n\n"
        "Rules:\n"
        "1) Keep track names and probabilities exactly as provided.\n"
        "2) Explain WHY each recommended track fits the student.\n"
        "3) Use the student's strongest capabilities to justify recommendations.\n"
        "4) Be clear, concise, and structured."
    )

    user_prompt = (
        "Student capability profile:\n"
        f"{json.dumps(student_caps, indent=2)}\n\n"
        "Model recommendation output:\n"
        f"{json.dumps(result, indent=2)}\n\n"
        "Please explain:\n"
        "1) List the top 3 recommended tracks.\n"
        "2) Explain why the first track is the best match.\n"
        "3) Highlight which capabilities influenced each recommendation the most.\n"
        "4) Give short advice for the student."
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        max_tokens=400
    )

    return response.choices[0].message.content