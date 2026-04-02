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
    result = recommend(student_caps)

    system_prompt = """You are a senior academic advisor at a Computer Science department.
Your role is to give honest, specific, data-driven track recommendations.
 
Rules you must follow:
1. Never use vague praise like "you have great potential" or "you are well-rounded".
2. Always reference specific capability scores and gap values when explaining fit.
3. For each track, name the 1-2 capabilities that most help AND the 1 capability gap that could limit the student.
4. Use a direct, professional tone — like a real advisor reading a transcript.
5. Keep the total response under 600 words.
6. Structure your response exactly as:
   ## Top recommendation: 
   [track name] ([probability]% match)
   [2-3 sentences explaining why, referencing specific scores]
   ## Also fits: [track 2] and [track 3]
   [1-2 sentences each, referencing scores]
   ## Capability gaps to address
   [Bullet list of 2-3 specific things the student should improve, with numbers]
   ## Advisor's note
   [1-2 sentences of honest, actionable advice]"""

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
