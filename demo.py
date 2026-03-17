# demo.py
from inference import recommend
from llm_explainer import explain

student = {
    "Programming":0.55,
    "Algorithms":0.65,
    "Math":0.35,
    "Theory":0.68,
    "Data":0.25,
    "Systems":0.45,
    "Hardware":0.90,
    "AI":0.20,
    "UX":0,
    "Security":0.50,
    "Graphics":0.36,
    "Biology":0.35,
}


result = recommend(student)

print("\n=== Student Strengths ===")
print(", ".join(result["student_strengths"]))

print("\n=== Recommendations ===")
for r in result["recommendations"]:
    print(f"{r['track']}")

print("\n=== AI Explanation ===")
print(explain(student))
