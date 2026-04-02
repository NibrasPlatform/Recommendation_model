from flask import Flask, request, jsonify
from inference import recommend
from mapper import grades_to_capabilities

# LLM (optional)
try:
    from llm_explainer import explain
    USE_LLM = True
except:
    USE_LLM = False

app = Flask(__name__)
# ---------------------------
# Health check
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Recommender API is running 🚀"
    })
# ---------------------------
# Main endpoint
# ---------------------------
@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    try:
        data = request.get_json()

        # validate request
        if not data or "grades" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'grades' field"
            }), 400

        grades = data["grades"]

        # check empty input
        if len(grades) == 0:
            return jsonify({
                "status": "error",
                "message": "Grades cannot be empty"
            }), 400

        # 1. grades → capabilities
        caps = grades_to_capabilities(grades)

        # check all zeros
        if all(v == 0 for v in caps.values()):
            return jsonify({
                "status": "error",
                "message": "All capabilities are zero"
            }), 400

        # 2. model
        result = recommend(caps)

        # 3. clean response
        track_names = [rec["track"] for rec in result["recommendations"]]

        response = {
            "strengths": result["student_strengths"],
            "recommendations": track_names
        }

        # 4. LLM explanation
        if USE_LLM:
            try:
                explanation = explain(caps)
                response["explanation"] = explanation
            except Exception as e:
                response["explanation"] = "Explanation not available"

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)