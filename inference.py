# inference.py

import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from config import CAPABILITIES, TRACK_PROFILES, profile_to_vec

# =========================
# Load trained artifacts
# =========================
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# =========================
# Precompute track vectors
# =========================
track_vecs = {
    track: profile_to_vec(profile).reshape(1, -1)
    for track, profile in TRACK_PROFILES.items()
}

# =========================
# Validate input
# =========================
def validate_input(student_caps):

    for cap in CAPABILITIES:
        if cap not in student_caps:
            raise ValueError(f"Missing capability: {cap}")

    for cap, value in student_caps.items():

        if not isinstance(value, (int, float)):
            raise ValueError(f"{cap} must be numeric")

        if value < 0 or value > 1:
            raise ValueError(f"{cap} must be between 0 and 1")

    if sum(student_caps.values()) == 0:
        raise ValueError(
            "Invalid profile: all capabilities are zero. Please provide at least one non-zero capability."
        )

# =========================
# Recommendation function
# =========================
def recommend(student_caps, top_k=3):

    validate_input(student_caps)

    # -----------------------
    # Build student vector
    # -----------------------
    cap_values = [student_caps[cap] for cap in CAPABILITIES]
    student_vec = np.array(cap_values).reshape(1, -1)

    # -----------------------
    # Similarity features
    # (same order as training)
    # -----------------------
    sim_values = []

    for track in TRACK_PROFILES:

        sim = cosine_similarity(
            student_vec,
            track_vecs[track]
        )[0, 0]

        sim_values.append(sim)

    # -----------------------
    # Final feature vector
    # -----------------------
    x = np.array(cap_values + sim_values).reshape(1, -1)

    # -----------------------
    # Model prediction
    # -----------------------
    probs = model.predict_proba(x)[0]

    top_idx = probs.argsort()[::-1][:top_k]

    # -----------------------
    # Student strengths
    # -----------------------
    sorted_caps = sorted(
        student_caps.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_strengths = [cap for cap, _ in sorted_caps[:3]]

    # -----------------------
    # Build results
    # -----------------------
    results = []

    for i in top_idx:

        track = le.classes_[i]

        similarity = cosine_similarity(
            student_vec,
            track_vecs[track]
        )[0, 0]

        results.append({

            "track": track,

            "probability": float(
                round(probs[i] * 100, 2)
            ),

            "similarity": float(
                round(similarity * 100, 2)
            )

        })

    # -----------------------
    # Return structured output
    # -----------------------
    return {

        "student_strengths": top_strengths,

        "recommendations": results

    }