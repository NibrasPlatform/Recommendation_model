# config.py
import numpy as np

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CAPABILITIES = [
    "Programming","Algorithms","Math","Theory",
    "Data","Systems","Hardware","AI",
    "UX","Security","Graphics","Biology"
]

CAPS = [f"cap_{c}" for c in CAPABILITIES]

TARGET_COL = "Track_1"

# Track profiles (weights for capabilities)
TRACK_PROFILES = {
    "Artificial Intelligence": {"AI":0.35,"Math":0.20,"Algorithms":0.20,"Data":0.15,"Programming":0.10},
    "Systems": {"Systems":0.35,"Programming":0.25,"Hardware":0.20,"Algorithms":0.10,"Security":0.10},
    "Theory": {"Math":0.45,"Theory":0.35,"Algorithms":0.20},
    "Human-Computer Interaction": {"UX":0.40,"Programming":0.25,"Data":0.15,"Theory":0.10,"Graphics":0.10},
    "Visual Computing": {"Graphics":0.40,"Math":0.25,"AI":0.20,"Programming":0.15},
    "Computer Engineering": {"Hardware":0.40,"Systems":0.30,"Programming":0.15,"Security":0.15},
    "Information Track": {"Data":0.40,"Programming":0.25,"Math":0.15,"UX":0.10,"Security":0.10},
    "Computational Biology": {"Biology":0.35,"Data":0.25,"AI":0.20,"Math":0.20}
}

# =========================
# Function to convert track profile to a full vector
# =========================
def profile_to_vec(profile):
    """
    Convert a track profile dict into a vector aligned with CAPABILITIES.
    Missing capabilities are set to 0.0
    """
    vec = []
    for cap in CAPABILITIES:
        vec.append(profile.get(cap, 0.0))  # default 0.0 if not present
    return np.array(vec, dtype=float)

# =========================
# Precompute vectors for all tracks
# =========================
TRACK_VECS = {
    track: profile_to_vec(profile).reshape(1, -1)
    for track, profile in TRACK_PROFILES.items()
}