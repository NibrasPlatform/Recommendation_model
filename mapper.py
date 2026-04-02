def normalize_grade(g):

    return g / 100.0

def compute_capability(grades, courses):
    vals = [normalize_grade(grades[c]) for c in courses if c in grades]

    if len(vals) == 0:
        return 0.0

    return sum(vals) / len(vals)

def grades_to_capabilities(grades):

    caps = {}

    caps["Programming"] = compute_capability(grades, ["CS106A","CS106B","CS106X","CS 107","CS 110"])
    caps["Algorithms"]  = compute_capability(grades, ["CS 161","CS 103"])
    caps["Math"]        = compute_capability(grades, [
        "MATH 18","MATH 19","MATH 20","MATH 21",
        "MATH 51","MATH 52","MATH 53",
        "MATH 104","MATH 107","MATH 108",
        "MATH 109","MATH 110","MATH 113"
    ])
    caps["Theory"]      = compute_capability(grades, ["CS 103","CS 109","PHIL 251"])
    caps["Data"]        = compute_capability(grades, ["CS 109","CS 181","CS 181W"])
    caps["Systems"]     = compute_capability(grades, ["CS 110","CS 107"])
    caps["Hardware"]    = compute_capability(grades, ["ENGR 40M","ENGR 76"])
    caps["AI"]          = compute_capability(grades, ["CS 181","CS 181W","CS 205L"])
    caps["UX"]          = compute_capability(grades, ["CS106A"])
    caps["Security"]    = caps["Systems"] * 0.5   # fallback
    caps["Graphics"]    = compute_capability(grades, ["CS 205L"])
    caps["Biology"]     = compute_capability(grades, ["BIO","CHEM"])

    return caps
