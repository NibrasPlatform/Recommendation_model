# training.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
from config import CAPABILITIES, CAPS, TARGET_COL, TRACK_PROFILES, profile_to_vec, RANDOM_STATE

DATA_PATH = "students_1000_capabilities_tracks.csv"

# =========================
# Add similarity features safely
# =========================
def add_similarity_features(df):
    track_vecs = {t: profile_to_vec(p).reshape(1, -1) for t, p in TRACK_PROFILES.items()}
    student_matrix = df[CAPS].fillna(0).values

    for track, tvec in track_vecs.items():
        col_name = "sim_" + track.replace(" ", "_")
        # Avoid zero vector issues
        sims = []
        for student_vec in student_matrix:
            if np.all(student_vec == 0):
                sims.append(0.0)
            else:
                sims.append(cosine_similarity(student_vec.reshape(1, -1), tvec)[0, 0])
        df[col_name] = sims

    return df

# =========================
# Training pipeline
# =========================
def train():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.fillna(0)  # safety for nulls

    df = add_similarity_features(df)

    sim_cols = [c for c in df.columns if c.startswith("sim_")]
    feature_cols = CAPS + sim_cols

    X = df[feature_cols]
    y = df[TARGET_COL]

    print(f"Number of features: {len(feature_cols)} | Dataset size: {len(df)}")

    # =========================
    # Encode target labels
    # =========================
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # =========================
    # Train/test split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
    )

    # =========================
    # Model and hyperparameter search
    # =========================
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE
    )

    param_dist = {
        "n_estimators": [300, 500],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        xgb, param_dist, n_iter=5,
        scoring="f1_macro", cv=cv, random_state=RANDOM_STATE, verbose=1, n_jobs=-1
    )

    print("\nTraining model...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)
    print("Best CV Macro F1:", round(search.best_score_, 4))

    # =========================
    # Evaluation
    # =========================
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n===== Test Evaluation =====")
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # =========================
    # Feature Importance
    # =========================
    feat_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 Important Features:")
    print(feat_imp.head(10))

    # =========================
    # Retrain on full dataset
    # =========================
    print("\nRetraining model on full dataset...")
    best_model.fit(X, y_encoded)

    # =========================
    # Save model, encoder, and metrics
    # =========================
    joblib.dump(best_model, "model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "best_params": search.best_params_
    }
    joblib.dump(metrics, "training_metrics.pkl")
    print("\nModel trained & saved successfully.")


# =========================
# Run training
# =========================
if __name__ == "__main__":
    train()