import joblib
import pandas as pd
import numpy as np

# ---------------- Load model & dataset ----------------
model = joblib.load("rf_disease_model.pkl")
symptoms = joblib.load("symptom_list.pkl")
classes = joblib.load("class_names.pkl")

# Load dataset (used for overlap scoring)
df = pd.read_csv("dataset_sorted.csv")
df = df.fillna("No Symptom")


# ---------------- Overlap function ----------------
def symptom_overlap(disease_data, given_symptoms):
    """
    Calculate overlap between patient's symptoms and disease's common symptoms.
    disease_data: subset of df where disease == current disease
    given_symptoms: list of symptoms patient has
    """
    # Average symptom presence for this disease
    disease_symptoms = (
        disease_data.drop(columns=["diseases"]).mean() > 0.1
    ).index.tolist()
    
    # Overlap with patient symptoms
    overlap = len(set(given_symptoms) & set(disease_symptoms))
    return overlap / len(given_symptoms) if given_symptoms else 0


# ---------------- Main Prediction Function ----------------
def predict_disease(selected_symptoms):
    """
    Takes a list of symptoms selected by user, returns top 3 disease predictions.
    """
    # Create patient vector
    patient_symptoms = {sym: 0 for sym in symptoms}
    for s in selected_symptoms:
        if s in patient_symptoms:
            patient_symptoms[s] = 1

    X_new = pd.DataFrame([patient_symptoms])

    # ML probabilities
    probs = model.predict_proba(X_new)[0] * 100  

    # Hybrid Scoring
    disease_scores = {}
    for idx, disease in enumerate(classes):
        prob = probs[idx]

        # Overlap score (0–100)
        disease_data = df[df["diseases"] == disease]
        overlap_score = symptom_overlap(disease_data, selected_symptoms) * 100

        # Final score = ML prob (strong) + small boost from overlap
        final_score = prob + (0.3 * overlap_score)   # adjust weight if needed
        disease_scores[disease] = final_score

    # -------- Softmax Normalization (for natural separation) --------
    scores = np.array(list(disease_scores.values()))
    exp_scores = np.exp(scores / 1.5)  # temperature=1.5 controls spread
    softmax_scores = exp_scores / np.sum(exp_scores)

    # Replace scores with normalized values (0–100)
    for i, d in enumerate(disease_scores.keys()):
        disease_scores[d] = softmax_scores[i] * 100

    # Sort top 6
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    return sorted_diseases
