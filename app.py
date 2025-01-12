from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

# Charger le modèle et le scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialiser l'application Flask
app = Flask(__name__)

# Classe pour valider les données d'entrée
class DonneesEntree(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

# Route d'accueil
@app.route('/', methods=['GET'])
def accueil():
    return jsonify({"message": "Bienvenue sur l'API de prédiction de diabète"})

# Route pour prédire
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Le contenu doit être de type JSON"}), 415

    try:
        # Validation des données avec Pydantic
        donnees = DonneesEntree(**request.json)

        # Convertir les données en DataFrame
        donnees_df = pd.DataFrame([donnees.dict()])

        # Normaliser les données
        donnees_scaled = scaler.transform(donnees_df)

        # Prédiction
        prediction = model.predict(donnees_scaled)
        probability = model.predict_proba(donnees_scaled)[:, 1]

        # Préparer les résultats
        resultats = donnees.dict()
        resultats['prediction'] = int(prediction[0])
        resultats['probabilite_diabete'] = float(probability[0])

        return jsonify({'resultats': resultats})

    except ValidationError as ve:
        return jsonify({"error": "Données invalides", "details": ve.errors()}), 400
    except Exception as e:
        return jsonify({"error": "Erreur interne", "details": str(e)}), 500

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
