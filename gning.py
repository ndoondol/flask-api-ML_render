from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

# Charger le modèle
model = joblib.load('logistic_regression_model.pkl')

# Initialiser l'application Flask
app = Flask(__name__)  # Corrigé ici

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
    if not request.json:
        return jsonify({"error": "Aucune donnée fournie"}), 400

    try:
        # Validation des données avec Pydantic
        donnees = DonneesEntree(**request.json)

        # Conversion des données en DataFrame
        donnees_df = pd.DataFrame([donnees.dict()])

        # Prédiction
        predictions = model.predict(donnees_df)
        probabilities = model.predict_proba(donnees_df)[:, 1]

        # Préparer les résultats
        resultats = donnees.dict()
        resultats['predictions'] = int(predictions[0])
        resultats['probabilite_diabete'] = float(probabilities[0])

        return jsonify({'resultats': resultats})
    except ValidationError as ve:
        return jsonify({"error": "Données invalides", "details": ve.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Lancer l'application Flask
if __name__ == '__main__':  # Corrigé ici
    app.run(debug=True)
