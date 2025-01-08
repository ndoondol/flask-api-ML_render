import requests

url_base = 'http://127.0.0.1:5000'

response = requests.get( f"{url_base}/" )
print("Reponse du eundpoint d'accueil:", response.text)

donnees_predire = {
    
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 62,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.127,
    "Age": 50
}

response = requests.post( f"{url_base}/predict", json=donnees_predire )
print("Reponse du endpoint de prediction :", response.text)