import requests

body = {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

response = requests.post(url='http://127.0.0.1:8001/predict', json=body)
# try:
#     response.raise_for_status()  # Lanza un error si la respuesta no es 200
#     print(response.json())  # Intenta obtener el JSON de la respuesta
# except requests.exceptions.HTTPError as err:
#     print(f"HTTP error occurred: {err}")
#     print("Response content:", response.text)  # Muestra el contenido de la respuesta
# except ValueError:
#     print("Response content is not valid JSON.")
#     print("Response content:", response.text)
print(response.json())

