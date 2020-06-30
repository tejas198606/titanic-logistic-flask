import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Pclass':2.0000,'Sex':9.0000,'Age':6.0000,'Siblings':2.00000,'Parents':9.00000,'Fare':6.00000})

print(r.json())
