import requests

url = "http://127.0.0.1:8000/predict"
headers = {
    "Content-Type": "application/json"
}
with open("input.json", "rb") as file:
    response = requests.post(url, headers=headers, data=file)
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.text}")