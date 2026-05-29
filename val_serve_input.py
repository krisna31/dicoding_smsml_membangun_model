from mlflow.models import validate_serving_input
import json
 
model_uri = 'runs:/a1f437d171104eee81326093c3204328/model'
data = "runs:/a1f437d171104eee81326093c3204328/model/serving_input_example.json"
 
# Membuka dan membaca file JSON 
with open("data.json", "r") as file: 
    data = json.load(file) # Memuat JSON menjadi dictionary
 
# Validate the serving payload works on the model
validate_serving_input(model_uri, data)