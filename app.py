import os
import numpy as np
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
from PIL import Image
from werkzeug.utils import secure_filename
import base64



# # === Load Environment Variables ===
# load_dotenv()
# PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY")
# print("Loaded API Key:", PLANT_ID_API_KEY)
# print("Current working dir:", os.getcwd())
# print("Env file exists:", os.path.exists(".env"))

# === Flask Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load Your Model and Class Labels ===
model = load_model("model/leaf_disease_model.h5")
print("Expected input shape:", model.input_shape)
with open("labels.txt", "r") as f:
    class_labels = [line.strip() for line in f]

# === Model Prediction Function ===
def predict_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        # Replace this list with your actual class labels if you have them
        class_labels = ["Healthy", "Bacterial Spot", "Leaf Mold", "Early Blight", "Late Blight"]
        label = class_labels[class_index] if class_index < len(class_labels) else "Unknown"

        return label, round(confidence, 2)

    except Exception as e:
        print("Prediction error:", e)
        return "Prediction Failed", 0.0

# === Plant.id API Call ===
def get_plant_info_from_plant_id(img_path):
    url = "https://api.plant.id/v2/identify"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": "n0k349D2etFOamRGDEalvvSsLt8EUdX8QAXpst4eGMD3rDtNU6"
    }

    with open(img_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    data = {
        "images": [img_base64],
        "modifiers": ["crops_fast", "disease"],
        "plant_language": "en",
        "plant_details": ["common_names", "url", "name_authority", "wiki_description", "taxonomy", "synonyms", "edible_parts", "growth_rate", "watering", "propagation_methods", "soil_humidity", "disease_details"]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()

        # Extract most likely suggestion
        suggestion = result.get("suggestions", [{}])[0]
        name = suggestion.get("plant_name", "Unknown")
        description = suggestion.get("plant_details", {}).get("wiki_description", {}).get("value", "No description available.")
        soil = suggestion.get("plant_details", {}).get("soil_humidity", "Unknown")
        diseases = suggestion.get("plant_details", {}).get("disease_details", [])

        disease_info = ""
        if diseases:
            disease_info = f"Disease: {diseases[0].get('name', 'N/A')}\nTreatment: {diseases[0].get('treatment', {}).get('biological', 'N/A')}"

        return f"Plant Name: {name}\n\nDescription: {description}\n\nSoil Humidity: {soil}\n\n{disease_info}"
    except Exception as e:
        return f"Error calling Plant.id API: {e}"

# === Main Route ===
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)

                prediction, confidence = predict_image(filepath)
                plant_info = get_plant_info_from_plant_id(filepath)

                return render_template("result.html",
                                   image_path=filepath,
                                   prediction=prediction,
                                   confidence=confidence,
                                   gpt_response=plant_info)

            except Exception as e:
                 return f"<h2 style='color:red;'>Error: {e}</h2>"
    return render_template("index.html")
# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True)
