# 🌱 Leaf Disease Detection & Plant Information Web App

This project is a **Flask-based web application** that:
- Uses a **CNN model** to detect plant leaf diseases from uploaded images.
- Identifies the plant and provides details (soil conditions, description, and diseases) using the **Plant.id API**.
- Provides disease detection confidence and suggested treatments.

---

## 📂 Project Structure

project-root/
│── model/
│ └── leaf_disease_model.h5 # Trained CNN model
│
│── templates/
│ ├── index.html # Upload page
│ └── result.html # Results display page
│
│── train_leaf/
│ └── python_train_leaf_model.py # CNN training script
│
│── static/uploads/ # Uploaded images (auto-created)
│
│── app.py # Flask app entry point
│── labels.txt # Class labels for diseases
│── api_key.env # Environment file (store Plant.id API key here)
│── requirements.txt # Python dependencies
│── README.md # Project documentation


---

## 🚀 Features
1. Upload a leaf image through the web interface.
2. CNN model predicts whether the leaf is **Healthy** or has a disease (e.g., Bacterial Spot, Leaf Mold, Early/Late Blight).
3. Calls **Plant.id API** to fetch:
   - Plant common names
   - Growth conditions (soil humidity, watering)
   - Disease information & treatment suggestions
4. Displays results in a clean HTML template.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/leaf-disease-detection.git
cd leaf-disease-detection

### 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

### 3. Install Requirements
pip install -r requirements.txt

#### 4. Setup API Key

Create a file named api_key.env in the project root:

PLANT_ID_API_KEY=your_plant_id_api_key_here


### 5. Run the Flask App
python app.py

## 🧠 Model Training (CNN)

Dataset: PlantVillage Dataset

Image size: 128×128

Model: Convolutional Neural Network built using TensorFlow/Keras

Training script: train_leaf/python_train_leaf_model.py

Saves best model as: model/leaf_disease_model.h5

## 📸 Example Workflow

Upload a tomato leaf image.

CNN model predicts: Late Blight (92.5% confidence).

Plant.id API provides:

Plant Name: Tomato

Description: Short Wikipedia description

Soil Humidity: Medium

Disease Info: Late Blight

Treatment: Suggested biological/chemical treatment

## 📝 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
