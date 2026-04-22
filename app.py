from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("crop_disease_model.keras")

class_names = ['Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus']

IMG_SIZE = 224

# Nutrient logic
def nutrient_deficiency(disease):
    disease = disease.lower()

    if "yellow" in disease:
        return "Nitrogen deficiency"
    elif "blight" in disease:
        return "Possible potassium deficiency"
    elif "spot" in disease:
        return "Possible nutrient imbalance"
    elif "virus" in disease:
        return "Not a nutrient imbalance, it's Virus"
    else:
        return "No clear deficiency"


# Prediction function
def predict(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    print("Predicted index:", pred_index)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    nutrient = nutrient_deficiency(predicted_class)

    return predicted_class, confidence, nutrient


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            # Ensure upload folder exists
            upload_folder = "static/uploads"
            os.makedirs(upload_folder, exist_ok=True)

            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)

            file.save(filepath)
            print("Saved:", filepath)  # debug

            disease, confidence, nutrient = predict(filepath)

            return render_template("index.html",
                                   prediction=disease,
                                   confidence=confidence,
                                   nutrient=nutrient,
                                   image=filepath)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)