from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Image description API is running!"})

@app.route("/upload", methods=["POST"])
def upload_image():
    # Check if the image is in the request
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected image"}), 400

    try:
        # Open the image
        image = Image.open(image_file)

        # Generate a description
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        description = processor.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"description": description}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
