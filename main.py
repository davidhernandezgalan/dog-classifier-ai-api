from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor
import PIL
import requests
import os
import json
import torch

# Load env variables for development or production
load_dotenv()

# Dog breeds model
image_processor = AutoImageProcessor.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")
model = AutoModelForImageClassification.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")

# NSFW model
nsfw_image_processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

app = FastAPI()

# Config CORS middleware
origins = ["*" if os.getenv("DEVELOPMENT") == "true" else "https://doggyfinder.netlify.app/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return { "status": "API is running", "development": os.getenv("DEVELOPMENT") }

@app.post("/breed")
def breed(data: dict = Body(...)):
    try:
        if type(data) is not dict or "url" not in data:
            return { "error": True, "message": "No se ha proporcionado una URL de imagen." }
        
        image = PIL.Image.open(requests.get(data["url"], stream=True).raw)
        inputs = image_processor(images=image, return_tensors="pt")

        outputs = model(**inputs)
        logits = outputs.logits

        # model predicts one of the 120 Stanford dog breeds classes
        predicted_class_idx = logits.argmax(-1).item()
    except Exception as error:
        print(error)
        return { "error": True, "message": "Ocurrió un error al clasificar la imagen." }
    
    return { "error": False, "breed": model.config.id2label[predicted_class_idx]}

@app.post("/nsfw")
def nsfw(data: dict = Body(...)):
    try:
        if type(data) is not dict or "url" not in data:
            return { "error": True, "message": "No se ha proporcionado una URL de imagen." }
        
        image = PIL.Image.open(requests.get(data["url"], stream=True).raw)
        with torch.no_grad():
            inputs = nsfw_image_processor(images=image, return_tensors="pt")
        
            outputs = nsfw_model(**inputs)
            logits = outputs.logits

        # model predicts if the image is NSFW or not
        predicted_class_idx = logits.argmax(-1).item()
    except Exception as error:
        print(error)
        return { "error": True, "message": "Ocurrió un error al clasificar la imagen." }
    
    result = True if nsfw_model.config.id2label[predicted_class_idx] == "nsfw" else False
    return { "error": False, "nsfw": result }

@app.post("/search_dog")
def search_dog(data: dict = Body(...)):
    try:
        if type(data) is not dict or "url" not in data:
            return { "error": True, "message": "No se ha proporcionado una URL de imagen." }
        
        datos = requests.post("https://api.clarifai.com/v2/users/clarifai/apps/main/models/general-image-detection/versions/1580bb1932594c93b7e2e04456af7c6f/outputs",
            headers={
                "Authorization": "Key " + os.getenv("CLARIFAI_API_KEY"),
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "inputs": [{
                    "data": {
                        "image": { "url": data["url"] }
                    }
                }]
            })
        )
        datos = datos.json()

        # Lista de objetos encontrados en la imagen
        regionesClasificadas = datos["outputs"][0]["data"]["regions"]

        includesDog = True in [region["data"]["concepts"][0]["name"].lower() == "dog" for region in regionesClasificadas]

        return { "error": False, "includes_dog": includesDog }
    except Exception as error:
        print(error)
        return { "error": True, "message": "Ocurrió un error al encontrar un perro en la imagen." }
