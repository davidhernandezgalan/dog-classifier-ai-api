from fastapi import FastAPI, Body
from transformers import AutoImageProcessor, AutoModelForImageClassification
import PIL
import requests

image_processor = AutoImageProcessor.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")
model = AutoModelForImageClassification.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")

app = FastAPI()

@app.post("/")
def home(data: dict = Body(...)):
    try:
        image = PIL.Image.open(requests.get(data["url"], stream=True).raw)
        inputs = image_processor(images=image, return_tensors="pt")

        outputs = model(**inputs)
        logits = outputs.logits

        # model predicts one of the 120 Stanford dog breeds classes
        predicted_class_idx = logits.argmax(-1).item()
    except:
        return { "error": True, "message": "Ocurrió un error al clasificar la imagen." }
    
    return { "error": False, "breed": model.config.id2label[predicted_class_idx]}
