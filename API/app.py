import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import torch

app = FastAPI()

def load_models(model):
    """
    load the models from disk
    and put them in a dictionary
    Returns:
        dict: loaded models
    """
    models = {
        "rnn": torch.load("../model/rnn_article_classifier.pt"),
    }
    print("models loaded from disk")
    return models[model]

class RequestBody(BaseModel):
    description : str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(data : RequestBody):
    model = load_models("rnn")
    pred = model(data.description)
    #pred = int(torch.max(output.data, 1)[1].numpy())

    target_classes = { 1 : "World", 2 : "Sports", 3 : "Business", 4 : "Sci/Tech" }
    return { 'Article Classification' : target_classes[pred] }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5049)