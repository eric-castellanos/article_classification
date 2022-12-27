import sys
sys.path.append("../model")

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn import functional as F

from rnn import RNNClassifier

import __main__
setattr(__main__, "RNNClassifier", RNNClassifier)

tokenizer = get_tokenizer("basic_english")

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
    
    vocab_description = build_vocab_from_iterator(data.description, min_freq=1, specials=["<UNK>"])
    vocab_description.set_default_index(vocab_description["<UNK>"])

    tokens = vocab_description(tokenizer(data.description))
    max_words = 25
    if len(tokens)<max_words:
        tokens+([0]* (max_words-len(tokens))) 
    else: 
        tokens[:max_words]
    pred_text = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)
    pred = model(pred_text)
    pred_class = F.softmax(pred, dim=-1).argmax(dim=-1).detach().numpy()
    pred_class = pred_class[0]
    #pred = int(torch.max(output.data, 1)[1].numpy())

    target_classes = { 1 : "World", 2 : "Sports", 3 : "Business", 4 : "Sci/Tech" }
    return { 'Article Classification' : target_classes[pred_class] }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5049)