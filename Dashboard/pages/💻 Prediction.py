import sys
sys.path.append("../model")

import streamlit as st
import torch
import pickle
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

def load_models(model):
    """
    load the models from disk
    and put them in a dictionary
    Returns:
        dict: loaded models
    """
    models = {
        "rnn": torch.load("../model/models/rnn_article_classifier.pt"),
    }
    #print("models loaded from disk")
    return models[model]

def predict(description):
    model = load_models("rnn")
    
    vocab_description = build_vocab_from_iterator(description, min_freq=1, specials=["<UNK>"])
    vocab_description.set_default_index(vocab_description["<UNK>"])

    tokens = vocab_description(tokenizer(description))
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

    target_classes = {0: "N/A",  1 : "World", 2 : "Sports", 3 : "Business", 4 : "Sci/Tech" }
    return target_classes[pred_class]

def main():
    """
    main function for app
    """

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.title("Enter you article description text:")
        description = st.text_input("Enter your article description here:", placeholder="Article Description")

        if st.button('Classify Article'):
            article_class = predict(description)
            st.success("Predicted Article Category: %s " % article_class, icon="✅")
            #st.success("Go Irish!")
    with col3:
        st.write(' ')