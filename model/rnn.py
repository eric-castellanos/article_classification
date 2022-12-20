from functools import partial

import torch
import torch.nn as nn
import torchtext
import pandas as pd
from torch.nn import functional as F
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import gc
from torch.optim import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Reproducing same results
SEED = 2019

#Torch
torch.manual_seed(SEED)

#Cuda algorithms
torch.backends.cudnn.deterministic = True  

embed_len = 50
hidden_dim = 50
n_layers=1

tokenizer = get_tokenizer("basic_english")

def build_vocabulary(datasets):
    for dataset in datasets:
        for text in dataset:
            yield tokenizer(text)

def vectorize_batch(batch, max_words, vocab):
    #Y, X = list(zip(batch))
    Y, X = list(zip(*batch))
    X = [vocab(tokenizer(text)) for text in X]
    #X = X.apply(lambda text: vocab(tokenizer(text)))
    X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X] ## Bringing all samples to max_words length.

    return torch.tensor(X, dtype=torch.int32), torch.tensor(Y) - 1 ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]

def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for X, Y in val_loader:
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

            Y_shuffled.append(Y)
            Y_preds.append(preds.argmax(dim=-1))

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))


def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    for i in range(1, epochs+1):
        losses = []
        for X, Y in tqdm(train_loader):
            Y_preds = model(X)

            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        CalcValLossAndAccuracy(model, loss_fn, val_loader)

    torch.save(model, 'rnn_article_classifier.pt')

def MakePredictions(model, loader):
    Y_shuffled, Y_preds = [], []
    for X, Y in loader:
        preds = model(X)
        Y_preds.append(preds)
        Y_shuffled.append(Y)
    gc.collect()
    Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

    return Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe[['Class Index', 'Description']]
        self.label = dataframe['Class Index']
        self.description = dataframe['Description']

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]

class RNNClassifier(nn.Module):
    def __init__(self, vocab, target_classes):
        super(RNNClassifier, self).__init__()
        self.vocab = vocab
        self.target_classes = target_classes
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, len(self.target_classes))

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings, torch.randn(n_layers, len(X_batch), hidden_dim))
        return self.linear(output[:,-1])

def main():
    train = pd.read_csv('../data/processed/train.csv')
    test = pd.read_csv('../data/processed/test.csv')

    target_classes = ["World", "Sports", "Business", "Sci/Tech"]

    max_words = 25

    vocab_description = build_vocab_from_iterator(build_vocabulary([train['Description'], test['Description']]), min_freq=1, specials=["<UNK>"])

    vocab_description.set_default_index(vocab_description["<UNK>"])

    train = PandasDataset(train)
    test = PandasDataset(test)

    train_loader_description = DataLoader(train, batch_size=1024, collate_fn=partial(vectorize_batch, max_words=max_words, vocab=vocab_description))
    test_loader_description  = DataLoader(test, batch_size=1024, collate_fn=partial(vectorize_batch, max_words=max_words, vocab=vocab_description))

    rnn_classifier_description = RNNClassifier(vocab_description, target_classes)

    epochs = 15
    learning_rate = 1e-3

    loss_fn = nn.CrossEntropyLoss()
    optimizer_description = Adam(rnn_classifier_description.parameters(), lr=learning_rate)

    TrainModel(rnn_classifier_description, loss_fn, optimizer_description, train_loader_description, test_loader_description, epochs)

    Y_actual, Y_preds = MakePredictions(rnn_classifier_description, test_loader_description)

    print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
    print("\nClassification Report : ")
    print(classification_report(Y_actual, Y_preds, target_names=target_classes))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(Y_actual, Y_preds))

if __name__ == "__main__":

    main() 

