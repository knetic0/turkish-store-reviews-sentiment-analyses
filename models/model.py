import importlib
import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import string
import nltk
from nltk.corpus import stopwords
import snowballstemmer
from models.vocab import Vocab
import json

stemmer = snowballstemmer.stemmer("turkish")
nltk.download("stopwords")
turkish_stopwords = set(stopwords.words("turkish"))

class Model:
    def __init__(self, model_dir_name: str):
        self.model_dir_name = model_dir_name
        self.model_dir      = os.path.join("models", model_dir_name)
        self.model_path     = os.path.join(self.model_dir, "model.pth")
        self.stoi_path      = os.path.join(self.model_dir, "stoi.pkl")
        self.itos_path      = os.path.join(self.model_dir, "itos.pkl")
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map      = { "Positive": 0, "Notr": 1, "Negative": 2 }
        self.config_path    = os.path.join(self.model_dir, "config.json")

        with open(self.stoi_path, "rb") as f:
            stoi = pickle.load(f)
        with open(self.itos_path, "rb") as f:
            itos = pickle.load(f)
        
        self.vocab = Vocab.from_dicts(stoi, itos)

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

    def load(self):
        module_path = f"models.{self.model_dir_name}.algorithm"
        alg_mod     = importlib.import_module(module_path)
        AlgClass    = getattr(alg_mod, "Algorithm")

        self.model = AlgClass(
            vocab_size = len(self.vocab),
            emb_dim     = self.config.get("emb_dim", 100),
            hid_dim     = self.config.get("hid_dim", 128),
            out_dim     = len(self.label_map),
            pad_idx     = self.vocab["<pad>"],
            n_layers    = self.config.get("n_layers", 1),
            dropout     = self.config.get("dropout", 0.5),
        ).to(self.device)

        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, raw_text: str):
        cleaned_text = self.clean_text(raw_text)
        tokens       = cleaned_text.split()
        idxs         = [self.vocab[t] for t in tokens]

        seq_tensor = torch.tensor(idxs, dtype=torch.long)
        padded     = pad_sequence([seq_tensor], batch_first=True,
                                  padding_value=self.vocab["<pad>"]).to(self.device)
        lengths    = torch.tensor([len(idxs)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits  = self.model(padded, lengths)
            pred_id = logits.argmax(dim=1).item()

        return {v:k for k,v in self.label_map.items()}.get(pred_id, str(pred_id))

    def __repr__(self):
        return f"<Model dir={self.model_dir_name}>"

    def clean_text(self, text: str) -> str:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        words = [
            stemmer.stemWord(tok)
            for tok in text.split()
            if tok not in turkish_stopwords and not tok.isdigit()
        ]
        return " ".join(words)
