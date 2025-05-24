# models/vocab.py
from collections import Counter

class Vocab:
    def __init__(self, counter: Counter, max_size: int = 20000, specials=None):
        specials = specials or ["<unk>", "<pad>"]
        most_common = [t for t,_ in counter.most_common(max_size)]
        self.itos = specials + most_common
        self.stoi = {t:i for i,t in enumerate(self.itos)}
        self.unk_index = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token: str):
        return self.stoi.get(token, self.unk_index)

    @classmethod
    def from_dicts(cls, stoi, itos):
        obj = cls.__new__(cls)
        obj.stoi = stoi
        obj.itos = itos
        obj.unk_index = stoi["<unk>"]
        return obj
