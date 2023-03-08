import os

import torch
import torch.nn as nn
import numpy as np


def load_features(path='.'):
    neg_path = os.path.join(path, 'neg_feats.npy')
    pos_path = os.path.join(path, 'pos_feats.npy')

    neg_feats = np.load(neg_path)
    pos_feats = np.load(pos_path)

    return pos_feats, neg_feats


class Probe(nn.Module):
    def __init__(self, d_model=768):
        super(Probe, self).__init__()
        self.fc1 = nn.Linear(d_model, 1)

    def forward(self, x):
        """ outputs a probability """
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
    

def loss_fn(probs):
    assert probs.shape == (2, 1)

    p0 = probs[0][0]
    p1 = probs[1][0]

    l_consistency = (p0 - (1 - p1)) ** 2
    l_confidence = torch.min(p0, p1) ** 2

    return l_consistency + l_confidence
    

def train_probe(pos_feats, neg_feats, epochs=10, lr=0.001):
    model = Probe()
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch}...")

        for i in range(pos_feats.shape[0]):
            pos_feat = pos_feats[i]
            neg_feat = neg_feats[i]

            batch = torch.stack((pos_feat, neg_feat), dim=0)
            print(batch.shape)
            probs = model(batch)
            print(probs)


if __name__ == "__main__":
    pos_feats, neg_feats = load_features()
    print(pos_feats.shape)
    print(neg_feats.shape)

    # pos feats shape is num_examples x d_model

    train_probe(torch.tensor(pos_feats), torch.tensor(neg_feats))



