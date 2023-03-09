import os
import random

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
    def __init__(self, d_model=1024):
        super(Probe, self).__init__()
        self.fc1 = nn.Linear(d_model, 1)

    def forward(self, x):
        """ outputs a probability """
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
    

def loss_fn(p0, p1):

    l_consistency = (p0 - (1 - p1)) ** 2
    # l_confidence = torch.min(p0**2, p1**2)
    # l_conf = -(p0 * torch.log(p0 + 1e-8) + (1 - p1) * torch.log(1 - p1 + 1e-8))
    l_conf = -(p0 - p1) ** 2

    loss = l_consistency.mean(0) + l_conf.mean(0)
    return loss
    

def train_probe(pos_feats, neg_feats, epochs=1000, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dim = pos_feats.shape[1]
    model = Probe(d_model=model_dim)
    model.to(device)
    model.train()

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        print(f"Epoch {epoch}...")

        pos_batch = pos_feats.to(device)
        neg_batch = neg_feats.to(device)
        p0 = model(pos_batch)
        p1 = model(neg_batch)
        loss = loss_fn(p0, p1)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the model
    torch.save(model.state_dict(), 'probe.pt')


if __name__ == "__main__":
    pos_feats, neg_feats = load_features()
    # pos feats shape is num_examples x d_model

    train_probe(torch.tensor(pos_feats), torch.tensor(neg_feats))



