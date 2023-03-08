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
    

def train_probe(pos_feats, neg_feats, epochs=1000, lr=0.01):
    model = Probe()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch}...")

        average_loss = 0
        for i in range(pos_feats.shape[0]):
            pos_feat = pos_feats[i]
            neg_feat = neg_feats[i]

            batch = torch.stack((pos_feat, neg_feat), dim=0)
            probs = model(batch)
            if i == 0:
                print(probs)
            loss = loss_fn(probs)
            average_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        average_loss /= pos_feats.shape[0]
        print(f"Average loss: {average_loss}")
    
    # Save the model
    torch.save(model.state_dict(), 'probe.pt')


if __name__ == "__main__":
    pos_feats, neg_feats = load_features()
    # pos feats shape is num_examples x d_model

    train_probe(torch.tensor(pos_feats), torch.tensor(neg_feats))



