import json
import torch
import numpy as np

from transformers import RobertaTokenizer, RobertaModel

from probe import Probe
from prep_examples import get_contrast_pair, contrast_features


def inference_q(q_dict, model, probe, tokenizer, layer=10):
    p1, p2, _ = get_contrast_pair(q_dict)
    pos_feats, neg_feats = contrast_features(p1, p2, tokenizer, model, layer=layer)

    # Normalize the features
    pos_feats = (pos_feats - np.load('pos_means.npy')) / np.load('pos_stds.npy')
    neg_feats = (neg_feats - np.load('neg_means.npy')) / np.load('neg_stds.npy')

    with torch.no_grad():
        pos_probs = probe(torch.tensor(pos_feats))
        neg_probs = probe(torch.tensor(neg_feats))

    pos_probs = pos_probs.cpu().detach().numpy()[0]
    neg_probs = neg_probs.cpu().detach().numpy()[0]

    print(pos_probs, neg_probs)

    res = 0.5*(pos_probs + (1-neg_probs))
    if res > 0.5:
        return True
    else:
        return False


if __name__=="__main__":
    # test_path = "../boolQ/dev.jsonl"
    test_path = "../boolQ/balanced_train.jsonl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large')
    model.to(device)
    model.eval()

    probe = Probe()
    probe.load_state_dict(torch.load('probe.pt'))
    probe.eval()

    print(probe)

    with open(test_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    correct = 0
    for idx, q_dict in enumerate(data):
        if idx > 100:
            break
        try:
            res = inference_q(q_dict, model, probe, tokenizer, layer=10)
            print('pred:', res, 'answer:', q_dict['answer'])
            if res == q_dict['answer']:
                correct += 1
        except ValueError:
            print("Input too long!")
            continue
        
    print(f"Accuracy: {correct/(idx+1)}")

