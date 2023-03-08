import json
import torch
import numpy as np

from transformers import RobertaTokenizer, RobertaModel


def get_contrast_pair(q_dict):
    """
    Accepts a question dictionary and returns a contrast pair.

    Contrast pair is a tuple of two strings:
    Passage: [passage]\n\nQuestion: [question]\n\nA: [True]
    Passage: [passage]\n\nQuestion: [question]\n\nA: [False]
    """

    # Get the passage and question
    passage = q_dict['passage']
    question = q_dict['question']

    # Get the correct answer
    answer = q_dict['answer']

    p1 = f"Passage: {passage}\n\nQuestion: {question}\n\nA: True"
    p2 = f"Passage: {passage}\n\nQuestion: {question}\n\nA: False"
    
    return (p1, p2)


def contrast_features(true_text, false_text, tokenizer, model, layer=10):
    pos_tok = tokenizer(true_text, return_tensors='pt')
    neg_tok = tokenizer(false_text, return_tensors='pt')

    if pos_tok.input_ids.shape[1] > 512:
        raise ValueError("Input too long!")

    with torch.no_grad():
        pos_feats = model(pos_tok.input_ids, output_hidden_states=True)['hidden_states'][layer][0][-1]
        neg_feats = model(neg_tok.input_ids, output_hidden_states=True)['hidden_states'][layer][0][-1]

    pos_feats = pos_feats.cpu().detach().numpy()
    neg_feats = neg_feats.cpu().detach().numpy()

    return pos_feats, neg_feats


def get_all_feats(data_path, tokenizer, model, layer=10, max_num=100):
    # Get the data
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    all_feat_pairs = []
    for idx, q_dict in enumerate(data):
        if idx > max_num:
            break
        # Get the contrast pair
        true_text, false_text = get_contrast_pair(q_dict)
        print(f"Processing question {idx}...")
        # print(len(true_text))
        # if idx == 55:
        #     print(true_text)
        #     print(false_text)
        try:
            feats = contrast_features(true_text, false_text, tokenizer, model, layer=layer)
        except ValueError:
            print("Input too long!")
            continue
        all_feat_pairs.append(feats)

    return all_feat_pairs


if __name__ == '__main__':
    data_path = "../../Downloads/boolQ/dev.jsonl"

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    model.eval()

    feat_pairs = get_all_feats(data_path, tokenizer, model, layer=10, max_num=100)

    # save as two numpy arrays
    pos_feats = [pair[0] for pair in feat_pairs]
    neg_feats = [pair[1] for pair in feat_pairs]

    # ###### normalise ######
    pos_feats = np.array(pos_feats)
    neg_feats = np.array(neg_feats)

    # subtract mean
    pos_feats -= pos_feats.mean(axis=0)
    neg_feats -= neg_feats.mean(axis=0)

    # divide by std
    pos_feats /= pos_feats.std(axis=0)
    neg_feats /= neg_feats.std(axis=0)

    np.save('pos_feats.npy', pos_feats)
    np.save('neg_feats.npy', neg_feats)
