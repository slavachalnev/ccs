import json
import torch
import numpy as np

from transformers import RobertaTokenizer, RobertaModel


def get_contrast_pair(q_dict):
    """
    Accepts a question dictionary and returns a contrast pair.

    Contrast pair is a tuple of two strings:
    Passage: [passage]\n\nQuestion: [question]\n\nAnswer: True
    Passage: [passage]\n\nQuestion: [question]\n\nAnswer: False
    """

    # Get the passage and question
    passage = q_dict['passage']
    question = q_dict['question']

    # Get the correct answer
    answer = q_dict['answer']

    p1 = f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer: True"
    p2 = f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer: False"
    
    return p1, p2, answer


def contrast_features(true_text, false_text, tokenizer, model, layer=-1):
    pos_tok = tokenizer(true_text, return_tensors='pt', truncation=True, padding="max_length").to(model.device)
    neg_tok = tokenizer(false_text, return_tensors='pt', truncation=True, padding="max_length").to(model.device)

    with torch.no_grad():
        pos_feats = model(pos_tok.input_ids, output_hidden_states=True)['hidden_states'][layer][0, -1]
        neg_feats = model(neg_tok.input_ids, output_hidden_states=True)['hidden_states'][layer][0, -1]

    pos_feats = pos_feats.cpu().detach().numpy()
    neg_feats = neg_feats.cpu().detach().numpy()

    return pos_feats, neg_feats


def get_all_feats(data_path, tokenizer, model, layer=-1, max_num=100):
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    print('len data is ', len(data))
    all_feat_pairs = []
    for idx, q_dict in enumerate(data):
        if idx > max_num:
            break
        true_text, false_text, answer = get_contrast_pair(q_dict)

        print(f"Processing question {idx}...")
        feats = contrast_features(true_text, false_text, tokenizer, model, layer=layer)
        all_feat_pairs.append(feats)
    
    return all_feat_pairs


if __name__ == '__main__':
    data_path = "../boolQ/balanced_train.jsonl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', model_max_length=512)
    model = RobertaModel.from_pretrained('roberta-large')
    model.to(device)
    model.eval()

    feat_pairs = get_all_feats(data_path, tokenizer, model, layer=-1, max_num=1000)

    # save as two numpy arrays
    pos_feats = [pair[0] for pair in feat_pairs]
    neg_feats = [pair[1] for pair in feat_pairs]

    # ###### normalise ######
    pos_feats = np.array(pos_feats)
    neg_feats = np.array(neg_feats)

    pos_means = pos_feats.mean(axis=0)
    neg_means = neg_feats.mean(axis=0)
    pos_stds = pos_feats.std(axis=0)
    neg_stds = neg_feats.std(axis=0)

    pos_feats = (pos_feats - pos_means) / pos_stds
    neg_feats = (neg_feats - neg_means) / neg_stds

    np.save("pos_means.npy", pos_means)
    np.save("neg_means.npy", neg_means)
    np.save("pos_stds.npy", pos_stds)
    np.save("neg_stds.npy", neg_stds)

    np.save('pos_feats.npy', pos_feats)
    np.save('neg_feats.npy', neg_feats)
