import json
from transformers import RobertaTokenizer, RobertaModel


def get_contrast_pair(q_dict):
    """
    Accepts a question dictionary and returns a contrast pair.

    Contrast pair is a tuple of two strings:
    Passage: [passage]\n\nQuestion: [question]\n\nA: [answer]
    Passage: [passage]\n\nQuestion: [question]\n\nA: [~answer]
    """

    # Get the passage and question
    passage = q_dict['passage']
    question = q_dict['question']

    # Get the correct answer
    answer = q_dict['answer']

    p1 = f"Passage: {passage}\n\nQuestion: {question}\n\nA: {answer}"
    p2 = f"Passage: {passage}\n\nQuestion: {question}\n\nA: {not answer}"
    
    return (p1, p2)


def contrast_features(true_text, false_text, tokenizer, model, layer=10):
    pos_tok = tokenizer(true_text, return_tensors='pt')
    neg_tok = tokenizer(false_text, return_tensors='pt')

    pos_feats = model(pos_tok.input_ids, output_hidden_states=True)['hidden_states'][layer][0][-1]
    neg_feats = model(neg_tok.input_ids, output_hidden_states=True)['hidden_states'][layer][0][-1]

    pos_feats = pos_feats.cpu().detach().numpy()
    neg_feats = neg_feats.cpu().detach().numpy()

    return pos_feats, neg_feats


if __name__ == '__main__':
    data_path = "../../Downloads/boolQ/dev.jsonl"
    # Get the data
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # Get the first question
    q_dict = data[0]

    # Get the contrast pair
    true_text, false_text = get_contrast_pair(q_dict)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    feats = contrast_features(true_text, false_text, tokenizer, model)
    print(feats)
