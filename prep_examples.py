import json


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


if __name__ == '__main__':
    data_path = "../../Downloads/boolQ/dev.jsonl"
    # Get the data
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # Get the first question
    q_dict = data[0]

    # Get the contrast pair
    contrast_pair = get_contrast_pair(q_dict)

    # print(contrast_pair)
    print(contrast_pair[0])
    print(contrast_pair[1])
