import json
from collections import Counter
import string
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix((remove_punc(lower(s))))


def overlap_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    else:
        return 1

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def precision_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision

def recall_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return recall

def cover_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return ground_truth in prediction

def metric_max_recall(metric_fn, prediction, ground_truths):
    score_recall = []
    for ground_truth in ground_truths:
        score_ground_truth = []
        for predict in prediction:
            score = metric_fn(predict, ground_truth)
            score_ground_truth.append(score)
        score_recall.append(score_ground_truth)
    # print(score_recall) TODO: have empty score?
    try:
        return max(max(score_recall))
    except ValueError:
        return 0

def evaluate(dataset, predictions):
    sentence_cover = precision = cover = sentence_recall = recall = f1 = exact_match = total = overlap = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + str(qa['id']) + ' will receive score 0.'
                    print(message)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = [predictions[qa['id']]]
                #prediction_sentence = predictions[qa['id']]['sentences']
                cover += metric_max_recall(cover_score, prediction, ground_truths)
                exact_match += metric_max_recall(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_recall(
                    f1_score, prediction, ground_truths)
                overlap += metric_max_recall(
                    overlap_score, prediction, ground_truths)
                precision += metric_max_recall(
                    precision_score, prediction, ground_truths)
                recall += metric_max_recall(
                    recall_score, prediction, ground_truths)
                #sentence_recall += metric_max_recall(recall_score, prediction_sentence, ground_truths)
                #sentence_cover += metric_max_recall(cover_score, prediction_sentence, ground_truths)
    print("total: {}".format(total))
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    recall = 100.0 * recall / total
    overlap = 100.0 * overlap / total
    cover = 100.0 * cover / total
    precision = 100.0 * precision / total
    #sentence_recall = 100.0 * sentence_recall / total
    #sentence_cover = 100.0 * sentence_cover / total

    return {'exact_match': exact_match, 'f1': f1, "recall": recall, 
            #"sentence_recall": sentence_recall, "sentence_cover": sentence_cover,
            "precision": precision, "cover": cover, "overlap": overlap}

def squad_v1_eval(dataset_filename, prediction_filename):
    with open(dataset_filename) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    with open(prediction_filename) as prediction_file:
        predictions = json.load(prediction_file)
    ans = evaluate(dataset, predictions)
    return ans
