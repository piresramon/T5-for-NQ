""" Very heavily inspired by the official evaluation script for SQuAD version
2.0 which was modified by XLNet authors to update `find_best_threshold`
scripts for SQuAD V2.0 In addition to basic functionality, we also compute
additional statistics and plot precision-recall curves if an additional
na_prob.json file is provided. This file is expected to map question ID's to
the model's predicted probability that a question is unanswerable. """
import collections
import re
import string
from typing import Dict, Optional

import torch
from pytorch_lightning import metrics


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))



def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_precision_recall(answers, preds):
    predictions, target = [], []
    precision = metrics.classification.Precision(num_classes=1)
    recall = metrics.classification.Recall(num_classes=1)
    for i, (a_gold, a_pred) in enumerate(zip(answers, preds)):

        gold = normalize_answer(a_gold)
        pred = normalize_answer(a_pred)

        target.append(1 if 'sim' in gold else 0)  # assuming if SIM is not part of answer, class is negative
        predictions.append(1 if 'sim' in pred else 0)   # assuming if SIM is not part of answer, class is negative

    predictions = torch.tensor(predictions)
    target = torch.tensor(target)

    return precision(predictions, target).item(), recall(predictions, target).item()
    

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact",
                 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def get_raw_scores(answers, preds):
    """Computes the exact and f1 scores from the examples and the model
    predictions.

    This version gets the answers and prediction in text format, as T5 returns.
    """
    exact_scores = {}
    f1_scores = {}

    for i, (answer, pred) in enumerate(zip(answers, preds)):
        exact_scores[i] = compute_exact(answer, pred)
        f1_scores[i] = compute_f1(answer, pred)

    return exact_scores, f1_scores


def t5_qa_evaluate(answers, preds, qid_dict: Optional[Dict] = None):
    """Evaluates T5 predictions.

    This is a siplification of `square_evaluate` to compute the exact and f1
    scores from predictions from T5.
    If required, this version returns subdicts with f1 and exact measures for
    pre-selected groups of question-answers.

    Examples:
        >>> qid_dict = {
        >>>     'matriculas': [0, 4],
        >>>     'comarca': [1, 4],
        >>>     'estado': [2, 6]
        >>>     'oficio': [3, 7]
        >>> }
        >>> t5_qa_evaluate(answers, preds, qid_dict=qid_dict)
        >>> {'exact': x, 'f1': y, 'total': 8, 'matriculas': {'exact': z, 'f1': w, 'total': 2}, ... }
    """
    if qid_dict is None:
        qid_dict = {}

    exact, f1 = get_raw_scores(answers, preds)
    evaluation = make_eval_dict(exact, f1)

    for (kword, qid_list) in qid_dict.items():
        evaluation[kword] = make_eval_dict(exact, f1, qid_list)

        answers_kw = [answers[i] for i in qid_list]
        preds_kw = [preds[i] for i in qid_list]

        precision, recall = compute_precision_recall(answers_kw, preds_kw)
        f1_measure = (2 * precision * recall) / (precision + recall)
        evaluation[kword]['precision'] = precision
        evaluation[kword]['recall'] = recall
        evaluation[kword]['f1-measure'] = f1_measure

    return evaluation
