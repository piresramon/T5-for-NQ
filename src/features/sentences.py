"""Methods to obtain and check validity of T5 input/output sentences."""
import numpy as np
import pandas as pd
import re
from typing import List, Tuple, Dict

SENTENCE_ID_PATTERN = r'\[SENT(.*?)\]'
SUBANSWER_PATTERN = r'([^[\]]+)(?:$|\[)'
TYPE_NAME_PATTERN = r'\[([A-Za-záàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑºª_ \/]*?)\]'

SENT_TOKEN = ' [SENT{}] '
T5_RAW_CONTEXT = str

# Type of a sentence that may have T5 identification tokens
# Example: '[SENT1] Campinas'
T5_SENTENCE = str


def _has_text(string: str) -> bool:
    """Returns True if a string has non whitespace text."""
    string_without_whitespace = string.strip()
    return len(string_without_whitespace) > 0


def _clean_sub_answer(sub_answer: str) -> str:
    """Removes undesired characters from a sub answer.

    Removes any `:` and whitespace the subanswer.
    """
    sub_answer = sub_answer.replace(':', '')
    sub_answer = sub_answer.strip()

    return sub_answer


def _find_sub_answers(prediction_str: str) -> List[str]:
    """Returns a list containing the sub answers of a T5 sentence in the order
    they appear.

    Examples:
        >>> sentence = '[SENT25] [Tipo de Logradouro]: Rua [SENT25] [Logradouro]: PEDRO BIAGI'
        >>> sub_answers = _find_sub_answers(sentence)
        >>> print(sub_answers)
        ['Rua', 'PEDRO BIAGI']
    """
    sub_answer_list = []
    for sub_answer in re.findall(SUBANSWER_PATTERN, prediction_str):
        if _has_text(sub_answer):
            sub_answer = _clean_sub_answer(sub_answer)
            sub_answer_list.append(sub_answer)

    return sub_answer_list


def _find_ids_of_sent_tokens(sentence: T5_SENTENCE) -> List[int]:
    """Returns a list containing the IDs of the SENT tokens if a T5 sentence in
    the order they appear.

    The ID is the number that follows a SENT token.

    Examples:
        >>> sentence = '[SENT1] Campinas'
        >>> ids = _find_ids_of_sent_tokens(sentence)
        >>> print(ids)
        [1]
    """
    ids = []
    for sentid in re.findall(SENTENCE_ID_PATTERN, sentence):
        ids.append(int(sentid))

    return ids


def _find_type_names(sentence: T5_SENTENCE) -> List[str]:
    """Returns a list containing the names of the type tokens of a T5 sentence
    in the order they appear.

    The name is the text that appears inside the type token.

    Examples:
        >>> sentence = '[Logradouro] Campinas'
        >>> type_names = _find_type_names(sentence)
        >>> print(type_names)
        ['Logradouro']
    """
    type_names = re.findall(TYPE_NAME_PATTERN, sentence)

    return type_names


def split_context_into_sentences(
        context: T5_RAW_CONTEXT
) -> List[str]:
    """Splits a question context into multiple questions.

    The criteria of splitting is simply every linebreak found.
    """
    return context.split('\n')


def split_t5_sentence_into_components(
        sentence: T5_SENTENCE,
        map_type: bool = True
) -> Tuple[List[int], List[str], List[str]]:
    """Splits the string outputed by T5 into its components.

    If no occurrences are found of a component, returns an empty list for it.

    Components:
        - sent ids: the ID that follows a SENT token.
        - type names: the name inside a answer type token.
        - sub answers: each answer fragment found.
    Args:
        sentence: a T5 output sentence.

    Examples:
        >>> sentence = '[SENT25] [Tipo de Logradouro]: Rua [SENT25] [Logradouro]: PEDRO BIAGI [SENT26] [Número]: 462 [SENT25] [Cidade]: Sertãozinho [SENT0] [Estado]: SP'
        >>> sent_ids, type_names, sub_answers = \
        >>>     split_t5_sentence_into_components(sentence)
        >>> print(sent_ids)
        [25, 25, 26 25, 0]
        >>> print(type_names)
        ['tipo_de_logradouro', 'logradouro', 'numero', 'cidade', 'estado']
        >>> print(sub_answers)
        ['Rua', 'PEDRO BIAGI', '462', 'Sertãozinho', 'SP'])

    Returns:
        Sentence ids, type names, answers/sub-answers
    """
    sent_ids = _find_ids_of_sent_tokens(sentence)
    type_names = _find_type_names(sentence, map_type=map_type)
    sub_answers = _find_sub_answers(sentence)

    return sent_ids, type_names, sub_answers


def check_sent_id_is_valid(
        context: T5_RAW_CONTEXT, sent_id: int
) -> bool:
    """Returns True if a SENT ID is valid.

    An ID is valid when it corresponds to the ID of a sentence or its ID is 0.
    """
    if sent_id < 0:
        return False

    sentences = split_context_into_sentences(context)

    if len(sentences) < sent_id:
        return False

    return True


def group_qas(example_ids: List[str]) -> Dict[str, List[int]]:
    """Groups the sentences according to qa-ids of the examples.

    Args:
        sentences: List of qa-ids (strings)

    Returns:
        Dict with qa-ids (document-type + type-name) as keys and list of indexes
        of grouped sentences as values.
    """
    qid_dict = {}
    for idx, example_id in enumerate(example_ids):
        if example_id in qid_dict.keys():
            qid_dict[example_id].append(idx)
        else:
            qid_dict[example_id] = [idx]

    return qid_dict


def deconstruct_answer(
    answer_sentence: T5_SENTENCE = ''
) -> Tuple[List[T5_SENTENCE], List[str]]:
    """Gets individual answer subsentences from the compound answer sentence.

    Args:
        answer sentence: a T5 output sentence.

    Examples:
        >>> sentence = '[SENT25] [Tipo de Logradouro]: Rua [SENT25] [Logradouro]: PEDRO BIAGI [SENT26] [Número]: 462 [SENT25] [Cidade]: Sertãozinho [SENT0] [Estado]: SP [aparece no texto] s paulo'
        >>> sub_sentences, type_names = deconstruct_answer(sentence)
        >>> print(sub_sentences)
        [
            '[SENT25] [tipo_de_logradouro] Rua', 
            '[SENT25] [logradouro] PEDRO BIAGI',
            '[SENT26] [numero] 462',
            '[SENT25] [cidade] Sertãozinho',
            '[SENT0] [estado] SP [aparece no texto] s paulo'
        ]
        >>> print(type_names)
        ['tipo_de_logradouro', 'logradouro', 'numero', 'cidade', 'estado']
        
    Returns:
        sub-ansers and type-names
    """
    sent_ids, type_names, sub_answers = split_t5_sentence_into_components(answer_sentence)
    sub_sentences = []
    all_type_names = []

    while len(sub_answers) > 0:
        sub_sentence = '' 

        if len(sent_ids) > 0:
            sent_id = sent_ids.pop(0)
            sentence_token = SENT_TOKEN.format(sent_id).strip()
            sub_sentence += sentence_token + ' '

        if len(type_names) > 0:
            type_name = type_names.pop(0)
            sub_sentence += f'[{type_name}]: '
            all_type_names.append(type_name)

        sub_answer = sub_answers.pop(0)
        sub_sentence += f'{sub_answer} '

        sub_sentences.append(sub_sentence.strip())

    return sub_sentences, all_type_names


def get_subanswer_from_subsentence(
    label_ss: T5_SENTENCE, pred_ss: T5_SENTENCE = ''
    ) -> Tuple[T5_SENTENCE, T5_SENTENCE]:
    """Get only the sub-answer from the current label/predicion subsentence.

    Args:
        label_ss: a T5 label subsentence.
        pred_ss: a T5 output subsentence.

    Examples:
        >>> label_ss = [SENT1] [no_da_matricula]: 88975
        >>> pred_ss = [SENT1] [no_da_matricula] 88975 [aparece no texto] 88.975
        >>> label_ss_, pred_ss_ = get_subanswer_from_subsentence(label_ss, pred_ss)
        >>> print(label_ss_)
        [no_da_matricula]: 88975
        >>> print(pred_ss_)
        [no_da_matricula]: 88975
        
    Returns:
        label and prediction subsentences without SENT_TOKEN and COMPLEMENT_TYPE

    """
    lab_sid, lab_tn, lab_ans = split_t5_sentence_into_components(label_ss, map_type=False)
    # if len(lab_sid) == 1 or COMPLEMENT_TYPE in lab_tn:
    if len(lab_sid) == 1:
        if len(lab_tn) == 0:
            label_ss_ = lab_ans[0]
        else:
            label_ss_ = f'[{lab_tn[0]}]: {lab_ans[0]}'

        if pred_ss == '':
            pred_ss_ = ''
        else:
            _, pred_tn, pred_ans = split_t5_sentence_into_components(pred_ss, map_type=False)
            if len(pred_tn) == 0:
                pred_ss_ = pred_ans[0]
            else:
                pred_ss_ = f'[{pred_tn[0]}]: {pred_ans[0]}'
        
        return label_ss_, pred_ss_
    return '', ''


def split_compound_labels_and_predictions(
    labels: List[T5_SENTENCE], predictions: List[T5_SENTENCE], document_ids: List[str],
    example_ids: List[str], probs: List[float], keep_original_compound: bool = True
    ) -> Tuple[List[T5_SENTENCE], List[T5_SENTENCE], List[str], List[str], List[float], List[int]]:
    """Splits compound answers as individual subsentences (complete sub-anwers) 
    like \"[SENT1] [Estado]: SP [aparece no texto]: São Paulo\" extending 
    original label and prediction sets.

    The function keeps for predictions only the first occurrence of the
    type-names that compose the labels.

    This is useful in inference for getting individual metrics for each
    subsentence that composes a compound answer.

    Examples:
        >>> labels = ['[Tipo de Logradouro]: Rua [Logradouro]: Abert Einstein']
        >>> predictions = ['[Tipo de Logradouro]: Rua [Logradouro]: 41bert Ein5tein [Bairro]: Cidade Universitária']
        >>> labels, predictions, document_ids, example_ids, probs, _ = \
        >>>     split_compound_labels_and_predictions(labels, predictions, ['doc_1'], ['matriculas.endereco'], [0.98], True)
        >>> print(labels)
        ['[tipo_de_logradouro]: Rua [logradouro]: Abert Einstein', '[tipo_de_logradouro]: Rua', '[logradouro]: Abert Einstein']
        >>> print(predictions)
        ['[tipo_de_logradouro]: Rua [logradouro]: 41bert Ein5tein [bairro]: Cidade Universitária', '[tipo_de_logradouro]: Rua', '[logradouro]: 41bert Ein5tein']
        >>> print(document_ids)
        ['doc_1', 'doc_1', 'doc_1']
        >>> print(example_ids)
        ['matriculas.endereco', 'matriculas.endereco~tipo_de_logradouro', 'matriculas.endereco~logradouro']
        >>> print(probs)
        [0.98, 0.0, 0.0]
        
    Returns:
        labels, predictions, document_ids, example_ids, probs
    """
    labels_new, predictions_new = [], []
    document_ids_new, example_ids_new, probs_new = [], [], []
    original_idx = []

    for label, prediction, doc_id, ex_id, prob in zip(labels, predictions, document_ids, example_ids, probs):
        label_subsentences, label_type_names = deconstruct_answer(label)
        prediction_subsentences, prediction_type_names = deconstruct_answer(prediction)

        # this is not compound answer, then get the original label/predicion pair
        if len(label_type_names) <= 1 or keep_original_compound:
            label = ' '.join(label_subsentences)
            prediction = ' '.join(prediction_subsentences)

            document_ids_new.append(doc_id)
            example_ids_new.append(ex_id)
            probs_new.append(prob)
            labels_new.append(label)
            predictions_new.append(prediction)

            # indexes to compute the f1 and exact ONLY with original (non-splitted) answers
            if keep_original_compound:
                idx = len(labels_new) - 1
                original_idx.append(idx)

            if len(label_type_names) <= 1:
                # remove sent-id and raw-text complement, if the label has,
                # in order to get metric only for the response per se
                label_ss_, pred_ss_ = get_subanswer_from_subsentence(label, prediction)
                if label_ss_ != '':
                    ex_id_ = ex_id + '*'
                    document_ids_new.append(doc_id)
                    example_ids_new.append(ex_id_)
                    probs_new.append(0.0)
                    labels_new.append(label_ss_)
                    predictions_new.append(pred_ss_)

        if len(label_type_names) > 1:
            for label_ss, label_tn in zip(label_subsentences, label_type_names):

                try:
                    # the same type-name was predicted, get the first occurrence
                    pred_idx = prediction_type_names.index(label_tn)
                    pred_ss = prediction_subsentences[pred_idx]
                except:
                    # the same type-name was not predicted, use empty
                    pred_ss = ''

                ex_id_ = ex_id + '~' + label_tn
                document_ids_new.append(doc_id)
                example_ids_new.append(ex_id_)
                probs_new.append(0.0)
                labels_new.append(label_ss)
                predictions_new.append(pred_ss)
                
                # remove sent-id and raw-text complement, if the label has,
                # in order to get metric only for the response per se
                label_ss_, pred_ss_ = get_subanswer_from_subsentence(label_ss, pred_ss)
                if label_ss_ != '':
                    ex_id_ = ex_id + '~' + label_tn + '*'
                    document_ids_new.append(doc_id)
                    example_ids_new.append(ex_id_)
                    probs_new.append(0.0)
                    labels_new.append(label_ss_)
                    predictions_new.append(pred_ss_)

    return labels_new, predictions_new, document_ids_new, example_ids_new, probs_new, original_idx


def get_highest_probability_window(
    labels: List[T5_SENTENCE], predictions: List[T5_SENTENCE], 
    document_ids: List[str], example_ids: List[str], probs: List[float],
    use_fewer_NA: bool = False
    ) -> Tuple[List[T5_SENTENCE], List[T5_SENTENCE], List[str], List[str], List[float]]:
    """Get the highest-probability components for each pair document, example.
    """
    if use_fewer_NA:
        na_cases = [ pred.count('N/A') for pred in predictions ]
        arr = np.vstack([np.array(labels), np.array(predictions),
                        np.array(document_ids), np.array(example_ids),
                        np.array(na_cases), np.array(probs)]).transpose()
        df1 = pd.DataFrame(arr, columns=['labels', 'predictions', 'document_ids', 'example_ids', 'na', 'probs'])
        
        # get the highest-probability sample among cases with fewer number if N/As
        # for each pair document-id / example-id
        df1 = df1.sort_values(['na', 'probs'], ascending=[True, False]).groupby(['document_ids', 'example_ids']).head(1)
        df1.sort_index(inplace=True)

        labels, predictions, document_ids, example_ids, _, probs = df1.T.values.tolist()

    else:
        arr = np.vstack([np.array(labels), np.array(predictions),
                        np.array(document_ids), np.array(example_ids),
                        np.array(probs)]).transpose()
        df1 = pd.DataFrame(arr, columns=['labels', 'predictions', 'document_ids', 'example_ids', 'probs'])

        # get the highest-probability sample for each pair document-id / example-id
        df1 = df1.sort_values('probs', ascending=False).groupby(['document_ids', 'example_ids']).head(1)
        df1.sort_index(inplace=True)

        labels, predictions, document_ids, example_ids, probs = df1.T.values.tolist()

    return labels, predictions, document_ids, example_ids, probs

    
def main():
    print('>> Testing deconstruct_answer...')
    sentence = '[Tipo de Logradouro]: Rua [aparece no texto]: rua [Logradouro]: Abert Einstein'
    sentence = '[SENT25] [Tipo de Logradouro]: Rua [SENT25] [Logradouro]: PEDRO BIAGI [SENT26] [Número]: 462 [SENT25] [Cidade]: Sertãozinho [SENT0] [Estado]: SP [aparece no texto] s paulo'

    subsentences, type_names = deconstruct_answer(sentence)
    print('inp: ', sentence)
    print('out: ', subsentences)

    print('>> Testing split_compound_labels_and_predictions...')
    labels = [
        '[SENT25] [Tipo de Logradouro]: Rua [SENT25] [Logradouro]: PEDRO BIAGI [SENT26] [Número]: 462 [SENT25] [Cidade]: Sertãozinho [SENT0] [Estado]: SP [aparece no texto] s paulo',
        'rondonia',
        '[SENT1] [Nº da Matrícula]: 88975',
        '[tipo de publicação]: Tipo 1 [órgão]: orgao 1',
        '[Tipo de Logradouro]: Rua [aparece no texto] rua [Logradouro]: Albert Einstein',
    ]
    predictions = [
        '[SENT25] [Cidade]: Sertãozinho [SENT0] [Estado]: SP [aparece no texto] s paulo [SENT1] [Complemento]: 13098-982',
        '[SENT1] rondonia [SENT2] RO',
        '[SENT1] [Nº da Matrícula]: 88975 [aparece no texto] 88.975',
        '[tipo de publicação]: Tipo 1 [órgão]: orgao 2',
        '[Tipo de Logradouro]: Rua [aparece no texto] rua [Logradouro]: 41bert Ein5tein [Bairro]: Cidade Universitária',
    ]
    document_ids = ['12345', '56789', '65789', '112233', '2522']
    example_ids = ['matriculas.endereco', 'matriculas.estado', 'matriculas.no_da_matricula', 'publicacoes.tipo_orgao', 'matriculas.endereco']
    probs = [0.71, 0.98, 0.93, 0.95, 0.89]

    lab, pred, doc, ex, prob, _ = split_compound_labels_and_predictions(labels, predictions, document_ids, example_ids, probs, keep_original_compound=False)

    for l, p, d, e, s in zip(lab, pred, doc, ex, prob):
        print(f'{d:10} | {e:50} | {l:60} | {p} ({s})')

    print('>> Testing get_subanser_from_subsentence with only label sentence...')
    label = '[SENT1] [cidade] Tanhaçu'
    label_, _ = get_subanswer_from_subsentence(label)
    print('inp: ', label)
    print('out: ', label_)


if __name__ == '__main__':
    main()
