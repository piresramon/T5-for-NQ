from typing import Dict, List, Union

# from src.features.questions import (
#     COMPLEMENT,
#     QUESTION,
#     QUESTION_DICT,
#     QUESTION_NAME,
#     SUBQUESTION_NAME,
# )
# from src.features.questions import QUESTIONS as ALL_QUESTIONS

# NOME_FRACAO_AREA_COMPLEMENT = ' {}'
# CPJ_FRACAO_AREA_COMPLEMENT = ' CPF/CNPJ {}'
# FULL_FRACAO_AREA_COMPLEMENT = f'{NOME_FRACAO_AREA_COMPLEMENT}, que possui{CPJ_FRACAO_AREA_COMPLEMENT}'


# def get_questions_for_chunk(
#         question_name: QUESTION_NAME, subquestion_name: SUBQUESTION_NAME = None,
#         all_questions: QUESTION_DICT = ALL_QUESTIONS):
#     """Returns all the questions, all the subquestion of a question or a
#     subquestion.

#     Args:
#         question_name: 'all' or name of the question. If 'all', returns a
#             dictionary containing all the possible questions types.
#         subquestion_name: name of a subquestion from the question. If
#             `question_name` is a name and this is None, then returns a
#             dictionary with all the subquestions and their names. If this a
#             name, returns just the subquestion.
#         all_questions: Dictionary with all the questions and subquestions.

#     Examples:
#         >>> questions = {'question1': {'subquestion1': ['What?']}}
#         >>> get_questions_for_chunk('all', all_questions=questions)
#         {'question1': {'subquestion1': ['What?']}}
#         >>> get_questions_for_chunk('question1', all_questions=questions)
#         {'subquestion1': ['What?']}
#         >>> get_questions_for_chunk('question1', 'subquestion1', questions)
#         ['What?']
        
#     Returns:
#         Dictionary with all the questions and its subquestions. Or all the
#         subquestions of a question. Or a subquestion.
#     """
#     if question_name == 'all':
#         return all_questions

#     if subquestion_name is None:
#         return all_questions[question_name]
#     else:
#         return all_questions[question_name][subquestion_name]


# def _check_sentence_intersects(sent: str, list_sents: List[str]) -> bool:
#     """Checks if any sentence in a list (questions) intersects with the sentence
#     (question + context)"""
#     for ssent in list_sents:
#         if ssent in sent:
#             return True
#     return False


# def complement_questions_to_require_rawdata(
#         questions: Union[QUESTION, List[QUESTION]], complement: str = COMPLEMENT
# ) -> Union[QUESTION, List[QUESTION]]:
#     """Add a complementary text to a question or questions.

#     This indicates to the model it must give a subanswer with part of the
#     context's raw text.
#     """
#     if isinstance(questions, str):  # simple question
#         questions = questions.replace('?', complement)
#     if isinstance(questions, list):  # list of questions
#         questions = [q.replace('?', complement) for q in questions]
#     return questions


# def complement_questions_of_proporcao_fracao_area(
#         questions: Union[QUESTION, List[QUESTION]],
#         nome: str = None, cpf_cnpj: str = None) -> List[QUESTION]:
#     """Add a complementary text to a question or questions.

#     This includes additional information to dinamic questions from the specific
#     type-name proporcao_fracao_area.
#     """
#     complements = []
#     if nome is not None and cpf_cnpj is not None:
#         complements.append(FULL_FRACAO_AREA_COMPLEMENT.format(nome, cpf_cnpj) + '?')
#         complements.append(CPJ_FRACAO_AREA_COMPLEMENT.format(cpf_cnpj) + '?')
#     elif nome is not None:
#         complements.append(NOME_FRACAO_AREA_COMPLEMENT.format(nome) + '?')
#     else:
#         complements.append(CPJ_FRACAO_AREA_COMPLEMENT.format(cpf_cnpj) + '?')

#     list_questions = []
#     if isinstance(questions, str):  # simple question
#         for complement in complements:
#             list_questions.append(questions.replace('?', complement))
#     if isinstance(questions, list):  # list of questions
#         for question in questions:
#             for complement in complements:
#                 list_questions.append(question.replace('?', complement))
#     return list_questions


# def group_qas_questions(sentences: List[str]) -> Dict[str, List[int]]:
#     """Groups the sentences according to pre-defined questions.

#     Args:
#         sentences: List of sentences (strings)

#     Returns:
#         Dict with type names as keys and list of indexes of grouped sentences as
#         values.
#     """
#     questions = get_questions_for_chunk('all')

#     qid_dict = {}
#     for idx, sentence in enumerate(sentences):
#         for (kchunk, list_or_dict_questions) in questions.items():
#             if isinstance(list_or_dict_questions, list):
#                 # remove ? to match questions with complements at the end
#                 list_questions = [ question[:-1] for question in list_or_dict_questions ]
#                 if _check_sentence_intersects(sentence, list_questions):
#                     if kchunk in qid_dict.keys():
#                         qid_dict[kchunk].append(idx)
#                     else:
#                         qid_dict[kchunk] = [idx]
#             if isinstance(list_or_dict_questions, dict):
#                 for (ksubchunk, list_questions) in list_or_dict_questions.items():
#                     # remove ? to match questions with complements at the end
#                     list_questions = [ question[:-1] for question in list_questions ]
#                     if _check_sentence_intersects(sentence, list_questions):
#                         k = '.'.join([kchunk, ksubchunk])
#                         if k in qid_dict.keys():
#                             qid_dict[k].append(idx)
#                         else:
#                             qid_dict[k] = [idx]
#     return qid_dict


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
    