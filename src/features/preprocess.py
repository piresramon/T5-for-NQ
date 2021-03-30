"""Utility methods to preprocess model input."""
from collections import Counter
from typing import List, Optional, Tuple, Union

from src.features.sentences import SENT_TOKEN

# Large number to not let the number of sentences be too large for a model.
MAX_SENTENCES = 9999


def _replace_brackets_with_parenthesis(text: str) -> str:
    text = text.replace('{', '(')
    text = text.replace('}', ')')

    return text


def _replace_linebreak_with_token_patterns(
        text: str, token_pattern: str = SENT_TOKEN
) -> Tuple[str, int]:
    """Returns new string with `\n` replaced with the token pattern and the
    number of tokens."""
    num_tokens = text.count('\n')
    text = text.replace('\n', token_pattern)

    return text, num_tokens


def _replace_linebreaks_with_tokens(text: str) -> str:
    r"""Replaces every `\n` in a string with a numbered SENT token.

    If the inputs string has brackets, they will be replaced with parenthesis.
    Always adds least one SENT token at the beginning of the new sentence.
    Tokens are numerated starting from 1.

    Args:
        text: string to have `\n` replaced. It can't be split into more than
            MAX_SENTENCES.

    Examples:
        >>> sentence = 'Rua PEDRO BIAGI 462 Apartamento nº 103, 1º Andar do RESIDENCIAL IMPERIAL. Sertãozinho\nSP'
        >>> new_sentence = _replace_linebreaks_with_tokens(sentence)
        >>> print(new_sentence)
        ' [SENT1] Rua PEDRO BIAGI 462 Apartamento nº 103, 1º Andar do RESIDENCIAL IMPERIAL. Sertãozinho [SENT2] SP'

    Returns:
        New string with token instead of `\n`
    """
    # Should have at least one SENT token at start
    text = '\n' + text
    text = _replace_brackets_with_parenthesis(text)
    text, num_tokens = _replace_linebreak_with_token_patterns(text)

    assert num_tokens <= MAX_SENTENCES, 'Maximum number of sentences violated.'

    # token numeration must start from 1
    text = text.format(*range(1, num_tokens + 1))

    return text


def _replace_linebreaks_with_spaces(text: str) -> str:
    r"""Replaces every `\n` in a string with a space.

    Examples:
        >>> sentence = 'Rua PEDRO BIAGI 462 Apartamento nº 103, 1º Andar do RESIDENCIAL IMPERIAL. Sertãozinho\nSP'
        >>> new_sentence = _replace_linebreaks_with_spaces(sentence)
        >>> print(new_sentence)
        'Rua PEDRO BIAGI 462 Apartamento nº 103, 1º Andar do RESIDENCIAL IMPERIAL. Sertãozinho SP'
    """
    text = text.replace('\n', ' ')

    return text


def _get_id_based_on_linebreaks(context: str, answer_position: int) -> int:
    """Recover the sentence-id assuming the context is always partiotioned based
    on occurrences of linebreaks.

    Args:
        context: text context of the question
        answer_position: index of last character from answer.
    """
    if answer_position == -1:
        return 0

    sent_id = Counter(context[:answer_position])['\n'] + 1

    return sent_id


def generate_t5_input_sentence(
        context: str, question: str, use_sentence_id: bool
) -> str:
    """Returns a T5 input sentence based on a question and its context.

    Args:
        context: text context of the question
        question: the question
        use_sentence_id: if True, every newline on the context will be replaced
            by a SENT token. Otherwise they are replaced with spaces.
    """
    if use_sentence_id:
        context = _replace_linebreaks_with_tokens(context)
    else:
        context = _replace_linebreaks_with_spaces(context)

    t5_sentence = f'question: {question} context: {context} </s>'

    return t5_sentence


def generate_t5_label_sentence(
        answer: str, answer_start: Union[List[int], int], context: str,
        use_sentence_id: bool
) -> str:
    """Returns a T5 label sentence for simple or compound answers.

    Args:
        answer: answer of the current questions
        answer_start: char position of answer starting
        context: text context of the question
        use_sentence_id: if True, every newline on the context will be replaced
            by a SENT token. Otherwise they are replaced with spaces.
    """
    if use_sentence_id:
        if isinstance(answer_start, list):
            # That is a compound_answer, like: "[Valor]: 500,00 [Unidade]: metro_quadrado"

            # Separate the compound answer in sub-answers: --, Valor] 500,00, Unidade] metro_quadrado
            # that could be problematic if some sub-answer has brackets, besides COMPLEMENT_TYPE
            sub_answers = answer.split('[')[1:] 
            token_pattern = SENT_TOKEN.strip()                    

            # Extract sentence-ids for each sub-answer
            sent_ids = []
            for sub_answer_start in answer_start:
                sent_ids.append(_get_id_based_on_linebreaks(context,
                                                            sub_answer_start))

            # Prepare the final answer with sentence-ids: "[SENTx] [Valor]: 500,00 [SENTy] [Unidade]: metro_quadrado"
            answer = ''
            for sub_answer in sub_answers:
                if sub_answer.startswith(COMPLEMENT_TYPE):
                    answer = f'{answer}[{sub_answer}'
                else:
                    answer = f'{answer}{token_pattern} [{sub_answer}'

            # Include the sentence-ids
            answer = answer.format(*sent_ids)
        elif isinstance(answer_start, int):
            # That is a simple answer

            sent_id = _get_id_based_on_linebreaks(context, answer_start)
            answer = f'[SENT{sent_id}] {answer}'
        else:
            # That is an occurrence of non-annotated data, as publicacoes (null in squad json)
            # [SENTX] is not included
            pass

    return answer
