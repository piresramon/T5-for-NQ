import math
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import T5Tokenizer, PreTrainedTokenizerBase


MAX_WINDOWS = 3


def update_max_size(document: Dict, max_size: int = 4000) -> int:
    """
    Get the start_char position of the first act in order to extract 'abertura'. 
    """
    for annotation in document['annotations']:
        if annotation['type_name'] == 'ato':
            for subchunk in annotation['subchunks']:
                if subchunk['type_name'] == 'id_ato':
                    if isinstance(subchunk['start_char'], int):
                        return subchunk['start_char']
                    else:
                        return max_size
    return max_size


def get_tokens_and_offsets(text: str, tokenizer: PreTrainedTokenizerBase) -> List[Tuple[Any, int, int]]:
    tokens = tokenizer.tokenize(text)
    token_lens = [len(token) for token in tokens]
    token_lens[0] -= 1  # Ignore first "_" token
    token_ends = np.cumsum(token_lens)
    token_starts = [0] + token_ends[:-1].tolist()
    tokens_and_offsets = list(zip(tokens, token_starts, token_ends))
    return tokens_and_offsets


def get_token_id_from_position(tokens_and_offsets: List[Tuple[Any, int, int]], position: int) -> int:
    for idx, tok_offs in enumerate(tokens_and_offsets):
        _, start, end = tok_offs
        if start <= position < end:
            return idx
    return len(tokens_and_offsets) - 1


def get_max_size_context(document: Dict, max_size: int = 4000) -> str:
    """Returns the first max_size characters of the document_text.
    """
    document_text = document['text']
    context = document_text[:max_size - 4]
    context = context + ' ...'
    return context


def get_abertura_context(document: Dict, max_size: int = 4000) -> str:
    """Returns the abertura content of the document_text.
    """
    document_text = document['text']
    max_size = update_max_size(document, max_size=max_size)
    context = document_text[:max_size]
    return context


def get_position_context(
    document: Dict,
    max_size: int = 4000,
    start_position: int = 0,
    proportion_before: float = 0.2,
    verbose: bool = False,
    ) -> Tuple[str, int]:
    """Returns the content around a specific position with size controlled by max_size.
    proportion_before indicates the proportion of max_size the must be taken before 
    the position, while 1 - position_before is after.
    """
    document_text = document['text']
    start = math.floor(max_size * proportion_before)
    start = max(0, start_position - start)
    end = min(len(document_text), max_size + start)

    if verbose:
        print('-- MUST CONTAIN: ' + document_text[start_position: start_position+30])
        print(f'-- start: {start}, end: {end}')
        c = document_text[start:end]
        print(f'-- len (char): {len(c)}')
        print(f'-- context: {c} \n')

    start_reticences, end_reticences = False, False

    # try to find a punctuation before/after the start_position
    position = document_text.find('.\n', start, start_position)
    if position != -1:
        start = position + 2    # keep .\n
        position_offset = start
    else:
        start = max(start, document_text.find(' ', start, start_position))
        start_reticences = True
        position_offset = start - 3  # reticences
    
    position = document_text.rfind('.\n', start_position, end)
    if position != -1:
        end = position + 1
    else:
        end = document_text.rfind(' ', start_position, end)
        end_reticences = True
    
    context = ('...' if start_reticences else '') \
        + document_text[start: end] \
        + ('...' if end_reticences else '')

    if verbose:
        print(f'-- start: {start}, end: {end}')
    
    return context, position_offset


def get_token_context(document: Dict, 
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    max_tokens: int = 512,
    question: str = 'Qual?',
    verbose: bool = False,
    ) -> Tuple[str, int]:
    """Returns the first max_tokens tokens of the document_text.
    """
    context, position_offset = get_position_token_context(document, start_position=0,
        proportion_before=0, tokenizer=tokenizer, max_tokens=max_tokens, question=question, verbose=verbose)
    return context, position_offset
    

def get_abertura_token_context(document: Dict,
    max_size: int = 4000,
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    max_tokens: int = 512,
    question: str = 'Qual?',
    verbose: bool = False,
    ) -> Tuple[str, int]:
    """Returns the first max_tokens tokens of abertura content of the document_text.
    """
    document_text = document['text']
    max_size = update_max_size(document, max_size=max_size)
    abertura_context = document_text[:max_size]
    document = {'text': abertura_context}
    context, position_offset = get_position_token_context({'text': abertura_context}, start_position=0,
        proportion_before=0, tokenizer=tokenizer, max_tokens=max_tokens, question=question, verbose=verbose)
    return context, position_offset


def get_position_token_context(
    document: Dict,
    start_position: int = 0,
    proportion_before: float = 0.2,
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    max_tokens: int = 512,
    tokens_and_offsets: Optional[List[Tuple[Any, int, int]]] = None,
    question: str = 'Qual?',
    verbose: bool = False,
    ) -> Tuple[str, int]:
    """Returns the content around a specific position, with size controlled by max_tokens.
    proportion_before indicates the proportion of max_size the must be taken before the 
    position, while 1 - position_before is after.
    """
    document_text = document['text']
    question_sentence = f'question: {question} context: '
    num_tokens_question = len(tokenizer.tokenize(question_sentence))

    remaining_tokens = max_tokens - num_tokens_question
    start_reticences, end_reticences = False, False

    if tokens_and_offsets is None:
        tokens_and_offsets = get_tokens_and_offsets(text=document_text, tokenizer=tokenizer)
    positional_token_id = get_token_id_from_position(tokens_and_offsets=tokens_and_offsets, position=start_position)
    start_token_id = max(0, positional_token_id - math.floor(remaining_tokens * proportion_before))
    end_token_id = min(positional_token_id + math.ceil(remaining_tokens * (1-proportion_before)), len(tokens_and_offsets))

    start = tokens_and_offsets[start_token_id][1]
    end = tokens_and_offsets[end_token_id-1][2]

    num_tokens_sentence_id = (document_text[start: end].count('\n') + 1) * 5 
    num_tokens_sentence_id = 0 
    size = end_token_id - start_token_id

    # remove tokens if current size + sentence-ids tokens exceed the remaining tokens
    if size + num_tokens_sentence_id > remaining_tokens:
        to_remove = (size + num_tokens_sentence_id) - remaining_tokens
        if start == start_position:
            end_token_id -= to_remove
        else:
            remove_before = math.floor(to_remove * proportion_before)
            remove_before = min(remove_before, positional_token_id - start_token_id)
            remove_after = to_remove - remove_before
            start_token_id += remove_before
            end_token_id -= remove_after 

        start = tokens_and_offsets[start_token_id][1]
        end = tokens_and_offsets[end_token_id-1][2]
        
    # check if it requires reticences
    # if it does, try to find a space before/after the start_position
    if start != 0:
        start_reticences = True
        start = max(start, document_text.find(' ', start, start_position))
        position_offset = start - 3  # reticences
    else:
        position_offset = tokens_and_offsets[start_token_id][1]

    if end < len(document_text):
       end_reticences = True
       end = document_text.rfind(' ', start_position, end)

    if verbose:
        print('-- MUST CONTAIN: ' + document_text[start_position: start_position+30])
        print(f'-- start: {start}, end: {end}')
        c = document_text[start: end]
        print(f'-- len (char): {len(c)}')
        print(f'-- len (toks): {end_token_id - start_token_id}')
        print(f'-- context: {c} \n')

    context = ('...' if start_reticences else '') \
        + document_text[start: end] \
        + ('...' if end_reticences else '')

    if verbose:
        # it can exceed the expected num of tokens because of reticences.
        print('--> testing the number of tokens:')
        t5_input = question_sentence + context
        n = len(tokenizer.tokenize(t5_input))
        print(f'>> The input occupies {n} tokens. '
            f'It will have additional {num_tokens_sentence_id} for sentence-ids. '
            f'Total: {n + num_tokens_sentence_id}. Expected: {max_tokens}.')

    return context, position_offset 


def get_windows_token_context(
    document: Dict,
    abertura: bool = False,
    window_overlap: float = 0.5,
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    max_tokens: int = 512,
    question: str = 'Qual?',
    verbose: bool = False,
    ) -> Tuple[List[str], List[int]]:
    """Returns a list of window contents with size controlled by max_tokens, with
    overlapping near to 50%.
    """
    document_text = document['text']
    if abertura:
        max_size = update_max_size(document, max_size=len(document_text))
        document_text = document_text[:max_size].strip()
        document = document.copy()
        document['text'] = document_text

    contexts, offsets = [], []
    tokens_and_offsets = get_tokens_and_offsets(text=document_text, tokenizer=tokenizer)

    assert len(document_text) == tokens_and_offsets[-1][2], (
        f'The original document ({document["uuid"]}) and the end of last token are not matching: {len(document_text)} != {tokens_and_offsets[-1][2]}')

    start_position, position_offset = 0, 0
    context = ''
    # the offset + current context size surpassing document size means the 
    # window reached the end of document
    while position_offset + len(context) < len(document_text):

        context, position_offset = get_position_token_context(document, start_position=start_position,
            proportion_before=0, tokenizer=tokenizer, max_tokens=max_tokens, tokens_and_offsets=tokens_and_offsets, 
            question=question, verbose=verbose)

        contexts.append(context)
        offsets.append(position_offset)

        if verbose:
            print(f'>>>>>>>>>> WINDOW: start_position = {start_position}, offset = {position_offset}')

        start_position += int(len(context) * (1 - window_overlap))

        if len(contexts) == MAX_WINDOWS: break

    return contexts, offsets


def get_context(
    document: Dict,
    context_content: str = 'abertura',
    max_size: int = 4000,
    start_position: int = 0,
    proportion_before: float = 0.2,
    return_position_offset: bool = False,
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    max_tokens: int = 512,
    question: str = 'Qual?',
    window_overlap: float = 0.5,
    verbose: bool = False,
    ) -> Union[str, List[str], Tuple[Union[str, List[str]], Union[int, List[int]]]]: 
    """Returns the context to use in T5 input based on context_content.
        
     Args:
        document: dict with all the information of current document.
        context_content: type of context (max_size, abertura, position, 
            token, abertura_token, position_token, windows_token or 
            windows_abertura_token).
            - max_size: gets the first max_size characters.
            - abertura: gets the text before first act in matriculas document.
            - position: gets a window text limited to max_size characters 
            around a start_position, respecting a proportion before and after 
            the position.
            - token: gets the first max_tokens tokens.
            - abertura_token: gets the first max_tokens tokens of the abertura.
            - position_token: gets a window text limited to max_tokens tokens
            around a start_position, respecting a proportion before and after 
            the position, and penalizing tokens that will be occupied by 
            question and sentence-ids in the T5 input.
            - windows_token: gets a list of sliding windows of max_tokens,
            comprising the complete document.
            - windows_abertura_token: gets a list of sliding windows of 
            max_tokens, comprising the abertura.
        max_size: maximum size of context, in chars (used for max_size and
            position).
        start_position: char index of a keyword in the original document text 
            (used for position, and position_token).
        proportion_before: proportion of maximum context size (max_size or 
            max_tokens) that must be before start_position (used for position, 
            and position_token).
        return_position_offset: if True, returns the position of returned 
            context with respect to original document text (used for position, 
            and position_token).
        tokenizer: T5Tokenizer used in the model (used for position_token and 
            windows_token).
        max_tokens: maximum size of context, in tokens (used for position_token 
            and windows token).
        question: question that will be used along with the context in the T5
            input (used for position_token and windows_token).
        window_overlap: overlapping between windows (used for windows_token).
        verbose: visualize the processing, tests, and resultant context.

    Returns:
        - the context.
        - the position_offset (optional).        
    """
    position_offset = 0

    # remove repeated breaklines, repeated spaces/tabs, space/tabs before
    # breaklines, and breaklines in start/end of document text to make the token 
    # positions match the char positions. Those rules avoid incorrect alignments.
    document['text'] = document['text'].replace('\t', ' ')            # '\t'
    document['text'] = re.sub(r'\s*\n+\s*', r'\n', document['text'])  #  space (0 or more) + '\n' (1 or more) + space (0 or more)
    document['text'] = re.sub(r'(\s)\1+', r'\1', document['text'])    # space (1 or more)
    # special characters that causes raw and tokinization texts to desagree
    document['text'] = document['text'].replace('´', '\'')          # 0 char --> 1 char in tokenization (common in publicacoes)
    document['text'] = document['text'].replace('™', 'TM')          # 1 char --> 2 chars in tokenization
    document['text'] = document['text'].replace('…', '...')         # 1 char --> 3 chars in tokenization
    document['text'] = document['text'].strip()

    if context_content == 'max_size':
        context = get_max_size_context(document, max_size=max_size)
    elif context_content == 'abertura':
        context = get_abertura_context(document)
    elif context_content == 'position':
        context, position_offset = get_position_context(document, max_size=max_size, 
            start_position=start_position, proportion_before=proportion_before, verbose=verbose)
    elif context_content == 'token':
        context, position_offset = get_token_context(document, 
            tokenizer=tokenizer, max_tokens=max_tokens, question=question, verbose=verbose)
    elif context_content == 'abertura_token':
        context, position_offset = get_abertura_token_context(document, max_size=max_size,
            tokenizer=tokenizer, max_tokens=max_tokens, question=question, verbose=verbose)
    elif context_content == 'position_token':
        context, position_offset = get_position_token_context(document, start_position=start_position,
            proportion_before=proportion_before, tokenizer=tokenizer, max_tokens=max_tokens, question=question, verbose=verbose)
    elif context_content == 'windows_token':
        context, position_offset = get_windows_token_context(document, abertura=False, window_overlap=window_overlap,
        tokenizer=tokenizer, max_tokens=max_tokens, question=question, verbose=verbose)
    elif context_content == 'windows_abertura_token':
        context, position_offset = get_windows_token_context(document, abertura=True, window_overlap=window_overlap,
        tokenizer=tokenizer, max_tokens=max_tokens, question=question, verbose=verbose)
    else:
        return '', position_offset

    if verbose:
        if isinstance(context, list):
            for (i, cont) in enumerate(context):
                print(f'--------\nWINDOW {i}\n--------')
                print(f'len: {len(cont)} context: {cont} \n')
        else:
            print(f'len: {len(context)} context: {context} \n')

    if return_position_offset:
        return context, position_offset
    else:
        return context


def main():
    document = {}
    document['uuid'] = '1234567'
    document['text'] = "Que tal fazer uma poc inicial para vermos a viabilidade e identificarmos as dificuldades?\nA motivação da escolha desse problema " \
    "foi que boa parte dos atos de matrícula passam de 512 tokens, e ainda não temos uma solução definida para fazer treinamento e predições em " \
    "janelas usando o QA.\nEssa limitação dificulta o uso de QA para problemas que não sabemos onde a informação está no documento (por enquanto, " \
    "só aplicamos QA em tarefas que sabemos que a resposta está nos primeiros 512 tokens da matrícula).\nComo esse problema de identificar a proporção " \
    "de cada pessoa são duas tarefas (identificação + relação com uma pessoa), podemos usar a localização da pessoa no texto para selecionar apenas " \
    "uma pedaço do ato de alienação pra passar como contexto pro modelo, evitando um pouco essa limitação dos 512 tokens."
    document['text'] = "PREFEITURA DE CAUCAIA\nSECRETARIA DE FINAN\u00c7AS,PLANEJAMENTO E OR\u00c7AMENTO\nCERTID\u00c3O NEGATIVA DE TRIBUTOS ECON\u00d4MICOS\nLA SULATE\nN\u00ba 2020000982\nRaz\u00e3o Social\nCOMPASS MINERALS AMERICA DO SUL INDUSTRIA E COMERC\nINSCRI\u00c7\u00c3O ECON\u00d4MICA Documento\nBairro\n00002048159\nC.N.P.J.: 60398138001860\nSITIO SALGADO\nLocalizado ROD CE 422 KM 17, S/N - SALA SUPERIOR 01 CXP - CAUCAIA-CE\nCEP\n61600970\nDADOS DO CONTRIBUINTE OU RESPONS\u00c1VEL\nInscri\u00e7\u00e3o Contribuinte / Nome\n169907 - COMPASS MINERALS AMERICA DO SUL INDUSTRIA E COMERC\nEndere\u00e7o\nROD CE 422 KM 17, S/N SALA SUPERIOR 01 CXP\nDocumento\nC.N.P.J.: 60.398.138/0018-60\nSITIO SALGADO CAUCAIA-CE CEP: 61600970\nNo. Requerimento\n2020000982/2020\nNatureza jur\u00eddica\nPessoa Juridica\nCERTID\u00c3O\nCertificamos para os devidos fins, que revendo os registros dos cadastros da d\u00edvida ativa e de\ninadimplentes desta Secretaria, constata-se - at\u00e9 a presente data \u2013 n\u00e3o existirem em nome do (a)\nrequerente, nenhuma pend\u00eancia relativa a tributos municipais.\nSECRETARIA DE FINAN\u00c7AS, PLANEJAMENTO E OR\u00c7AMENTO se reserva o direito de inscrever e cobrar as\nd\u00edvidas que posteriormente venham a ser apurados. Para Constar, foi lavrada a presente Certid\u00e3o.\nA aceita\u00e7\u00e3o desta certid\u00e3o est\u00e1 condicionada a verifica\u00e7\u00e3o de sua autenticidade na internet, nos\nseguinte endere\u00e7o: http://sefin.caucaia.ce.gov.br/\nCAUCAIA-CE, 03 DE AGOSTO DE 2020\nEsta certid\u00e3o \u00e9 v\u00e1lida por 090 dias contados da data de emiss\u00e3o\nVALIDA AT\u00c9: 31/10/2020\nCOD. VALIDA\u00c7\u00c3O 2020000982"
    document['text'] = 'M Santander\nProposta de Abertura de Conta Poupança, Utilizando de\nProdutos e Serviços e Outras Avenças - Pessoa Física\nP3ID008159105563\n1695\nAgência Nº\nQuantidade de Titulares\nPAB N°\n1\nCondição de Movimentação da Conta\nModalidade de Poupança\nPOUPANCA ESPECIAL PF\nConta Poupança\n0033-1695-000600100141\nConta Corrente Vinculada\nConta Corrente Associada\nDados Básicos do Titular 1\nCPF |06621595271\nNome Completo\nTAISSA RIBEIRO GOMES\nDocumento de Identificação\n02-IDENTIDADE-RG\nNº do Documento \\/ N° da Série (CTPS)\n19117457\nÓrgão Emissor\nPC\nUF PA\nData de Emissão\n24\\/09\\/2018\nData de Vencimento\nData de Nascimento\n(12\\/04\\/2002\nCartório\nNº Livro\nNº Folha\nSexo FEMININO\nPaís de Nascimento\nBRASIL\nNacionalidade\nBRASILEIRA\nNaturalidade\nUF PA\nALTAMIRA\nSOLTEIRO(A)\nEstado Civil\nCondição Pessoal\n101-MAIOR COM RENDA\nNome da Mãe\nCREUZA ROSA RIBEIRO\nNome do Pai\nWELSON GOMES\nCidadania\nBRASILEIRA\nOutro domicilio fiscal\n| BRASIL\nEndereços\nEndereço Residencial\nRua\\/Av\\/Pça\\/Estrada\nTV AGRARIO CAVALCANTE\nNúmero\n| 338 _\n\\/\nComplemento\nCASA 04 VILA\nBairro\nRECREIO\nMunicipio ALTAMIRA\nPaís BRASIL\n| UF PA\n168371140\nCEP\nEndereço Comercial\nRua\\/Av\\/Pça\\/Estrada\nNúmero\nComplemento\nBairro\nMunicípio\n| UF\nUF |\nPaís BRASIL\nCEP\nEndereço Alternativo\nRua\\/Av\\/Pça\\/Estrada\nNúmero\nComplemento\nBairro |\nPag. 1 16\n'
        
    context_content = 'token' #'position_token'
    start_position = 158
    max_size = 200

    # tokenizer = T5Tokenizer.from_pretrained('models/', do_lower_case=False)
    tokenizer = T5Tokenizer.from_pretrained('/home/ramonpires/git/NLP/qa-t5/models/', do_lower_case=False)
    max_tokens = 512 #150
    question = 'Qual é a proporção?'

    context, offset = get_context(
        document,
        context_content=context_content,
        max_size=max_size,
        start_position=start_position,
        proportion_before=0.2,
        return_position_offset=True,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        question=question,
        window_overlap=0.5,
        verbose=True)

    print('--> testing the offset:')
    if isinstance(context, list):
        context, offset = context[-1], offset[-1]  # last window
    print('>>>>>>>>>> using the offset\n' + document['text'][offset:offset + len(context)])
    print('>>>>>>>>>> returned context\n' + context)


if __name__ == "__main__":
    main()
