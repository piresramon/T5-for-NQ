import numpy as np
import torch
from tqdm import tqdm
from typing import List, Union
from transformers import PreTrainedTokenizerBase

from src.data.process_nq import process_example
from src.features.context import get_context
from src.features.preprocess import generate_t5_input_sentence, generate_t5_label_sentence


class QADataset(torch.utils.data.Dataset):
    """
    This is a simple Dataset that receives four lists of strings. The first list in the input to T5 in 
    expected format, the second one is target in T5 format, the third is the document ids, the fourth is example_ids.
    Ex.:
    examples    = ['question: When was the Third Assessment Report published? context: Another example of scientific research ...']
    labels      = ['2011']
    document_ids= ['ec57d59d-972c-40fc-82ff-c7c818d7dd39']
    example_ids = ['uhl81sp2']
    """

    def __init__(self, examples, labels, document_ids, example_ids, return_ids=False):
        self.examples = examples
        self.labels = labels
        self.document_ids = document_ids
        self.example_ids = example_ids
        self.return_ids = return_ids

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        if self.return_ids:
            return self.examples[idx], self.labels[idx], self.document_ids[idx], self.example_ids[idx]
        else:
            return self.examples[idx], self.labels[idx]
            

def squad_convert_examples_to_t5_format(
    examples: List,
    use_sentence_id: bool = True,
    max_tokens: int = 512,
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    evaluate: bool = False,
    return_dataset: Union[bool, str] = False,
    tqdm_enabled: bool = True,
):
    """Converts a list of examples into a list to the T5 format for
        question-answer with prefix question/context.

        Args:
            examples: examples to convert to T5 format.
            evaluate:
            return_dataset: Default False. Either 'pt' or 'tf'.
                if 'pt': returns a torch.data.TensorDataset,
                if 'tf': returns a tf.data.Dataset
            tqdm_enabled:

        Returns:
            list of examples into a list to the T5 format for
        question-answer with prefix question/context.

        Examples:
            >>> examples = read_nq_examples(input_file=train_file, is_training=True)
            >>> examples, labels = squad_convert_examples_to_t5_format(
            >>>     examples=examples)
    """

    examples_t5_format = []
    labels_t5_format = []
    document_ids = []   # which document the example came from? (e.g, 54f94949-0fb4-45e5-81dd-c4385f681e2b)
    example_ids = []    # which document-type and type-name does the example belong to? (e.g., matriculas.endereco)

    for example in tqdm(examples, total=len(examples), desc="convert examples to T5 format", disable=not tqdm_enabled):

        context, answer, start_char, end_char = process_example(example)
        if context == '' or answer == '':
            continue 

        # prepare the context
        question = example.questions[0] + '?'
        document = {'text': context}
        context = get_context(
            document,
            context_content='position_token',
            start_position=start_char,
            proportion_before=np.random.uniform(0.2, 0.8),
            return_position_offset=False,
            tokenizer=tokenizer,
            max_tokens=max_tokens - 4,  # to keep the </s>
            question=question,
            verbose=False)

        # prepare the input
        x = generate_t5_input_sentence(context, question, use_sentence_id)

        # prepate the target
        y = generate_t5_label_sentence(answer, start_char, context, use_sentence_id)
        
        examples_t5_format.append(x)
        labels_t5_format.append(y)
        document_ids.append(example.example_id)
        example_ids.append(example.qas_id)

    if return_dataset == "pt":

        # Create the dataset
        dataset = QADataset(examples_t5_format, labels_t5_format, document_ids, example_ids, return_ids=evaluate)

        return examples_t5_format, labels_t5_format, dataset
    elif return_dataset == "tf":
        raise RuntimeError("This is not implemented for TensorFlow.")
    else:
        return examples_t5_format, labels_t5_format