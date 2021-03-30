import configargparse
import json
import numpy as np

import torch

import pytorch_lightning as pl

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

from src.features.sentences import (
        get_highest_probability_window,
        split_compound_labels_and_predictions,
        group_qas)
from src.utils.metrics import t5_qa_evaluate
from src.utils.project import group_qas

class QAClassifier(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        config = T5Config.from_pretrained(
            self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            do_lower_case=self.hparams.do_lower_case,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=config,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )

    def forward(self, x):
        return self.model(x)

class LitQA(QAClassifier, pl.LightningModule):
    
    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer

    def training_step(self, batch, batch_idx):
        sentences, labels = batch

        sentences_tokens = self.tokenizer.batch_encode_plus(
            sentences, padding=True, truncation=True,
            max_length=self.hparams.max_seq_length, return_tensors='pt')
        labels = self.tokenizer.batch_encode_plus(
            labels, padding=True, truncation=True,
            max_length=self.hparams.max_seq_length, return_tensors='pt')

        # This is a hack to put the input parameters in same device as the model 
        inputs = {
            "input_ids": sentences_tokens['input_ids'].type_as(self.model.lm_head.weight).long(),
            "labels": labels['input_ids'].type_as(self.model.lm_head.weight).long(),
            "attention_mask": sentences_tokens['attention_mask'].type_as(self.model.lm_head.weight).long(),
            }

        outputs = self.model(**inputs)

        self.log('train_loss', outputs[0], on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': outputs[0]}

    def validation_step(self, batch, batch_idx):
        sentences, labels, _, _ = batch
           
        sentences_tokens = self.tokenizer.batch_encode_plus(
            sentences, padding=True, truncation=True,
            max_length=self.hparams.max_seq_length, return_tensors='pt')

        inputs = {
            "input_ids": sentences_tokens['input_ids'].type_as(self.model.lm_head.weight).long(),
            "attention_mask": sentences_tokens['attention_mask'].type_as(self.model.lm_head.weight).long(),
            "max_length": self.hparams.max_length,
            # "num_beams": self.hparams.num_beams,
            # "early_stopping": True,
            }

        outputs = self.model.generate(**inputs)
        predictions = [self.tokenizer.decode(output) for output in outputs]

        return {'labels': labels, 'preds': predictions}

    def test_step(self, batch, batch_idx):
        sentences, labels, document_ids, example_ids = batch

        sentences_tokens = self.tokenizer.batch_encode_plus(
            sentences, padding=True, truncation=True,
            max_length=self.hparams.max_seq_length, return_tensors='pt')

        # This is handled differently then the others because of conflicts of
        # the previous approach with quantization.
        inputs = {
            "input_ids": sentences_tokens['input_ids'].to(self.device).long(),
            "attention_mask": sentences_tokens['attention_mask'].to(self.device).long(),
            "max_length": self.hparams.max_length,
            #"num_beams": self.hparams.num_beams,
            #"early_stopping": True,
            #"repetition_penalty": 100,
            }

        outputs = self.model.generate(**inputs)
        predictions = [self.tokenizer.decode(output) for output in outputs]

        #for l, p in zip(labels, predictions):
        #    print(l, ' -------- ', p)

        # compute probs
        probs = self._compute_probs(sentences, predictions)

        return {'labels': labels, 'preds': predictions, 'doc_ids': document_ids, 'ex_ids': example_ids, 'probs': probs}

    def validation_epoch_end(self, outputs):
        predictions, labels = [], []
        for output in outputs:
            for label, pred in zip(output['labels'], output['preds']):
                predictions.append(pred)
                labels.append(label)

        results = t5_qa_evaluate(labels, predictions)
        exact = torch.tensor(results['exact'])
        f1 = torch.tensor(results['f1'])

        log = {
            'val_exact': exact,       # for monitoring checkpoint callback
            'val_f1': f1,             # for monitoring checkpoint callback
        }
        self.log_dict(log, logger=True, prog_bar=True, on_epoch=True)

    def test_epoch_end(self, outputs):
        predictions, labels, document_ids, example_ids, probs = [], [], [], [], []
        qid_dict = {}

        for output in outputs:
            for label, pred, doc_id, ex_id, prob in zip(output['labels'], output['preds'], output['doc_ids'], output['ex_ids'], output['probs']):
                predictions.append(pred)
                labels.append(label)
                document_ids.append(doc_id)
                example_ids.append(ex_id)
                probs.append(prob)
                
        # pick up the highest-probability prediction for each pair document, example
        if self.hparams.get_highestprob_answer:
            labels, predictions, document_ids, example_ids, probs = get_highest_probability_window(
                labels, predictions, document_ids, example_ids, probs, use_fewer_NA=True)

        # split compound answers to get metrics to visualize and compute metrics for each subsentence
        if self.hparams.split_compound_answers:
            labels, predictions, document_ids, example_ids, probs, original_idx = split_compound_labels_and_predictions(
                labels, predictions, document_ids, example_ids, probs)
        else:
            original_idx = list(range(len(labels)))

        # for each example_id, extract its indexes to get specific metrics
        if self.hparams.group_qas:
            qid_dict = group_qas(example_ids)
            qid_dict['ORIG'] = original_idx
        else:
            qid_dict = {'ORIG': original_idx}

        results = t5_qa_evaluate(labels, predictions, qid_dict=qid_dict)
        exact = torch.tensor(results['exact'])
        f1 = torch.tensor(results['f1'])

        # write a metric file
        with open('metric.json', 'w') as f:
            json.dump(results, f)

        # save labels and predictions
        self._save_outputs(labels, predictions, document_ids, probs, qid_dict)

        log = {
            'exact': exact,
            'f1': f1
        }
        self.log_dict(log, logger=True, on_epoch=True)

    @torch.no_grad()
    def _compute_probs(self, sentences, predictions):
        probs = []
        for sentence, prediction in zip(sentences, predictions):
            input_ids = self.tokenizer.encode(sentence, truncation=True, 
                max_length=self.hparams.max_seq_length, return_tensors="pt").to(self.device).long()
            output_ids = self.tokenizer.encode(prediction, truncation=True, 
                max_length=self.hparams.max_seq_length, return_tensors="pt").to(self.device).long()

            outputs = self.model(input_ids=input_ids, labels=output_ids)

            loss = outputs[0]
            prob = (loss * -1) / output_ids.shape[1]
            prob = np.exp(prob.cpu().numpy())
            probs.append(prob)
        return probs

    def _save_outputs(self, labels, predictions, doc_ids, probs, qid_dict=None):
        if qid_dict is None:
            qid_dict = {}

        f = open('outputs.txt', 'w')
        f.write('{0:<50} | {1:50} | {2:30} | {3}\n'.format('label', 'prediction', 'uuid', 'prob'))
        if qid_dict == {}:
            for label, prediction, doc_id, prob in zip(labels, predictions, doc_ids, probs):
                if label != prediction or label == prediction and not self.hparams.only_misprediction_outputs:
                    f.write('{0:<50} | {1:50} | {2:30} | {3}\n'.format(label, prediction, doc_id, prob))
        else:
            for (kword, list_indices) in qid_dict.items():
                if kword == 'ORIG':
                    continue
                f.write(f'===============\n{kword}\n===============\n')
                for idx in list_indices:
                    label, prediction, doc_id, prob = labels[idx], predictions[idx], doc_ids[idx], probs[idx]
                    if label != prediction or label == prediction and not self.hparams.only_misprediction_outputs:
                        f.write('{0:<50} | {1:50} | {2:30} | {3}\n'.format(label, prediction, doc_id, prob))
        f.close()

    def get_optimizer(self,) -> torch.optim.Optimizer:
        optimizer_name = self.hparams.optimizer
        scheduler_name = self.hparams.scheduler
        lr = self.hparams.lr
        weight_decay=self.hparams.weight_decay
        optimizer = getattr(torch.optim, optimizer_name)

        # Prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        
        optimizer = optimizer(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        print(f'=> Using {optimizer_name} optimizer')

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = configargparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--model_name_or_path",
            default='t5-small',
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
        )
        parser.add_argument(
            "--max_seq_length",
            default=384,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded.",
        )
        parser.add_argument(
            "--max_length",
            default=120,
            type=int,
            help="The maximum total output sequence length generated by the model."
        )
        parser.add_argument(
            "--num_beams",
            default=1,
            type=int,
            help="Number of beams for beam search. 1 means no beam search."
        )
        parser.add_argument(
            "--get_highestprob_answer", 
            action="store_true",
            help="If true, get the answer from the sliding-window that gives highest probability."
        )
        parser.add_argument(
            "--split_compound_answers",
            action="store_true",
            help="If true, split the T5 outputs into individual answers.",
        )
        parser.add_argument(
            "--group_qas",
            action="store_true",
            help="If true, use group qas to get individual metrics ans structured output file for each type-name.",
        )
        parser.add_argument(
            "--only_misprediction_outputs",
            action="store_true",
            help="If true, return only mispredictions in the output file.",
        )

        return parser
