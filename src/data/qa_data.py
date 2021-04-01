import os
import configargparse
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import T5Tokenizer

from src.data.convert_nq_to_t5 import squad_convert_examples_to_t5_format
from src.data.process_nq import read_nq_examples

class QADataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            do_lower_case=self.hparams.do_lower_case,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )

    def setup(self, stage: Optional[str] = None):
        input_dir = '.' #self.hparams.data_dir if self.hparams.data_dir else "."

        # Prepare train and valid datasets
        if stage == 'fit' or stage is None:
            # Load data examples from cache or dataset file
            cached_examples_train_file = os.path.join(
                input_dir,
                "cached_train_{}".format(
                    list(filter(None, self.hparams.model_name_or_path.split("/"))).pop()
                )
            )
            cached_examples_valid_file = os.path.join(
                input_dir,
                "cached_valid_{}".format(
                    list(filter(None, self.hparams.model_name_or_path.split("/"))).pop()
                )
            )

            # Init examples and dataset from cache if it exists
            if os.path.exists(cached_examples_train_file) and os.path.exists(cached_examples_valid_file) and not self.hparams.overwrite_cache:
                print("Loading examples from cached files %s and %s" % (cached_examples_train_file, cached_examples_valid_file))

                examples_and_dataset = torch.load(cached_examples_train_file)
                self.train_dataset = examples_and_dataset["dataset"]
                examples_and_dataset = torch.load(cached_examples_valid_file)
                self.valid_dataset = examples_and_dataset["dataset"]
            else:
                print("Creating examples from dataset file at %s" % input_dir)

                examples_train = read_nq_examples(input_file=self.hparams.train_file, is_training=True)
                examples_valid = read_nq_examples(input_file=self.hparams.valid_file, is_training=True)
                        
                _, _, self.train_dataset = squad_convert_examples_to_t5_format(
                    examples=examples_train,
                    use_sentence_id=self.hparams.use_sentence_id,
                    max_tokens=self.hparams.max_seq_length,
                    tokenizer=self.tokenizer,
                    evaluate=False,
                    return_dataset="pt",
                )
                _, _, self.valid_dataset = squad_convert_examples_to_t5_format(
                    examples=examples_valid,
                    use_sentence_id=self.hparams.use_sentence_id,
                    max_tokens=self.hparams.max_seq_length,
                    tokenizer=self.tokenizer,
                    evaluate=True,
                    return_dataset="pt",
                )

                print("Saving examples into cached file %s" % cached_examples_train_file)
                torch.save({"dataset": self.train_dataset}, cached_examples_train_file)
                print("Saving examples into cached file %s" % cached_examples_valid_file)
                torch.save({"dataset": self.valid_dataset}, cached_examples_valid_file)
            
            print(f'>> train-dataset: {len(self.train_dataset)} samples')
            print(f'>> valid-dataset: {len(self.valid_dataset)} samples')

        # Prepare test dataset
        if stage == 'test' or stage is None:

            assert self.hparams.test_file, 'test_file must be specificed'

            cached_examples_test_file = os.path.join(
                input_dir,
                "cached_test_{}".format(
                    list(filter(None, self.hparams.model_name_or_path.split("/"))).pop()
                )
            )

            # Init examples and dataset from cache if it exists
            if os.path.exists(cached_examples_test_file) and not self.hparams.overwrite_cache:

                print("Loading examples from cached file %s" % (cached_examples_test_file))

                examples_and_dataset = torch.load(cached_examples_test_file)
                self.test_dataset = examples_and_dataset["dataset"]
            else:
                print("Creating examples from dataset file at %s" % input_dir)

                examples_test = read_nq_examples(input_file=self.hparams.test_file, is_training=True)

                _, _, self.test_dataset = squad_convert_examples_to_t5_format(
                    examples=examples_test,
                    use_sentence_id=self.hparams.use_sentence_id,
                    max_tokens=self.hparams.max_seq_length,
                    tokenizer=self.tokenizer,
                    evaluate=True,
                    return_dataset="pt",
                )

                print("Saving examples into cached file %s" % cached_examples_test_file)
                torch.save({"dataset": self.test_dataset}, cached_examples_test_file)
            
            print(f'>> test-dataset: {len(self.test_dataset)} samples')

    def get_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def train_dataloader(self,) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=self.hparams.shuffle_train,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self,) -> DataLoader:
        return self.get_dataloader(
            self.valid_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self,) -> DataLoader:
        return self.get_dataloader(
            self.test_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = configargparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_file",
            default=None,
            type=str,
            help="The input training file. If a data dir is specified, will look for the file there"
        )
        parser.add_argument(
            "--valid_file",
            default=None,
            type=str,
            help="The input evaluation file. If a data dir is specified, will look for the file there"
        )
        parser.add_argument(
            "--test_file",
            default=None,
            type=str,
            help="The input test file. If a data dir is specified, will look for the file there"
        )
        parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
        parser.add_argument("--val_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--shuffle_train", action="store_true", help="Shuffle the train dataset")
        parser.add_argument("--use_sentence_id", action="store_true", help="Set this flag if you are using the approach that breaks the contexts into sentences")
        parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
        
        return parser