# coding=utf-8
""" Finetuning the T5 model for question-answering with Natural Questions."""

import configargparse

import numpy as np

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.models.qa_model import LitQA
from src.data.nq_data import QADataModule


def main():

    parser = configargparse.ArgParser('Training and evaluation script for training T5 model for QA', 
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path')

    # optimizer parameters
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--num_workers', default=8, type=int)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    # add model specific args
    parser = LitQA.add_model_specific_args(parser)
    # add datamodule specific args
    parser = QADataModule.add_model_specific_args(parser) 
    args, unknown = parser.parse_known_args()

    # setting the seed for reproducibility
    if args.deterministic:
        seed_everything(args.seed)

    # data module
    dm = QADataModule(args) 
    dm.setup('fit')

    # Defining the model
    model = LitQA(args)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(filepath='lightning_logs/{epoch}-{val_exact:.2f}-{val_f1:.2f}',
                                          monitor='val_exact', verbose=False, save_last=True, save_top_k=1,
                                          save_weights_only=False, mode='max', period=1)

    # Defining the Trainer, training... and finally testing
    trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)
    trainer.fit(model, datamodule=dm)

    dm.setup('test')
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
