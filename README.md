# T5 for Natural Questions

T5-for-NQ is a text-to-text question-answering for natural questions. It performs the fine-tuning of T5 model with the Natural Questions (NQ) dataset, designed for training and evaluation of automatic QAs systems using real user questions and respective answers found from Wikipedia by annotators. 

## Installation

1. Clone the repo, and enter the directory.
2. Run `pip install -e .`.

## Dataset

For downloading the dataset, first [install gsutil](https://cloud.google.com/storage/docs/gsutil_install). Thus, create the directory `data/natural-questions/` and download the complete dataset in **original format** (not the simplified train set) using:

```
gsutil -m cp -R gs://natural_questions/v1.0 data/natural-questions
```

## Usage

Configure all the hyper-parameters for the experiment editing the `params.yaml`. Thus, go ahead with:

```
python src/main.py -c params.yaml
```
