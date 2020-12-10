# Laboro DistilBERT Japanese

## Introduction

### About Our DistilBERT Model

Several months ago, we released our BERT model pre-trained with our own web corpus. This time we decided to release a DistilBERT model. According to the paper of DistilBERT, the DistilBERT model can reduce the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster. Therefore, to show the above advantages and the quality of our model, we list the performance evaluation of two down-stream tasks as well as the inference time in sections below. It turned out that for a classification task, our DistilBERT model retains 98% of the BERT's performance, while saving 58% of the inference time. For another question answering task, the DistilBERT model retains 90% of the BERT's performance, while saving 47% of the inference time.

Although it is possible to distill from the BERT model we released, we chose to pre-train a new BERT base model and use the new model as the teacher model to train our DistilBERT mdoel. The main difference between the released BERT and the new BERT model is the corpus they are pre-trained on. Hoping to improve the performance, this time we used a Japanese corpus extracted from Common Crawl (CC) data by using [Extractor](https://github.com/paracrawl/extractor), and then cleaned the data in the same fashion Google prepared the [C4](https://github.com/google-research/text-to-text-transfer-transformer#c4) corpus.

Download the base DistilBERT model from [here](model-link-未定).

### How well is the performance

The DistilBERT model has been evaluated on two tasks, Livedoor news classification task and driving-domain question answering (DDQA) task. In Livedoor news classification, each piece of news is supposed to be classified into one of nine categories. In DDQA task, given question-article pairs, answers to the questions are expected to be found from the articles. The results of the evaluation are shown below, in comparison with a baseline [model trained by Bandai Namco](https://github.com/BandaiNamcoResearchInc/DistilBERT-base-jp), whose teacher model is [the pretrained Japanese BERT models from TOHOKU NLP LAB](https://www.nlp.ecei.tohoku.ac.jp/news-release/3284/). Note that due to the small size of the evaluation datasets, the results may differ a little every time.

For Livedoow news classification task:

| model name | model size | corpus | corpus size | eval evironment | batch size | epoch | learning rate | accuracy (%) | performance retaining rate |
|-|-|-|-|-|-|-|-|-|-|
| Laboro-BERT-teacher | Base | clean CC corpus | 13G | GPU | 4 | 10 | 5e-5 | 97.90 | - |
| Laboro-DistilBERT | Base | clean CC corpus | 13G | GPU | 4 | 10 | 5e-5 | 95.86 | 98% |
| BandaiNamco-DistilBERT | Base | JA-Wikipedia | 2.9G | GPU | 4 | 10 | 5e-5 | 94.03 | - |

For Driving-domain QA task:
| model name | model size | corpus | corpus size | eval evironment | batch size | epoch | learning rate | accuracy (%) | performance retaining rate |
|-|-|-|-|-|-|-|-|-|-|
| Laboro-BERT-teacher | Base | clean CC corpus | 13G | TPU | 32 | 3 | 5e-5 | 75.21 | - |
| Laboro-DistilBERT | Base | clean CC corpus | 13G | GPU | 32 | 3 | 5e-5 | 67.92 | 90% |
| BandaiNamco-DistilBERT | Base | JA-Wikipedia | 2.9G | GPU | 32 | 3 | 5e-5 | 52.60 | - |


### How fast is the inference

We also tested the inference time of the two tasks. Because the inference time is influenced by the status of the GPU or TPU and is not very stable, we calculated the average time over 5 times of measurement.

| model name | task | eval evironment | number of examples | ave inf. time (seconds) | time saving rate |
|-|-|-|-|-|-|
| Laboro-BERT-teacher | Livedoor | Tesla T4 GPU | 1473 | 65.4 | - |
| Laboro-DistilBERT | Livedoor | Tesla T4 GPU | 1473 | 27.2 | 58% |
| BandaiNamco-DistilBERT | Livedoor | Tesla T4 GPU | 1473 | 28 | - |
| Laboro-BERT-teacher | DDQA | TPU | 1042 | 39.2 | - |
| Laboro-DistilBERT | DDQA | Tesla T4 GPU | 1042 | 20.8 | 47% |
| BandaiNamco-DistilBERT | DDQA | Tesla T4 GPU | 1042 | 20.8 | - |

### To cite this work
We haven't published any paper on this work.
Please cite this repository:
```
@article{Laboro DistilBERT Japanese,
  title = {Laboro DistilBERT Japanese},
  author = {"Zhao, Xinyi and Hamamoto, Masafumi and Fujihara, Hiromasa"},
  year = {2020},
  howpublished = {\url{github-link-未定}}
}
```

### License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a></br>

   This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.  
   For commercial use, please [contact Laboro.AI Inc.](https://laboro.ai/contact/other/)


## To Use Our Model via Huggingface Model Hub

1. To load the pre-trained DistilBERT model, run

```
from transformers import DistilBertModel
# DistilBertModel can be replaced by AutoModel

pt_model = DistilBertModel.from_pretrained('laboro-ai/distilbert-base-japanese')
```

2. To load the SentencePiece tokenizer, run

```
from transformers import AlbertTokenizer
# AlbertTokenizer can't be replaced by AutoTokenizer

sp_tokenizer = AlbertTokenizer.from_pretrained('laboro-ai/distilbert-base-japanese')
```

3. To load the DistilBERT model fine-tuned on Livedoor News corpus, run

```
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained('laboro-ai/distilbert-base-japanese-finetuned-livedoor')
```

4. To load the DistilBERT model fine-tuned on DDQA corpus, run

```
from transformers import DistilBertForQuestionAnswering

model = DistilBertForQuestionAnswering.from_pretrained('laboro-ai/distilbert-base-japanese-finetuned-ddqa')
```


\* Note that the model on Huggingface Model Hub is also released under CC-BY-NC-4.0 license.

## Fine-Tuning DistilBERT Models

To fine-tune our DistilBERT model, download the model and tokenizer from [here](model-link-未定), and then put everything in ```./model/laboro_distilbert/``` directory.

Similarly, to fine-tune the model trained by Bandai Namco, follow their instruction [here](https://github.com/BandaiNamcoResearchInc/DistilBERT-base-jp) to download the model and vocab file, and then put everything in ```./model/namco_distilbert/``` directory.

### Classification

Text classification means assigning labels to text. Because the labels can be defined to describe any aspect of the text, text classification has a wide range of application. The most straightforward one would be categorizing the topic or sentiment of the text. Besides those, other examples include recognizing spam email, judging whether two sentences have same or similar meaning.

#### Dataset - Livedoor News Corpus

In the evaluation of English language models in classification task, several datasets (e.g. SST-2, MRPC) can be used as common benchmarks. As for Japanese DistilBERT model, [Livedoor news corpus](https://www.rondhuit.com/download.html#ldcc) can be used in the same fashion. Each piece of news in this corpus can be classified into one of the nine categories.

The original corpus is not devided in training, evaluation, and testing data. In this repository, we provide [index lists of the dataset](https://github.com/laboroai/Laboro-DistilBERT-Japanese/tree/master/data/livedoor) used in our experiments. Our dataset is pre-processed based on Livedoor News Corpus in following steps: 
* concatenating all of the data
* shuffling randomly
* deviding into train:dev:test = 2:2:6

<a name="req1"></a>

#### Requirements

* Python 3.6
* Transformers==3.4.0
* torch==1.7.0
* sentencepiece>=0.1.85
* GPU is recommended
* (for Bandai Namco model) mecab-python3 with MeCab and its dictionary mecab-ipadic-2.7.0-20070801 installed

<a name="code1"></a>

#### STEP 1 Download and Create Dataset

The Livedoor News corpus can be downloaded from [here](https://www.rondhuit.com/download.html#ldcc).

To use the same dataset splits as we used in our experiments, follow the instruction [here](https://github.com/laboroai/Laboro-DistilBERT-Japanese/tree/master/data/livedoor). 

The dataset should be put in ```./data/livedoor/``` directory.

#### STEP 2 Pre-Tokenize Data

1. Our DistilBERT

   SentencePiece tokenizer should be used.  
   
   To pre-tokenize Livedoor dataset, use ```./src/data_prep/sentencepiece_tokenize_livedoor.ipynb```. 

2. Bandai Namco DistilBERT 

   First follow ```./src/data_prep/gen_vocab_lower.ipynb``` to generate lowercase vocab file for later use.

   Mecab tokenizer with mecab-ipadic-NEologd dictionary should be used first, then transformers WordPiece tokenizer.    
   
   To pre-tokenize Livedoor dataset, use ```./src/data_prep/mecab_wordpiece_tokenize_livedoor.ipynb```. 

#### STEP 3 Fine-Tune

1. Our DistilBERT

   Rename the file ```config_livedoor.json``` as ```config.json```, and don't forget to backup the original ```config.json```.

   To fine-tune the model for Livedoor task, use ```./src/laboro_distilbert/finetune-livedoor.ipynb```. 

2. Bandai Namco DistilBERT

   Rename the file ```config_livedoor.json``` as ```config.json```, and don't forget to backup the original ```config.json```.

   To fine-tune the model for Livedoor task, use ```./src/namco_distilbert/finetune-livedoor.ipynb```. 

### Question Answering

Question answering is another task to evaluate and apply language models. In English NLP, [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is one the of most widely used datasets for this task. In SQuAD, questions and corresponding Wikipedia pages are given, and the answers to the questions are supposed to be found from the Wikipedia pages.

#### Dataset - Driving Domain QA

For QA task, we used [Driving Domain QA dataset](http://nlp.ist.i.kyoto-u.ac.jp/index.php?Driving%20domain%20QA%20datasets) for evaluation. This dataset consists of PAS-QA dataset and RC-QA dataset. So far, we have only evaluated our model on the RC-QA dataset. The dataset is already in the format of SQuAD 2.0, so no pre-processing is needed for further use.

<a name="req2"></a>

#### Requirements

* Python 3.6
* Transformers==3.4.0
* torch==1.7.0
* sentencepiece>=0.1.85
* GPU is recommended
* (for Bandai Namco model) mecab-python3 with MeCab and its dictionary mecab-ipadic-2.7.0-20070801 installed

<a name="code2"></a>

#### STEP 1 Download and Create Dataset

Download DDQA dataset following the instruction [here](http://nlp.ist.i.kyoto-u.ac.jp/index.php?Driving%20domain%20QA%20datasets).

Since we only use the data in ```RC-QA``` dataset, please copy the dataset into ```./data/ddqa/RC-QA/``` directory.

#### STEP 2 Pre-Tokenize Data

1. Our DistilBERT

   To pre-tokenize DDQA dataset, use ```./src/data_prep/sentencepiece_tokenize_ddqa.ipynb```.  

2. Bandai Namco DistilBERT

   First follow ```./src/data_prep/gen_vocab_lower.ipynb``` to generate lowercase vocab file for later use.

   To pre-tokenize DDQA dataset, use ```./src/data_prep/mecab_wordpiece_tokenize_ddqa.ipynb```. 

#### STEP 3 Fine-Tune

1. Our DistilBERT

   Rename the file ```config_ddqa.json``` as ```config.json```, and don't forget to backup the original ```config.json```.

   To fine-tune the model for DDQA task, use ```./laboro_distilbert/finetune-ddqa.ipynb```.

2. Bandai Namco DistilBERT

   Rename the file ```config_ddqa.json``` as ```config.json```, and don't forget to backup the original ```config.json```.

   To fine-tune the model for DDQA task, use ```./namco_distilbert/finetune-ddqa.ipynb```.
