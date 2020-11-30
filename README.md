# Laboro DistilBERT Japanese

## Introduction

### About Our DistilBERT Model

Several months ago, we released our BERT model pre-trained with our own web corpus. This time we decided to release a DistilBERT model. According to the paper of DistilBERT, the DistilBERT model can reduce the size of a BERT model by 40%, while retaining 97%
of its language understanding capabilities and being 60% faster. Therefore, to show the above advantages and the quality of our model, we list the performance evaluation of two down-stream tasks as well as the inference time in sections below. It turned out that for a classification task, our DistilBERT model retains 98% of the BERT's performance, while saving 54% of the inference time. For another question answering task, the DistilBERT model retains 90% of the BERT's performance, while saving 42% of the inference time.

Although it is possible to distill from the BERT model we released, we chose to pre-train a new BERT base model and use the new model as the teacher model to train our DistilBERT mdoel. The main difference between the released BERT and the new BERT model is the corpus they are pre-trained on. Hoping to improve the performance, this time we used a Japanese corpus extracted from Common Crawl (CC) data by using [Extractor](https://github.com/paracrawl/extractor), and then cleaned the data in the same fashion Google prepared the [C4](https://github.com/google-research/text-to-text-transfer-transformer#c4) corpus.

Download the base DistilBERT model from [here](model-link-未定).

### How well is the performance

The DistilBERT model has been evaluated on two tasks, Livedoor news classification task and driving-domain question answering (DDQA) task. In Livedoor news classification, each piece of news is supposed to be classified into one of nine categories. In DDQA task, given question-article pairs, answers to the questions are expected to be found from the articles. The results of the evaluation are shown below, in comparison with a baseline [model trained by Bandai Namco](https://github.com/BandaiNamcoResearchInc/DistilBERT-base-jp), whose teacher model is [the pretrained Japanese BERT models from TOHOKU NLP LAB](https://www.nlp.ecei.tohoku.ac.jp/news-release/3284/). Note that due to the small size of the evaluation datasets, the results may differ a little every time.

For Livedoow news classification task:

| model name | model size | corpus | corpus size | eval evironment | batch size | epoch | learning rate | accuracy (%) | performance retaining rate |
|-|-|-|-|-|-|-|-|-|-|
| Laboro-BERT-teacher | Base | clean CC corpus | 13G | GPU | 4 | 10 | 2e-5 | 97.69 | - |
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


## Fine-Tuning DistilBERT Models

### Classification

Text classification means assigning labels to text. Because the labels can be defined to describe any aspect of the text, text classification has a wide range of application. The most straightforward one would be categorizing the topic or sentiment of the text. Besides those, other examples include recognizing spam email, judging whether two sentences have same or similar meaning.

#### Dataset - Livedoor News Corpus

In the evaluation of English language models in classification task, several datasets (e.g. SST-2, MRPC) can be used as common benchmarks. As for Japanese DistilBERT model, [Livedoor news corpus](https://www.rondhuit.com/download.html#ldcc) can be used in the same fashion. Each piece of news in this corpus can be classified into one of the nine categories.

The original corpus is not devided in training, evaluation, and testing data. In this repository, we provide [index lists of the dataset](index-link-未定) used in our experiments. Our dataset is pre-processed based on Livedoor News Corpus in following steps: 
* concatenating all of the data
* shuffling randomly
* deviding into train:dev:test = 2:2:6

<a name="req1"></a>

#### Requirements

* Python 3.6
* Transformers==3.1.0
* torch==1.7.0
* sentencepiece>=0.1.85
* GPU is recommended
* Mecab (for Fine-tuning Namco)

<a name="code1"></a>

#### STEP 1 Pre-Tokenize Data

1. Our DistilBERT

   SentencePiece tokenizer should be used.  
   To pre-tokenize Livedoor dataset, use ```./data_prep/setnencepiece_tokenize_livedoor.ipynb```. 

2. Bandai Namco DistilBERT 

   Mecab tokenizer with mecab-ipadic-NEologd dictionary should be used first, then WordPiece tokenizer.    
   To pre-tokenize Livedoor dataset, use ```./data_prep/mecab_wordpiece_tokenize_livedoor.ipynb```. 

#### STEP 2 Fine-Tune

1. Our DistilBERT

   ```./ccc13g_distilbert/tokenization.py``` file is needed to clean and tokenize text.  
   To fine-tune the model for Livedoor task, use ```./ccc13g_distilbert/finetune-livedoor.ipynb```.  

2. Bandai Namco DistilBERT

   To fine-tune the model for Livedoor task, copy ```./namco_distilbert/config_for_livedoor.json``` to the model directory as ```config.json```, then use ```./namco_distilbert/finetune-livedoor.ipynb```.  

### Question Answering

Question answering is another task to evaluate and apply language models. In English NLP, [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is one the of most widely used datasets for this task. In SQuAD, questions and corresponding Wikipedia pages are given, and the answers to the questions are supposed to be found from the Wikipedia pages.

#### Dataset - Driving Domain QA

For QA task, we used [Driving Domain QA dataset](http://nlp.ist.i.kyoto-u.ac.jp/index.php?Driving%20domain%20QA%20datasets) for evaluation. This dataset consists of PAS-QA dataset and RC-QA dataset. So far, we have only evaluated our model on the RC-QA dataset. The dataset is already in the format of SQuAD 2.0, so no pre-processing is needed for further use.

<a name="req2"></a>

#### Requirements

* Python 3.6
* Transformers==3.1.0
* torch==1.7.0
* sentencepiece>=0.1.85
* GPU is recommended

<a name="code2"></a>

#### STEP 1 Pre-Tokenize Data

1. Our DistilBERT

   To pre-tokenize DDQA dataset, use ```./data_prep/sentencepiece_tokenize_ddqa.ipynb```.  

2. Bandai Namco DistilBERT

   To pre-tokenize DDQA dataset, use ```./data_prep/mecab_wordpiece_tokenize_ddqa.ipynb```. 

#### STEP 2 Fine-Tune

1. Our DistilBERT

   ```./ccc13g_distilbert/tokenization.py``` file is needed to clean and tokenize text.  
   To fine-tune the model for DDQA task, use ```./ccc13g_distilbert/finetune-ddqa.ipynb```.

2. Bandai Namco DistilBERT

   To fine-tune the model for DDQA task, copy ```./namco_distilbert/config_for_ddqa.json``` to the model directory as ```config.json```, then use ```./namco_distilbert/finetune-ddqa.ipynb```.

## About the Pre-Training of Our Model

### Corpus
TBA

### Training
TBA

#### Hyper-parameters

#### Environment