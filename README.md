# NLP-Studies
Studies on Linear and NN Approaches for Sentiment Analysis and QA Problem. This Mini-Projects are part of an NLP course taken during my final year of undergraduate studies 
at Department of Informatics and Telecommunications, NKUA.

## Prerequisities
- python3.x
- torch
- transformers, datasets (hugging face library)
- GPU accelerator for RNN and transformer training

### Python3.x and PyPI
Notebooks are meant to run on google colab or kaggle enviroments, where python and pip are preinstalled

### PyTorch, Hugging-Face libraries
```sh
pip3 install torch transformers datasets
```

## Datasets
The datasets used for these Studies were based on tweets and have 3 labels neutral, negative, positive. Specifically each tweets expresses the user's opinion in vaccines.

For the QA research we used 5 different datasets and tried to reproduce the results from [this paper](https://arxiv.org/pdf/2004.03490.pdf). For further information on the datasets check my [kaggle account](https://www.kaggle.com/vissarionmoutafis/datasets) 

## Multi-Label Logistic Regression for Sentiment Analysis
In this part we are using multi label logistic regression models to predict the tweets sentiment. We first transform the tweets to vectors using BoW or TF-IDF tokenizer after some cleaning process.

## Feed-Forward NN for Sentiment Analysis
In the second version of our sentiment analysis study we are using the same preprocessing by the addition of some pretrained embeddings transformation, pipelined to a Dense neural network with 1-2 hidden layers.

## RNN for Sentiment Analysis
Continuing our research we tried to solve the problem using some RNN architectures like GRU's and LSTM's that would keep a sense of remembering the context. The same preprocessing methods were used.

## Trasformers for Sentiment Analysis
In this final part we used a pretrained BERT transformer and its tokenizer and fine tuned it on our twitter dataset. We provided a tokenization mapper and a post processing function to acquire performance scores. For further elaboration on how transformers work go check hugging face's tutorials, since they pointed me the right direction during the fine tuning developement. 


## QA Fine-Tuning Research
In this part we used the same fine tuning methods with 5 different datasets, evaluating on the same datasets testing-slices for each produced model. 


