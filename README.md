
<br>
 
 
 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]


<br>

# <p align="center"> 5- Social [Buzz AI]() - Natural Language Processing (NLP)  [Class 2 - Project]() - Lesson on Vector Representations and Text Classification



<br><br>


<p align="center">
   <img src="https://github.com/user-attachments/assets/791a69e2-d09a-429f-9257-f6667fff5c04 ">
 </p>

<br><br>

[**Course:**]() Humanistic AI & Data Science (4th Semester)  
[**Institution:**]() PUC-SP  
**Professor:**  [✨ Rooney Ribeiro Albuquerque Coelho](https://www.linkedin.com/in/rooney-coelho-320857182/)



<br><br>


#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-%C2%B7%C2%B7%C2%B7%20Mindful%20AI%20Assistants%20%C2%B7%C2%B7%C2%B7-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)


<br><br>


> [!TIP]
>
>  This repository is part of the main project 1-social-buzz-ai-main.
>  To explore all related materials, analyses, and notebooks, visit the main repository 
>
> * [1-social-buzz-ai-main](https://github.com/Mindful-AI-Assistants/1-social-buzz-ai-main)
> *Part of the Humanistic AI Research & Data Modeling Series — where data meets human insight.*
>
> * [4- Social Buss: NLP - Class 1](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NLP-Class_1) 
> 
> * [Embedding Projector](https://projector.tensorflow.org/)
> 
>



<br><br>


<!--Confidentiality Statement-->


> [!NOTE]
>
> ⚠️ Heads Up 
>
> * Projects and deliverables may be made [publicly available]() whenever possible.
>
> * The course prioritizes [**hands-on practice**]() with real data in consulting scenarios.
>
> *  All activities comply with the [**academic and ethical guidelines of PUC-SP**]().
>
> * [**Confidential information**]() from this repository remains private in [private repositories]().
>
>

#  

<br><br><br>

<!--End-->

> [!TIP]
>
> * [Access](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1/blob/7a5b1e2ad8bee693c6842a3b79a38c3d7d668239/1-Workbook_Natural%20Language%20Processing%20-%20Class%201.pdf)  Workbook - (Class 1 and 2)
> 
> * [Access](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1/blob/1170f52a88b432225e216b3519810285d65b3066/1_%F0%9F%87%AC%F0%9F%87%A7_NLP_Pre_Processing_ENGLISH.ipynb):  🇬🇧 1- NLP_Pre_Processing_ENGLISH
>
> * [Access](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1/blob/f395b6b6ffba24b65dd6e593e4bb3b3b899301e0/1_%F0%9F%87%A7%F0%9F%87%B7NLP_PreProcessing_Portuguese.ipynb):   🇧🇷 1-Code NLP_Pre_Processing_Portuguese
>
>
>  * [Access](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1):  NLP  - Class 1 Repo
>
 



<br><br><br>


## [Overview]()

**Social Buzz AI** is an end-to-end Natural Language Processing (NLP) project focused on sentiment classification for social media data.

The repository demonstrates how raw, unstructured text is systematically transformed into structured numerical representations that machines can interpret. Through a complete machine learning pipeline, the project illustrates how linguistic patterns are extracted, modeled, and used to predict sentiment categories such as positive, negative, and neutral.

 <br>  

### * [The notebook follows a progressive workflow:]()

 <br> 

[*]() Text preprocessing and normalization   <br> 
[*]() Feature engineering and vectorization (Bag-of-Words)   <br> 
[*]() Mathematical foundations (vectors and cosine similarity)   <br> 
[*]() Word embeddings and contextual representations   <br> 
[*]() Transformer-based modeling with BERT   <br> 
[*]() Model training, evaluation, and persistence   <br> 
[*]() Interactive sentiment inference  <br> 

 <br> 

By combining foundational concepts with modern NLP architectures, the project bridges theory and practical implementation. Its structure mirrors real-world systems used in social media analytics, customer feedback intelligence, brand monitoring, opinion mining, and automated text classification.

  <br> 


The result is a structured learning journey — from mathematical fundamentals to a fully operational sentiment analysis model ready for practical application.



<br><br>


## [Project Objectives]()

<br>

[*]() Build a complete NLP pipeline from raw text to prediction    <br> 
[*]() Apply text cleaning and normalization techniques    <br> 
[*]() Convert text into numerical feature representations    <br> 
[*]() Implement Bag of Words and word embedding approaches    <br> 
[*]() Train and evaluate a supervised classification model    <br> 
[*]() Persist trained models for reuse    <br> 
[*]() Enable real-time prediction from user input  

<br><br>

## [Libraries Used]()

<br>

[*]() `numpy` — numerical computation and linear algebra    <br> 
[*]() `pandas` — data manipulation and analysis    <br> 
[*]() `nltk` — text preprocessing and stopword filtering    <br> 
[*]() `scikit-learn` — machine learning models and vectorization tools    <br> 
[*]() `gensim` — word embedding models    <br> 
[*]() `transformers` — contextual embeddings (BERT)    <br> 
[*]() `torch` — deep learning backend    <br> 
[*]() `safetensors` — optimized tensor storage    <br> 
[*]() `pickle` — model serialization  

<br><br>


## [NLP Pipeline Architecture]()

<br>

Raw Text ↓ Text Cleaning ↓ Tokenization ↓ Stopword Removal ↓ Feature Engineering ↓ Vectorization ↓ Model Training ↓ Model Evaluation ↓ Model Persistence ↓ Prediction

<br><br>


## 📚 Table of Contents

1. [What Is This Notebook About?](#1-what-is-this-notebook-about)
2. [Vector Foundations](#2-vector-foundations)
3. [Cosine Similarity](#3-cosine-similarity)
4. [Word Embeddings](#4-word-embeddings)
5. [Transformers and BERT](#5-transformers-and-bert)
6. [Loading Data](#6-loading-data)
7. [Text Preprocessing](#7-text-preprocessing)
8. [Bag of Words Vectorization](#8-bag-of-words-vectorization)
9. [Train-Test Split](#9-train-test-split)
10. [Model Training](#10-model-training)
11. [Model Evaluation](#11-model-evaluation)
12. [Model Persistence](#12-model-persistence)
13. [Interactive Prediction](#13-interactive-prediction)
14. [NLP Applications in Data Science](#14-nlp-applications-in-data-science)
15. [Technologies Demonstrated](#15-technologies-demonstrated)
16. [Bibliographic References](#16-bibliographic-references)

<br><br>


## 1. [What Is This Notebook About?]()

<br>

This notebook demonstrates how machines learn to understand human language through structured numerical transformations.

The objective is to classify textual messages into sentiment categories such as:

- Positive  
- Negative  
- Neutral  


<br><br>



## 2. [Vector Foundations]()

<br>

Vectors enable mathematical operations required for NLP.

<br>

```python
import numpy as np

vector1 = np.array([1., 2., 1., 4.])
vector3 = np.ones(4)

print("Dot Product:", np.dot(vector1, vector3))
```


<br><br>


## 3. [Cosine Similaritys]()

<br>

Cosine similarity measures angular similarity between vectors.

<br>

```python
import numpy as np
from numpy.linalg import norm

A = np.array([1, 2, 3])
B = np.array([2, 3, 4])

cos_sim = A @ B / (norm(A) * norm(B))
print("Cosine Similarity:", cos_sim)
```

<br><br>



## 4. [Word Embeddings]()

<br>
   
Dense vector representations that capture semantic meaning.

Example:

<br>

```python
king - man + woman ≈ queen
```

<br><br>


## 5. [Transformers and BERT]()

<br>
   
Context-aware embeddings generated by transformer architectures.

<br>


```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-large-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-large-portuguese-cased")

text = "Eu vou ao banco pagar a conta hoje."
input_ids = tokenizer.encode(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(input_ids)

print(outputs.last_hidden_state.shape)
```


<br><br>



## 6. [Loading Data]()

<br>


```python
import pandas as pd

df = pd.read_csv("TweetsMg.csv")
print(df.head())
```


<br><br>


## 7. [Text Preprocessing]()

<br>


```python
import nltk
nltk.download("stopwords")

stopwords = nltk.corpus.stopwords.words("portuguese")
```


<br><br>


## 8. [Bag of Words Vectorization]()

<br>


```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(df["Text"])
```


<br><br>


## 9. [Train-Test Split]()

<br>


```python
from sklearn.model_selection import train_test_split

y, labels = pd.factorize(df["Classificacao"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```


<br><br>



## 10. [Model Training]()

<br>


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```


<br><br>


## 11. [Model Evaluation]()

<br>

[*]() Accuracy <br>
[*]() Precision <br>
[*]() Recall <br>
[*]() F1-score <br>
[*]() Confusion matrix <br>


<br><br>


## 12. [Model Persistence]()

<br>

```python
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```


<br><br>


## 13. [Interactive Prediction]()

<br>

```python
text_input = input("Type a message: ")
vector = vectorizer.transform([text_input])
prediction = model.predict(vector)
print("Predicted sentiment:", prediction)
```


<br><br>


## 14. [NLP Applications in Data Science]()

<br>

Natural Language Processing is widely used in real-world systems. Key applications:


<br> 


| [Application]()                      | [Description]() |
|------------------------------------|------------|
| [Sentiment Analysis]()                 | Determines the emotional tone (positive, negative, neutral) in text such as customer reviews, social media posts, or news articles. |
| [Named Entity Recognition (NER)]()     | Identifies and categorizes key entities such as people, organizations, locations, and dates. |
| [Machine Translation]()                | Automatically translates text or speech from one language to another. |
| [Chatbots and Virtual Assistants]()     | Enables machines to understand and respond to user queries in a human-like way. |
| [Text Summarization]()                 | Generates concise summaries of large documents or articles. |
| [Topic Modeling]()                    | Discovers abstract topics in a collection of documents for clustering and analysis. |



<br><br> 



## 15. [Technologies Demonstrated()


[*]() Natural Language Processing <br> 
[*]()  Machine Learning <br> 
[*]()  Text Classification <br> 
[*]()  Word Embeddings <br> 
[*]()  Transformer Models (BERT) <br> 
[*]()  Feature Engineering <br> 
[*]()  Model Persistence <br> 



<br><br> 


## 16.[ Bibliographic References]()

<br> 

### *  [**Core Textbooks**]()

[1.]() **Jurafsky, D., & Martin, J. H.** *Speech and Language Processing* (3rd ed.). Pearson, 2023. (Foundational NLP pipeline, tokenization, preprocessing) <br> 
[2.]() **Bird, S., Klein, E., & Loper, E.** *Natural Language Processing with Python*. O'Reilly Media, 2009. (NLTK, text preprocessing, sentiment basics) <br> 
[3.]() **Eisenstein, J.** *Introduction to Natural Language Processing*. MIT Press, 2019. (Vector representations, cosine similarity) <br> 
[4.]() **Manning, C. D., Raghavan, P., & Schütze, H.** *Introduction to Information Retrieval*. Cambridge University Press, 2008. (TF-IDF, vector space model for tweets) <br> 

 <br>   

### *  [**Key Academic Papers**]()

 <br>   

[1.]() **Pang, B., & Lee, L.** "Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales." *ACL 2005*. (Early Twitter sentiment analysis foundation) <br> 
[2.]() **Pak, A., & Paroubek, P.** "Twitter as a Corpus for Sentiment Analysis and Opinion Mining." *LREC 2010*. (Tweet-specific preprocessing challenges) <br> 
[3.]() **Severyn, A., & Moschitti, A.** "Twitter sentiment analysis with deep convolutional neural networks." *SIGIR 2015*. (Modern vector-based sentiment) <br> 
[4.]() **Pennington, J., Socher, R., & Manning, C. D.** "GloVe: Global Vectors for Word Representation." *EMNLP 2014*. (Word embeddings for social media) <br> 

 <br>   

###  *  [**Technical Standards & Frameworks**]()

 <br>  

[1.]() **NLTK Documentation** (Bird et al., 2009) - Tokenization, stemming, stop words <br> 
[2.]() **Scikit-learn Text Feature Extraction** - TF-IDF, cosine similarity metrics <br> 
[3.]() **Hugging Face Transformers** (Wolf et al., 2020) - Modern NLP pipelines <br> 


 <br>   


### *  [**Code & Implementation References**]()

 <br>   


[1.]() **NLTK Corpus Guidelines** for Twitter data preprocessing  <br>
[2.]() **Scikit-learn Vectorizers** (Pedregosa et al., 2011) for pipeline implementation <br> 
[3.]() **Cosine Similarity Math** from Information Retrieval theory (Manning et al., 2008) <br> 


 <br><br> 


<!-- ======================================= Start DEFAULT Footer ===========================================  -->

## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br>


#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />


<br><br>

<p align="center">  ────────────── ⊹🔭๋ ──────────────

<!--
<p align="center">  ────────────── 🛸๋*ੈ✩* 🔭*ੈ₊ ──────────────
-->

<br>

<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>
  

  
#
 
##### <p align="center">Copyright 2026 Mindful-AI-Assistants. Code released under the  [MIT license.](https://github.com/Mindful-AI-Assistants/CDIA-Entrepreneurship-Soft-Skills-PUC-SP/blob/21961c2693169d461c6e05900e3d25e28a292297/LICENSE)




<!-- ======================================= End  DEFAULT Footer ===========================================  -->



















