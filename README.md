
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

A complete educational implementation of a Natural Language Processing (NLP) pipeline for social media sentiment classification. This project illustrates how computers convert human language into structured numerical representations, learn linguistic patterns, and predict emotional tone in text data.

The notebook presents an end-to-end NLP workflow, guiding the reader from raw text preprocessing to model training and evaluation. It demonstrates how unstructured language is transformed into machine-readable features and how a machine learning model leverages those features to classify sentiment accurately.

The project covers both foundational and advanced NLP concepts, including vector mathematics, cosine similarity, word embeddings, Bag-of-Words vectorization, and transformer-based contextual embeddings with BERT. Each stage is implemented with educational clarity, connecting theory to practical application.

The architecture mirrors real-world NLP systems used in sentiment analysis, social media monitoring, customer feedback analytics, and intelligent automation solutions.

The repository is structured as a progressive learning journey — beginning with mathematical fundamentals and culminating in a fully functional sentiment classification model for social media text.


<br><br>


## [Overview]()

<br>

**Social Buzz AI** is a production-oriented Natural Language Processing (NLP) project designed to transform raw textual data into structured numerical representations and perform sentiment classification using machine learning.

This repository demonstrates a complete end-to-end NLP workflow, including preprocessing, feature engineering, vectorization, model training, evaluation, persistence, and interactive inference.

The architecture reflects real-world NLP systems used in:

- Social media monitoring  
- Customer feedback analysis  
- Brand intelligence  
- Opinion mining  
- Automated text classification  


<br><br>


## [Project Objectives]()

<br>

- Build a complete NLP pipeline from raw text to prediction  
- Apply text cleaning and normalization techniques  
- Convert text into numerical feature representations  
- Implement Bag of Words and word embedding approaches  
- Train and evaluate a supervised classification model  
- Persist trained models for reuse  
- Enable real-time prediction from user input  

<br><br>

## [Libraries Used]()

<br>

- `numpy` — numerical computation and linear algebra  
- `pandas` — data manipulation and analysis  
- `nltk` — text preprocessing and stopword filtering  
- `scikit-learn` — machine learning models and vectorization tools  
- `gensim` — word embedding models  
- `transformers` — contextual embeddings (BERT)  
- `torch` — deep learning backend  
- `safetensors` — optimized tensor storage  
- `pickle` — model serialization  

<br><br>


## [NLP Pipeline Architecture]()

<br>

Raw Text ↓ Text Cleaning ↓ Tokenization ↓ Stopword Removal ↓ Feature Engineering ↓ Vectorization ↓ Model Training ↓ Model Evaluation ↓ Model Persistence ↓ Prediction

<br><br>


## Table of Contents

1. [What Is This Notebook About?](#what-is-this-notebook-about)  
2. [Playing With Numbers (Vectors)](#playing-with-numbers-vectors)  
3. [Finding Out If Two Things Are Alike (Cosine Similarity)](#finding-out-if-two-things-are-alike-cosine-similarity)  
4. [Using Secret Codes For Words (Word Embeddings)](#using-secret-codes-for-words-word-embeddings)  
5. [Super-Secret Codes: Transformers and BERT](#super-secret-codes-transformers-and-bert)  
6. [Getting Our Messages (Loading Data)](#getting-our-messages-loading-data)  
7. [Cleaning Up Our Messages (Text Preprocessing)](#cleaning-up-our-messages-text-preprocessing)  
8. [Turning Words Into Numbers (Bag of Words)](#turning-words-into-numbers-bag-of-words)  
9. [Splitting Our Messages (Training and Testing)](#splitting-our-messages-training-and-testing)  
10. [Teaching Our Computer To Guess Feelings (Model Training)](#teaching-our-computer-to-guess-feelings-model-training)  
11. [Saving Our Computer’s Brain For Later (Model Persistence)](#saving-our-computers-brain-for-later-model-persistence)  
12. [Asking The Computer To Guess For Us! (Interactive Prediction)](#asking-the-computer-to-guess-for-us-interactive-prediction)  
13. [NLP Applications in Data Science](#nlp-applications-in-data-science)  


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



















