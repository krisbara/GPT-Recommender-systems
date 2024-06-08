# Recommender systems based on GPT

The project is devoted to GPT-based news recommender systems (NRS). To be more precise, it tests ChatGPT as NRS and introduces three NRSs (content, collaborative, hybrid) based on the OpenAI's text-embedding-ada-002.

## Data

The data from the Microsoft News Dataset (MIND) was selected for testing recommender systems.

## Sentiment analysis

The data was not initially annotated in terms of sentiment. The annotation was made automatically using TextBlob. TextBlob is a Python library that provides a straightforward API for conducting funda- mental NLP tasks, such as sentiment analysis, trained on real feedback examples from an e-commerce website.

## Evaluation

Evaluation of the RSs include performance (precision, recall, F1 score) and diversity metrics. Person correlation was applied to estimate category, subcategory, and sentiment distribution in user profiles and recommended items. 