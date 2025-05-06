# Comprehensive NLP Analysis of Global News Headlines (2019-Present)

This document presents a thorough Natural Language Processing (NLP) analysis of global news headlines spanning from 2019 to the present. The analysis demonstrates proficiency in fundamental NLP techniques, preprocessing steps, and insightful visualizations that extract meaningful patterns from textual data.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Exploration](#data-exploration)
3. [Data Preprocessing](#data-preprocessing)
   - [Text Normalization](#text-normalization)
   - [Tokenization](#tokenization)
   - [Stop Words Removal](#stop-words-removal)
   - [Lemmatization](#lemmatization)
   - [Part-of-Speech Tagging](#part-of-speech-tagging)
   - [Named Entity Recognition](#named-entity-recognition)
4. [Feature Extraction](#feature-extraction)
   - [Bag of Words](#bag-of-words)
   - [TF-IDF Vectorization](#tf-idf-vectorization)
   - [Word Embeddings](#word-embeddings)
5. [Sentiment Analysis](#sentiment-analysis)
   - [Monthly Sentiment Trends](#monthly-sentiment-trends)
   - [Year-over-Year Sentiment Comparison](#year-over-year-sentiment-comparison)
6. [Topic Modeling](#topic-modeling)
   - [Latent Dirichlet Allocation](#latent-dirichlet-allocation)
   - [Topic Evolution Over Time](#topic-evolution-over-time)
7. [Entity Analysis](#entity-analysis)
   - [Most Mentioned Entities](#most-mentioned-entities)
   - [Entity Co-occurrence Networks](#entity-co-occurrence-networks)
8. [Time Series Analysis](#time-series-analysis)
   - [Headline Complexity Over Time](#headline-complexity-over-time)
   - [Topic Seasonality](#topic-seasonality)
9. [Conclusion](#conclusion)

## Introduction

This analysis explores a rich dataset of news headlines from 2019 to the present day, covering 25 of the world's most read news sources. The dataset is structured with dates in the first column followed by headlines from each source. By applying various NLP techniques, we aim to uncover patterns, trends, and insights that reveal how global news discourse has evolved over this significant period.

## Data Exploration

Let's begin by loading the dataset and exploring its basic structure:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
news_df = pd.read_csv('news_headlines.csv')

# Display basic information
print(f"Dataset shape: {news_df.shape}")
print(f"Column names:")
print(news_df.columns.tolist())

# Check the first few rows
print("\nFirst few rows of the dataset:")
print(news_df.head(2))

# Convert Date column to datetime (note capital "D")
# Since the date format is "May 01, 2018", we need to parse it correctly
news_df['Date'] = pd.to_datetime(news_df['Date'], format="%b %d, %Y")

# Display the date range
print(f"\nTime period: {news_df['Date'].min()} to {news_df['Date'].max()}")

# Create a year-month column for temporal analysis
news_df['year_month'] = news_df['Date'].dt.to_period('M')

# Create a column for the year
news_df['year'] = news_df['Date'].dt.year

# Create a column for the month
news_df['month'] = news_df['Date'].dt.month

# Sample a few rows
print("\nSample data:")
news_df.head()
```

Next, let's create a function to combine all headlines for each date into a single text corpus:

```python
def combine_headlines(row):
    """Combine all headlines in a row into a single string"""
    headlines = []
    # Start from column 'Top1' through 'Top25'
    for col in news_df.columns[1:26]:  # Skip the Date column, include only Top1-Top25
        if pd.notna(row[col]):
            headlines.append(str(row[col]))
    return ' '.join(headlines)

# Apply the function to create a new column with combined headlines
news_df['combined_headlines'] = news_df.apply(combine_headlines, axis=1)

# Check the distribution of headlines over time
plt.figure(figsize=(14, 6))
news_df.groupby(news_df['Date'].dt.to_period('M')).size().plot(kind='bar')
plt.title('Number of Headlines per Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

## Data Preprocessing

### Text Normalization

Before diving into complex NLP tasks, we need to normalize our text data:

```python
import re
import string

def normalize_text(text):
    """
    Normalize text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing numbers
    4. Removing extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Apply normalization
news_df['normalized_text'] = news_df['combined_headlines'].apply(normalize_text)

# Display a sample of normalized text
print("Original headline:")
print(news_df['combined_headlines'].iloc[0][:200])
print("\nNormalized headline:")
print(news_df['normalized_text'].iloc[0][:200])
```

### Tokenization

Tokenization is the process of breaking text into individual words or tokens:

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK resources
nltk.download('punkt')

# Word tokenization
def tokenize_text(text):
    """Split text into individual words"""
    return word_tokenize(text)

# Sentence tokenization
def sentence_tokenize(text):
    """Split text into sentences"""
    return sent_tokenize(text)

# Apply tokenization
news_df['tokens'] = news_df['normalized_text'].apply(tokenize_text)
news_df['sentences'] = news_df['combined_headlines'].apply(sentence_tokenize)

# Display sample tokenization
print("Tokenized words (first 20):")
print(news_df['tokens'].iloc[0][:20])
print("\nTokenized sentences (first 2):")
print(news_df['sentences'].iloc[0][:2])

# Calculate basic token statistics
news_df['token_count'] = news_df['tokens'].apply(len)
news_df['avg_token_length'] = news_df['tokens'].apply(lambda x: np.mean([len(token) for token in x]) if x else 0)
news_df['sentence_count'] = news_df['sentences'].apply(len)

# Plot token count distribution
plt.figure(figsize=(10, 6))
sns.histplot(news_df['token_count'], bins=30, kde=True)
plt.title('Distribution of Token Counts per Day')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
```

### Stop Words Removal

Stop words are common words like "the", "and", "is" that don't carry much meaning:

```python
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Add custom stop words relevant to news headlines
custom_stops = {'says', 'said', 'reuters', 'ap', 'afp', 'report', 'reports'}
stop_words.update(custom_stops)

def remove_stopwords(tokens):
    """Remove stop words from a list of tokens"""
    return [token for token in tokens if token not in stop_words]

# Apply stopwords removal
news_df['tokens_nostop'] = news_df['tokens'].apply(remove_stopwords)

# Compare before and after
print("Original token count:", sum(news_df['token_count']))
print("Token count after stopword removal:", sum(news_df['tokens_nostop'].apply(len)))
print("Reduction: {:.2f}%".format((1 - sum(news_df['tokens_nostop'].apply(len))/sum(news_df['token_count'])) * 100))

# Display sample
print("\nBefore stopword removal (first 20):")
print(news_df['tokens'].iloc[0][:20])
print("\nAfter stopword removal (first 20):")
print(news_df['tokens_nostop'].iloc[0][:20])
```

### Lemmatization

Lemmatization reduces words to their base form (lemma):

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required resources
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to noun if tag not found

def lemmatize_tokens(tokens):
    """Lemmatize a list of tokens with POS context"""
    return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

# Apply lemmatization
news_df['tokens_lemma'] = news_df['tokens_nostop'].apply(lemmatize_tokens)

# Compare before and after
print("Before lemmatization (sample):")
print(news_df['tokens_nostop'].iloc[0][:15])
print("\nAfter lemmatization (sample):")
print(news_df['tokens_lemma'].iloc[0][:15])

# Calculate unique token reduction
unique_before = len(set([token for tokens in news_df['tokens_nostop'] for token in tokens]))
unique_after = len(set([token for tokens in news_df['tokens_lemma'] for token in tokens]))
print(f"\nUnique tokens before lemmatization: {unique_before}")
print(f"Unique tokens after lemmatization: {unique_after}")
print(f"Reduction: {(1 - unique_after/unique_before) * 100:.2f}%")
```

### Part-of-Speech Tagging

POS tagging identifies the grammatical category of each word:

```python
# Apply POS tagging to a sample of the data
def pos_tag_text(tokens):
    """Apply POS tagging to a list of tokens"""
    return nltk.pos_tag(tokens)

# Apply to a sample for demonstration
sample_size = min(100, len(news_df))
pos_tagged_sample = [pos_tag_text(tokens) for tokens in news_df['tokens'][:sample_size]]

# Create a distribution of POS tags
pos_counts = Counter([tag for tagged_doc in pos_tagged_sample for _, tag in tagged_doc])

# Plot POS tag distribution
plt.figure(figsize=(12, 6))
pos_df = pd.DataFrame([{'POS': pos, 'Count': count} for pos, count in pos_counts.most_common(15)])
sns.barplot(data=pos_df, x='POS', y='Count')
plt.title('Most Common Parts of Speech in Headlines')
plt.xlabel('Part of Speech')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display POS tag descriptions
pos_descriptions = {
    'NN': 'Noun, singular',
    'NNP': 'Proper noun, singular',
    'IN': 'Preposition or conjunction',
    'DT': 'Determiner',
    'JJ': 'Adjective',
    'NNS': 'Noun, plural',
    'VBZ': 'Verb, 3rd person singular present',
    'CD': 'Cardinal number',
    'VBN': 'Verb, past participle',
    'CC': 'Coordinating conjunction',
    'VB': 'Verb, base form',
    'TO': 'To',
    'VBG': 'Verb, gerund or present participle',
    'VBD': 'Verb, past tense',
    'PRP': 'Personal pronoun'
}

print("POS Tag Descriptions:")
for pos, desc in pos_descriptions.items():
    if pos in pos_counts:
        print(f"{pos}: {desc} - {pos_counts[pos]} occurrences")
```

### Named Entity Recognition

NER identifies entities like persons, organizations, locations, etc.:

```python
from nltk import ne_chunk
from collections import defaultdict

# Download required resource
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_entities(pos_tagged_tokens):
    """Extract named entities from POS tagged tokens"""
    chunk_tree = ne_chunk(pos_tagged_tokens)
    entities = defaultdict(list)
    
    for subtree in chunk_tree:
        if hasattr(subtree, 'label'):
            entity_type = subtree.label()
            entity_text = ' '.join(word for word, tag in subtree.leaves())
            entities[entity_type].append(entity_text)
    
    return dict(entities)

# Apply NER to a sample of the data
sample_ner = [extract_entities(pos_tag_text(tokens)) for tokens in news_df['tokens'][:sample_size]]

# Count entity types
entity_type_counts = Counter()
for doc_entities in sample_ner:
    for entity_type, entities in doc_entities.items():
        entity_type_counts[entity_type] += len(entities)

# Plot entity type distribution
plt.figure(figsize=(10, 6))
entity_df = pd.DataFrame([{'Entity Type': etype, 'Count': count} 
                          for etype, count in entity_type_counts.most_common()])
sns.barplot(data=entity_df, x='Entity Type', y='Count')
plt.title('Named Entity Types in Headlines')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Entity type descriptions
entity_descriptions = {
    'PERSON': 'People, including fictional',
    'ORGANIZATION': 'Companies, agencies, institutions',
    'GPE': 'Geopolitical entities (countries, cities)',
    'LOCATION': 'Non-GPE locations, mountain ranges, bodies of water',
    'FACILITY': 'Buildings, airports, highways, bridges',
    'DATE': 'Absolute or relative dates',
    'TIME': 'Times smaller than a day',
    'MONEY': 'Monetary values',
    'PERCENT': 'Percentage',
    'PRODUCT': 'Products',
    'EVENT': 'Named events'
}

print("Entity Type Descriptions:")
for etype, desc in entity_descriptions.items():
    if etype in entity_type_counts:
        print(f"{etype}: {desc} - {entity_type_counts[etype]} occurrences")
```

## Feature Extraction

### Bag of Words

The Bag of Words model represents text as a multiset of words:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer
count_vectorizer = CountVectorizer(max_features=5000, min_df=5)

# Apply to the lemmatized tokens (converted back to text)
lemma_texts = [' '.join(tokens) for tokens in news_df['tokens_lemma']]
bow_matrix = count_vectorizer.fit_transform(lemma_texts)

# Get feature names
feature_names = count_vectorizer.get_feature_names_out()

# Show dimensions
print(f"Bag of Words matrix shape: {bow_matrix.shape}")
print(f"Number of features (vocabulary size): {len(feature_names)}")

# Most frequent terms
term_frequencies = bow_matrix.sum(axis=0).A1
term_freq_df = pd.DataFrame({'term': feature_names, 'frequency': term_frequencies})
term_freq_df = term_freq_df.sort_values('frequency', ascending=False)

# Plot most frequent terms
plt.figure(figsize=(12, 8))
sns.barplot(data=term_freq_df.head(20), x='frequency', y='term')
plt.title('20 Most Frequent Terms in Headlines')
plt.xlabel('Frequency')
plt.ylabel('Term')
plt.tight_layout()
plt.show()
```

### TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) weighs terms based on their importance:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5)

# Apply to the lemmatized tokens
tfidf_matrix = tfidf_vectorizer.fit_transform(lemma_texts)

# Get feature names
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Show dimensions
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Calculate average TF-IDF scores across documents
avg_tfidf = tfidf_matrix.mean(axis=0).A1
tfidf_scores_df = pd.DataFrame({'term': tfidf_feature_names, 'tfidf': avg_tfidf})
tfidf_scores_df = tfidf_scores_df.sort_values('tfidf', ascending=False)

# Plot terms with highest TF-IDF scores
plt.figure(figsize=(12, 8))
sns.barplot(data=tfidf_scores_df.head(20), x='tfidf', y='term')
plt.title('20 Most Important Terms by TF-IDF')
plt.xlabel('Average TF-IDF Score')
plt.ylabel('Term')
plt.tight_layout()
plt.show()
```

### Word Embeddings

Word embeddings capture semantic relationships between words:

```python
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

# Train Word2Vec model
w2v_model = Word2Vec(sentences=news_df['tokens_lemma'].tolist(), 
                     vector_size=100, 
                     window=5, 
                     min_count=5, 
                     workers=4)

# Get most similar words for some key terms
key_terms = ['covid', 'trump', 'climate', 'economy', 'technology']
for term in key_terms:
    try:
        similar_words = w2v_model.wv.most_similar(term, topn=5)
        print(f"Words similar to '{term}':")
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
    except KeyError:
        print(f"'{term}' not in vocabulary")

# Visualize embeddings using PCA
def plot_embeddings(model, words):
    """Visualize word embeddings in 2D using PCA"""
    # Get embeddings for words that exist in the model
    word_vectors = []
    valid_words = []
    
    for word in words:
        try:
            vector = model.wv[word]
            word_vectors.append(vector)
            valid_words.append(word)
        except KeyError:
            continue
    
    if not word_vectors:
        return
        
    # Reduce dimensions with PCA
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(word_vectors)
    
    # Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], marker='o')
    
    # Add labels
    for i, word in enumerate(valid_words):
        plt.annotate(word, (coordinates[i, 0], coordinates[i, 1]), fontsize=10)
    
    plt.title('Word Embeddings Visualization (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Get top terms to visualize
top_words = term_freq_df['term'][:50].tolist()
plot_embeddings(w2v_model, top_words)
```

## Sentiment Analysis

### Monthly Sentiment Trends

Let's analyze the sentiment of headlines over time:

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Apply sentiment analysis to each day's combined headlines
def get_sentiment(text):
    """Get sentiment scores using VADER"""
    if not isinstance(text, str) or text == "":
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    return sid.polarity_scores(text)

# Apply sentiment analysis
news_df['sentiment'] = news_df['combined_headlines'].apply(get_sentiment)

# Extract sentiment components
news_df['negative'] = news_df['sentiment'].apply(lambda x: x['neg'])
news_df['neutral'] = news_df['sentiment'].apply(lambda x: x['neu'])
news_df['positive'] = news_df['sentiment'].apply(lambda x: x['pos'])
news_df['compound'] = news_df['sentiment'].apply(lambda x: x['compound'])

# Group by month and calculate average sentiment
monthly_sentiment = news_df.groupby(news_df['Date'].dt.to_period('M')).agg({
    'negative': 'mean',
    'neutral': 'mean',
    'positive': 'mean',
    'compound': 'mean'
}).reset_index()

# Convert Period to datetime for plotting
monthly_sentiment['date_ts'] = monthly_sentiment['date'].dt.to_timestamp()

# Plot compound sentiment over time
plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_sentiment, x='date', y='compound')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Monthly Average Sentiment of News Headlines')
plt.xlabel('Date')
plt.ylabel('Compound Sentiment Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot all sentiment components
plt.figure(figsize=(14, 7))
plt.plot(monthly_sentiment['date'], monthly_sentiment['positive'], 'g-', label='Positive')
plt.plot(monthly_sentiment['date'], monthly_sentiment['negative'], 'r-', label='Negative')
plt.plot(monthly_sentiment['date'], monthly_sentiment['neutral'], 'b-', label='Neutral')
plt.title('Monthly Sentiment Components of News Headlines')
plt.xlabel('Date')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Year-over-Year Sentiment Comparison

Let's compare sentiment across years:

```python
# Add year column
news_df['year'] = news_df['date'].dt.year

# Calculate average sentiment by year
yearly_sentiment = news_df.groupby('year').agg({
    'negative': 'mean',
    'neutral': 'mean',
    'positive': 'mean',
    'compound': 'mean'
}).reset_index()

# Plot yearly sentiment
plt.figure(figsize=(12, 7))
bar_width = 0.2
x = np.arange(len(yearly_sentiment))

plt.bar(x - bar_width, yearly_sentiment['negative'], width=bar_width, label='Negative', color='red')
plt.bar(x, yearly_sentiment['neutral'], width=bar_width, label='Neutral', color='blue')
plt.bar(x + bar_width, yearly_sentiment['positive'], width=bar_width, label='Positive', color='green')

plt.xlabel('Year')
plt.ylabel('Average Sentiment Score')
plt.title('Year-over-Year Sentiment Comparison')
plt.xticks(x, yearly_sentiment['year'])
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Plot compound sentiment by year
plt.figure(figsize=(10, 6))
sns.barplot(data=yearly_sentiment, x='year', y='compound')
plt.title('Yearly Average Compound Sentiment')
plt.xlabel('Year')
plt.ylabel('Compound Score')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Find the year with the most negative sentiment
worst_year = yearly_sentiment.loc[yearly_sentiment['compound'].idxmin()]
print(f"The year with the most negative news sentiment was {worst_year['year']} with a compound score of {worst_year['compound']:.4f}")
```

## Topic Modeling

### Latent Dirichlet Allocation

LDA is used to discover abstract topics in a collection of documents:

```python
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn

# Number of topics
n_topics = 10

# Create LDA model
lda = LatentDirichletAllocation(n_components=n_topics, 
                                random_state=42,
                                max_iter=10,
                                learning_method='online')

# Fit the model to the BoW matrix
lda_output = lda.fit_transform(bow_matrix)

# Print top words for each topic
print("Top words per topic:")
feature_names = count_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx+1}: {', '.join(top_words)}")

# Add dominant topic to each document
news_df['dominant_topic'] = lda_output.argmax(axis=1) + 1  # Add 1 to get topics 1-based

# Visualize topics with pyLDAvis
pyLDAvis.enable_notebook()  # For Jupyter notebooks
vis = pyLDAvis.sklearn.prepare(lda, bow_matrix, count_vectorizer)
pyLDAvis.display(vis)
```

### Topic Evolution Over Time

Let's track how topics evolve over time:

```python
# Calculate topic prevalence per month
topic_cols = [f'Topic_{i+1}' for i in range(n_topics)]
topic_df = pd.DataFrame(lda_output, columns=topic_cols)
topic_df['Date'] = news_df['Date'].reset_index(drop=True)
topic_df['year_month'] = topic_df['Date'].dt.to_period('M')

# Group by month and calculate average topic prevalence
monthly_topics = topic_df.groupby('year_month').mean().reset_index()
monthly_topics['year_month_ts'] = monthly_topics['year_month'].dt.to_timestamp()

# Plot topic evolution over time
plt.figure(figsize=(15, 10))
for i, topic_col in enumerate(topic_cols):
    plt.plot(monthly_topics['year_month_ts'], monthly_topics[topic_col], label=f'Topic {i+1}')

plt.title('Evolution of Topics Over Time')
plt.xlabel('Date')
plt.ylabel('Topic Prevalence')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Find dominant topic for each month
monthly_topics['dominant_topic'] = monthly_topics[topic_cols].idxmax(axis=1)
monthly_topics['dominant_topic'] = monthly_topics['dominant_topic'].apply(lambda x: int(x.split('_')[1]))

# Create a heatmap of topics over time
pivot_df = pd.pivot_table(
    data=topic_df,
    values=topic_cols,
    index=topic_df['year_month'],
    aggfunc='mean'
).reset_index()

# Convert period to string for better display
pivot_df['year_month'] = pivot_df['year_month'].astype(str)
pivot_df = pivot_df.set_index('year_month')

plt.figure(figsize=(16, 10))
sns.heatmap(pivot_df, cmap='viridis')
plt.title('Topic Distribution Over Time')
plt.xlabel('Topics')
plt.ylabel('Year-Month')
plt.tight_layout()
plt.show()
```

## Entity Analysis

### Most Mentioned Entities

Let's analyze the most frequently mentioned entities:

```python
from collections import Counter
import itertools

# Extract all entities from the sample
all_entities = []
for doc_entities in sample_ner:
    for entity_type, entities in doc_entities.items():
        all_entities.extend([(entity_type, entity) for entity in entities])

# Count entities
entity_counter = Counter(all_entities)

# Get top entities by type
person_entities = [entity for (entity_type, entity), count in entity_counter.most_common() if entity_type == 'PERSON']
org_entities = [entity for (entity_type, entity), count in entity_counter.most_common() if entity_type == 'ORGANIZATION']
gpe_entities = [entity for (entity_type, entity), count in entity_counter.most_common() if entity_type == 'GPE']

# Plot top entities by type
def plot_top_entities(entities, entity_type, n=10):
    """Plot top n entities of a given type"""
    entities_to_plot = entities[:n]
    counts = [entity_counter[(entity_type, entity)] for entity in entities_to_plot]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts, y=entities_to_plot)
    plt.title(f'Top {n} {entity_type} Entities Mentioned in Headlines')
    plt.xlabel('Count')
    plt.ylabel(entity_type)
    plt.tight_layout()
    plt.show()

# Plot top entities
plot_top_entities(person_entities, 'PERSON')
plot_top_entities(org_entities, 'ORGANIZATION')
plot_top_entities(gpe_entities, 'GPE')
```

### Entity Co-occurrence Networks

Let's analyze how entities appear together:

```python
import networkx as nx
from itertools import combinations

# Create entity co-occurrence network
entity_network = nx.Graph()

# Extract co-occurring entities within documents
for doc_entities in sample_ner:
    # Get all entities in the document
    all_doc_entities = []
    for entity_type, entities in doc_entities.items():
        all_doc_entities.extend([(entity_type, entity) for entity in entities])
    
    # Add edges between co-occurring entities
    for (type1, entity1), (type2, entity2) in combinations(all_doc_entities, 2):
        if entity1 != entity2:  # Avoid self-loops
            edge_key = (f"{type1}:{entity1}", f"{type2}:{entity2}")
            
            # Add nodes if they don't exist
            if not entity_network.has_node(f"{type1}:{entity1}"):
                entity_network.add_node(f"{type1}:{entity1}", 
                                        type=type1, 
                                        name=entity1)
            
            if not entity_network.has_node(f"{type2}:{entity2}"):
                entity_network.add_node(f"{type2}:{entity2}", 
                                        type=type2, 
                                        name=entity2)
            
            # Add edge or increase weight if it exists
            if entity_network.has_edge(*edge_key):
                entity_network[edge_key[0]][edge_key[1]]['weight'] += 1
            else:
                entity_network.add_edge(*edge_key, weight=1)

# Plot entity co-occurrence network
def plot_entity_network(G, min_edge_weight=2, entity_types=None):
    """
    Plot the entity co-occurrence network
    
    Parameters:
    - G: NetworkX graph
    - min_edge_weight: Minimum edge weight to include
    - entity_types: List of entity types to include (None for all)
    """
    # Filter by edge weight
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_edge_weight]
    G_filtered = G.copy()
    G_filtered.remove_edges_from(edges_to_remove)
    
    # Filter by node types if specified
    if entity_types:
        nodes_to_keep = [n for n, d in G_filtered.nodes(data=True) if d['type'] in entity_types]
        G_filtered = G_filtered.subgraph(nodes_to_keep)
    
    # Remove isolated nodes
    G_filtered.remove_nodes_from(list(nx.isolates(G_filtered)))
    
    # Skip if no nodes remain
    if len(G_filtered.nodes()) == 0:
        print("No nodes remain after filtering.")
        return
    
    # Get node positions using a layout algorithm
    pos = nx.spring_layout(G_filtered, k=0.3, iterations=50)
    
    # Get node sizes based on degree centrality
    degree_centrality = nx.degree_centrality(G_filtered)
    node_sizes = [5000 * degree_centrality[node] + 100 for node in G_filtered.nodes()]
    
    # Get edge weights
    edge_weights = [G_filtered[u][v]['weight'] * 0.5 for u, v in G_filtered.edges()]
    
    # Set up colors for different entity types
    color_map = {
        'PERSON': 'red',
        'ORGANIZATION': 'blue',
        'GPE': 'green',
        'LOCATION': 'purple',
        'FACILITY': 'orange',
        'DATE': 'brown',
        'TIME': 'pink',
        'MONEY': 'cyan',
        'PERCENT': 'magenta',
        'PRODUCT': 'yellow',
        'EVENT': 'gray'
    }
    
    # Get node colors
    node_colors = [color_map.get(G_filtered.nodes[node]['type'], 'black') for node in G_filtered.nodes()]
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Draw network
    nx.draw_networkx(
        G_filtered, 
        pos=pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color='gray',
        width=edge_weights,
        alpha=0.7,
        font_size=8
    )
    
    # Create legend for node types
    if entity_types:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[t], markersize=10, 
                                     label=t) for t in entity_types if t in color_map]
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Entity Co-occurrence Network')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot network for PERSON, ORGANIZATION, and GPE entities
plot_entity_network(entity_network, min_edge_weight=2, 
                   entity_types=['PERSON', 'ORGANIZATION', 'GPE'])

# Calculate and display network metrics
print("Entity Network Statistics:")
print(f"Number of nodes: {entity_network.number_of_nodes()}")
print(f"Number of edges: {entity_network.number_of_edges()}")

# Calculate network centrality measures
degree_cent = nx.degree_centrality(entity_network)
betweenness_cent = nx.betweenness_centrality(entity_network)
closeness_cent = nx.closeness_centrality(entity_network)

# Most central entities by degree centrality
top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nMost central entities by degree centrality:")
for entity, score in top_degree:
    print(f"{entity}: {score:.4f}")
