📰 NLP News Headlines Analysis (2019–2023)
A comprehensive natural language processing (NLP) project analyzing global news headlines from 2019 to 2023. The focus is on extracting insights from language trends, sentiment patterns, and entity frequencies using classic NLP techniques.

📌 Project Highlights
✅ Cleaned and normalized thousands of real-world news headlines

✅ Removed stopwords and unified variations of key terms (e.g., "coronavirus" vs. "covid")

✅ Performed tokenization, lemmatization, and POS tagging

✅ Extracted named entities and top keywords

✅ Conducted sentiment analysis over time

✅ Visualized trends using matplotlib and seaborn

🧠 Techniques & Libraries Used
Purpose	Tools & Libraries
Data Handling	pandas, numpy
NLP Core	nltk, spaCy
Visualization	matplotlib, seaborn
Sentiment Analysis	nltk.sentiment.vader
Normalization	Lowercasing, token unification, lemmatization
POS Tagging & NER	nltk, spaCy

🔍 Key Features
📅 Temporal Sentiment Trends
Sentiment score evolution from 2019 to 2023

Visualized sentiment polarity by month/year

🗣️ Token Frequency Analysis
Top tokens after removing stopwords

Unified semantically similar tokens (e.g., "covid" + "coronavirus")

📊 Headline Complexity
Avg. tokens per headline

Distribution of sentence/token counts

📈 Sample Visuals
📦 Token frequency bar charts

📉 Sentiment score over time

📊 Box plots of token counts per month

🔠 Named Entity Recognition (NER) counts

More advanced topics such as topic modeling (LDA) and transformer-based models (like DistilBERT) may be added in future iterations.

🚀 Getting Started
Clone the repo

bash
Copy
Edit
git clone https://github.com/emrehannn/NLPNewsAnalyzer.git
cd nlp-news-analysis
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Launch JupyterLab

bash
Copy
Edit
jupyter lab
🙋‍♂️ Author
Emre — solo developer passionate about data science and NLP.
Feel free to connect or contribute!

🧭 Future Plans
 Integrate topic modeling (LDA)

 Add more entity co-occurrence graphs

 Compare results using transformer models like DistilBERT
