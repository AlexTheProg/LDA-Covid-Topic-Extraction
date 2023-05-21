import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# combine the csv files into a single dataframe for fakenews and for real news
fake_combined_df = pd.DataFrame()
real_combine_df = pd.DataFrame()

fake_news_claims_csv_files = ['../data/ClaimFakeCovid-19_5.csv', '../data/ClaimFakeCovid-19_7.csv']

for file in fake_news_claims_csv_files:
    df = pd.read_csv(file)
    fake_combined_df = pd.concat([fake_combined_df, df])

stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])

    return text


fake_combined_df['title'] = fake_combined_df['title'].apply(preprocess)

# Step 5: Create a document-term matrix using CountVectorizer
vectorizer = CountVectorizer(max_features=1000, max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(fake_combined_df['title'])

# Step 6: Use LDA to identify the topics in the text
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(dtm)

# Step 7: Print the top 10 words for each topic
feature_names = vectorizer.get_feature_names_out()

for i, topic in enumerate(lda_model.components_):
    print(f"Topic {i}:")
    top_words_indices = topic.argsort()[-10:]
    top_words = [feature_names[index] for index in top_words_indices]
    print(top_words)