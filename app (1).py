import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

import pandas as pd

st.title("Fake News Detection")

df = pd.read_csv("Fake_Real_Data.csv")

df['label'] = df['label'].map({'Fake': 0, 'Real': 1})

x = df['Text']

y = df['label']

model = Pipeline([

  ('v', TfidfVectorizer()),

  ('m', MultinomialNB())

])

model.fit(x, y)

user_input = st.text_area("Enter the news:")

model.predict([user_input])

if st.button('Analyse'):

  pred = model.predict([user_input])

  if pred[0] == 1:

    st.success("The news is Real")

  else:

    st.error("The news is Fake")

