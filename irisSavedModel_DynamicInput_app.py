# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:49:52 2024

@author: Dr Narayana
"""

# import necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import pickle


### *********** THE MODEL ******************

with open('rf_model.pkl' , 'rb') as f:
    savedModel = pickle.load(f)


## ******************* THE web APP ************************
# title and description:
st.title('Classifying Iris Flowers')
st.markdown('Toy model to play with the iris flowers dataset and classify the three Species into \
     (setosa, versicolor, virginica) based on their sepal/petal \
    and length/width.')

# features sliders for the four plant features:
st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Pepal characteristics")
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

st.text('Hi')
label = ''
# prediction button
if st.button("Predict type of Iris"):
    result = savedModel.predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    label = result[0]
    st.text(result[0])

newData = pd.DataFrame([[sepal_l, sepal_w, petal_l, petal_w, label]], columns=['sl', 'sw', 'pl', 'pw', 'prediction'])
newData.to_csv('newDf.csv')

st.text('')
st.text('')
st.markdown(
    '`Initial code was developed by` [abc](https://twitter.com/abc) | \
         `Code modification and update by:` [bcd](https://github.com/bcd/iris-streamlit/tree/main)')
