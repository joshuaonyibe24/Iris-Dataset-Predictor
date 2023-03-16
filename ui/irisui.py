# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:35:09 2023

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from joblib import load
import streamlit as st

model = load('../model/irislog_model.joblib')


### This is the function/method that handles 
def prediction(sepallength, sepalwidth, petallength, petalwidth):
    prediction = model.predict(np.array([[sepallength, sepalwidth, petallength, petalwidth]]))
    return prediction

# function to create the ui
def main():
    st.title("iris dataset Prediction")
    st.header("flower type")
    petallength = st.number_input('Enter PetalLength detail: ')
    sepallength = st.number_input('Enter SepalLength detail: ')
    sepalwidth = st.number_input('Enter SepalWidth detail: ')
    petalwidth = st.number_input('Enter PetalWidth detail: ')
    
    button = st.button('Predict')
    
    result = ''
    
    
    if (button):
        result = prediction(sepallength, sepalwidth, petallength, petalwidth)
        if result == 0:
            st.success('This is a Setosa flower')
            
        elif result == 1:
            st.success('This is Versicolor flower')
        
        elif result == 2:
            st.success('This is Virginica flower')
            
        else:
            st.success('not available')

if __name__ == '__main__':
    main()