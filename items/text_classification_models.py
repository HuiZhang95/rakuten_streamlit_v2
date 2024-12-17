import streamlit as st
from PIL import Image
import pickle
import numpy as np
import pandas as pd
from items.fasttext import fasttext

def text_classification_models():

    st.markdown("<h3>Fast Text and roBERTa</h3>", unsafe_allow_html = True)

    with st.expander("Click here for details"):
        fasttext()
    
    st.markdown("<h3>Customized Recurrent Neural Network (RNN)</h3>", unsafe_allow_html = True)
    with st.expander("Click here for details"):
        st.markdown("<h3>Background and Procedure</h3>", unsafe_allow_html = True)

        st.write("Recurrent layers are designed to capture sequential relationships within the data. "
                "This is especially important for the text information in the present project. ")
        st.write("Procedure:")
        st.write("Train-test split --> tokenize sequences --> train model")
        st.write("Here is the achitechture of the model:")
        
        img = Image.open("images/RNN achitechture.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Training process</h3>", unsafe_allow_html = True)
        img = Image.open("images/RNN training.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Results</h3>", unsafe_allow_html = True)
        st.write("Weighted F1 score is 0.75")
        img = Image.open("images/RNN result.jpg")
        st.image(img, use_container_width = True)

        img = Image.open("images/RNN result 2.jpg")
        st.image(img, use_container_width = True)

    st.markdown("<h3>SVM with gridsearch</h3>", unsafe_allow_html = True)
    with st.expander("Click here for details"):
        st.markdown("<h3>Background and Procedure</h3>", unsafe_allow_html = True)

        st.write("SVM aims to identify decision boundaries that maximize the margin between categories,"
                 " which helps generalization on unseen data. "
                 "Additionally, SVM uses a subset of training points to define the hyperplane, "
                 "making it memory efficient. This is particularly beneficial for systems with limited resources.. "
                " \n\n"
                "Procedure: \n "
                "TF-IDF transformation --> search for optimal hyperparameters"
                "Here is the gridsearch result:")
        
        img = Image.open("images/SVM model.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Results</h3>", unsafe_allow_html = True)
        st.write("Weighted F1 score is 0.81")
        img = Image.open("images/SVM result.jpg")
        st.image(img, use_container_width = True)

    st.markdown("<h3>Model Testing</h3>", unsafe_allow_html = True)
    with st.expander("Click here to see model demo"):

        # use the st.selectbox() method to choose between the RandomForest classifier, the SVM classifier and the LogisticRegression classifier.
        #Then return to the Streamlit web application to view the select box.
        df = pd.read_csv('{}rnn_svc_result.csv'.format('datasets/'), index_col = 0)
        len_df = len(df)
        index_df = df.index

        user_input_number = st.number_input("Input a number between 0 and {}: ".format(str(len_df)), 0)

        output_st = df.loc[index_df[user_input_number], 'designation']
        st.write('\nHere is the text information from designation column:\n', output_st)

        output_st = df.loc[index_df[user_input_number], 'text']
        st.write('\nHere is the text information from description column:\n', output_st)

        output_st = df.loc[index_df[user_input_number], 'text']
        st.write('\nHere is the text information after pre-processing:\n', output_st)
        
        choice = ['Choose a Model','RNN', 'SVC']
        option = st.selectbox('Choice of the model', choice)
        st.write('The chosen model is :', option)

        if option == 'Choose a Model':
            st.write('Please choose a model.')

        if option == 'RNN':
            output_st = df.loc[index_df[user_input_number], 'y_RNN']
            st.write('The predicted category is: ', output_st)

            output_st = df.loc[index_df[user_input_number], 'y_true']
            st.write('The true category is: ', output_st)
        
        if option == 'SVC':
            output_st = df.loc[index_df[user_input_number], 'y_SVC']
            st.write('The predicted category is: ', output_st)

            output_st = df.loc[index_df[user_input_number], 'y_true']
            st.write('The true category is: ', output_st)


        
            


