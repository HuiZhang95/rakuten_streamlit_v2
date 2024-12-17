import streamlit as st
from PIL import Image

def ensemble_model():
    
    st.markdown("<h3>Image Ensemble Models</h3>", unsafe_allow_html = True)
    with st.expander("Click here for details"):

        st.write("The Classifier head was removed from each image classification model and the input "
                 "was fit such that it was suitable input for each model in a new CustomImageClassifier "
                 "CNN. The output feature maps were then concatinated and sent into a linear classifier "
                 "head which output the results. The stacked image classifier achieved a 0.02 increase "
                 "over each image classifier on it's own.")

        with Image.open("images/stacked_image_classification_report.jpg") as img:
            st.image(img)

        st.markdown("<h3>Feature Fusion Approach</h3>", unsafe_allow_html = True)

    st.markdown("<h3>Feature Fusion Approach</h3>", unsafe_allow_html = True)
    with st.expander("Click here for details"):

        st.write("Data from various modalities were processed using different models. "
                 "In this study, pre-trained RoBERTa and BERT model was reweighted with the text data, "
                 "while a pre-trained custom ResNet50 model, EfficientNetV2 model, and a ViT model "
                 "were reweighted using the image data.")

        st.write("The Classifier head was removed from each image and text NN and the dataloader was "
                 "build such that it would split the text and image data appropriately. Each of the "
                 "models, BERT, roBERTa, ResNet50, EfficientNetV2, and ViT were given matching "
                 "input batches and the feature outputs were concatinated (fused) before being sent to a "
                 "classifier head. This model performed better than text only models by approximately "
                 "0.03 (0.85) F1-Score, and significantly better than image only models.")

        with Image.open("images/stacked_image_text_classifier.png") as img:
            st.image(img)
                 
        st.markdown("<h3>SVM</h3>", unsafe_allow_html = True)
        st.write("<b>Step 1:</b> Extract intermediate features from the deepest layer of the trained models "
                 "prior to the classifier head. 2,815 features for images and 2,407 features for text. <br>"
                 "<b>Step 2:</b> Scale text and image features separately to a range from -1 to 1. <br>"
                 "<b>Step 3:</b> Concatenated text and image features.<br>"
                 "<b>Step 4:</b> Train a SVM with gridsearch", unsafe_allow_html = True)
        
        img = Image.open("images/feature_fusion procedure.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Results</h3>", unsafe_allow_html = True)
        st.write("The weighted F1-score was 0.8725")
        img = Image.open("images/feature_fusion result.jpg")
        st.image(img, use_container_width = True)

        img = Image.open("images/feature_fusion comparing models.jpg")
        st.image(img, use_container_width = True)

    st.markdown("<h3>Voting Classifier</h3>", unsafe_allow_html = True)
    with st.expander("Click here for details"):
        st.write("The previous approaches provided an improvement over single models. However, we wanted "
                 "to see if the addition of a voting classifier based on NN outputs could improve the "
                 "results even more. We combined a variety of models trained on the feature outputs of "
                 "the NN models to make classification judgements. We combined linear regression, XGBoost, "
                 "SVC, Random Forests, and K-Nearest Neighbors Classifiers because they all represent unique and "
                 "different stragiest of making classifications. Therefore, we believed that they could "
                 "each contribute some novel weighting to the classification which would improve the overall "
                 "output. A Figure of the classifiers is presented below.")

        with Image.open("images/voting_classifier_layout.png") as img:
            st.image(img)

        st.write("The results were among the highest, at a total of 0.87. A printout of the classification "
                 "report is found below.")

        with Image.open("images/voting_classifier_scores.png") as img:
            st.image(img)

