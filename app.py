import streamlit as st
from items.project_description import project_description
from items.text_description import text_description
from items.data_description_images import image_description
from items.text_classification_models import text_classification_models
# from items.image_classification_models import image_classification_models
from items.ensemble_model import ensemble_model

st.html(
    """
<style>
[data-testid="stSidebarContent"] {
    background: white;
    /* Gradient background */
    color: white; /* Text color */
    padding: 5px; /* Add padding */
}

/* Main content area */
[data-testid="stAppViewContainer"] {
    background: white;
    padding: 5px; /* Add padding for the main content */
    border-radius: 5px; /* Add rounded corners */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
}

/* Apply Times New Roman font style globally */
body {
    font-family: 'Roboto', sans-serif;
    font-size: 16px; /* Set the font size */
    color: black; /* Set text color */   
}

/* style other elements globally */
h1, h2, h3 {
    font-family: 'Roboto', sans-serif;
    color: black; /* Set a color for headers */
    width: 100% !important;
}

/* Customize the sidebar text */
[data-testid="stSidebarContent"] {
    font-family: 'Roboto', sans-serif;
    color: black;
}

/* Change the text color of the entire sidebar */
[data-testid="stSidebar"] {
    color: black !important;
}

/* Change the color of the radio button labels */
.stRadio label {
    color: black !important;
}

/* Change the color of the radio button option text */
.stRadio div {
    color: black !important;
}

/* Change the text color for the entire main content area */
body {
    color: black !important;
}

/* Change the color of text in markdown and other text elements */
.stMarkdown, .stText {
    color: black !important;
}

/* Adjust the width of the main content area */
div.main > div {
    width: 80% !important;
    margin: 0 auto;  /* Center the content */
}

</style>
"""
)

st.sidebar.image("images/rakuten_logo.png", use_container_width = False)

menu = st.sidebar.radio("Menu", ["Poject Description",
                             "Data Description: Text",
                             "Data Description: Images",
                             "Text Classification Models",
                             #"Image Classification Models",
                             "Ensemble Classification Models"],
                        label_visibility = "collapsed")

if menu == "Poject Description":
    project_description()
    
elif menu == "Data Description: Text":
    text_description()

elif menu == "Data Description: Images":
    image_description()
    
elif menu == "Text Classification Models":
    text_classification_models()

# elif menu == "Image Classification Models":
#     image_classification_models()

elif menu == "Ensemble Classification Models":
    ensemble_model()
