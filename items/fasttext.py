import streamlit as st

def fasttext():
	
	import streamlit as st
	import pandas as pd
	import numpy as np

	import os
	
	import random
	
	from PIL import Image
	
	
	work_dir = os.getcwd()
	dataframe_dir = '\\dataframes\\'
	image_dir = '\\images\\'

	st.header("Text classification with fastText")
	
	st.write( 'fastText offers the possibility to represent words by vectors that '
			'can be used in modern machine learning models. The idea of these vectors is '
			'to capture hidden information about a language, like word analogies or semantics. ' 
			'Furthermore, it offers the possibility to perform classification tasks in order '
			'to assign text to multiple classes. fastText is a quite efficient library and the '
			'adaptation to a new labeled dataset can be performed on a short time scale. It '
			'offers a small set of hyper-parameters for the training on new dataset but does not '
			'offer a large spectrum of fine tuning as in NLP models like BERT. In order to perform the ' 
			'classification with fastText, the translated \'designation\' column was used without any further pre-processing.'
			)
	st.write( 'Overall the classification with the fastText library offers the possibility for multi-class '
			'text classification. Compared to more complex large NLPs terms of performance, '
			'the weighted F1-score is not as good. An F1-score above 0.80 is still an acceptable '
			'result. The fastText library would be an ideal candidate in an environment of often changing '
			'datasets. In such a setting it would be a good compromise of fast learning and acceptable performance. (s. KPIs below)'
			)
	
	img = Image.open( work_dir + image_dir + 'image_01.png' )
	st.image(img, use_container_width = True)
	
	st.header("Text classification with roBERTa")
	
	st.write( 'In natural language processing NLP tasks, transformer-based models like BERT '
			'have pushed the state-of-the-art in a broad range of fields. The idea behind such '
			'models is to use a pre-trained general-purpose model that was trained on a large '
			'corpus of text input. The pre-trained model weights were re-trained on the custom dataset '
			'in order to adapt to the features of the our data. '
			'This process is called fine-tuning. With this approach one can benefit from the training '
			'on a much larger data basis and usually the training on a custom model could be performed '
			'on a much shorter time-scale compared to training of own models directly from scratch.'
			)
	
	st.write( 'The training of the model is performed in two steps. The first step is to tokenize '
			'the textual input with the corresponding \'RobertaTokenizer\' in order to convert the '
			'text into vectors that the neural networks can process. The training curve for the '
			'accuracy and the loss is depicted in Figure 22. From the learning curve you can see '
			'that such a pre-trained model can quickly adapt to the new dataset up to a '
			'value acc = 0.7 within the first 10 epochs. (s. training curve below)'
			)
	
	
	img = Image.open( work_dir + image_dir + 'image_04.png' )
	st.image(img, use_container_width = True)
		
	st.write( 'The final KPIs are presented below. The class report shows that compared '
			'to the RNN and the fastText model (cf. paragraphs above) the score for all classes has improved and '
			'an overall weighted F1 score of 0.84 could be achieved . This means that the pre-trained '
			'\'roberta-large\' model is capable of better identifying the relevant features of the classes '
			'within our dataset and perform better in the predictions.'
			)
	
	img = Image.open( work_dir + image_dir + 'image_02.png' )
	st.image(img, use_container_width = True)
	

	st.write( 'In order to get a visual representation of the classification '
			'accuracy we performed dimension reduction with PCA and t-SNE on predictions of the train and test dataset. '
			'From the dimension reduction with PCA (see image below) one can see that we achieved a certain degree '
			'of clustering and classes are mostly well separated from each other. '
			'We also performed dimension reduction with t-SNE which is better suited for non-linear relations '
			'in the dataset. In this 2D representation one can see that certain classes are well separated from '
			'each other and some classes even seem to have separated sub-classes. Other classes have a certain '
			'overlap indicated by the black ovals. One can see from the predictions of the test dataset that '
			'there is a stronger crossing especially at the intersections within the marked oval areas, which '
			'is probably the cause of the lower F1-score in certain \'prdtypcodes\'.'

			)

	img = Image.open( work_dir + image_dir + 'image_03.png' )
	st.image(img, use_container_width = True)
