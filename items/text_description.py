import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text
import streamlit as st
import pandas as pd
import numpy as np
import ftfy
import unicodedata
from bs4 import BeautifulSoup
import re
from lingua import Language, LanguageDetectorBuilder
from langdetect import detect_langs, DetectorFactory
from deep_translator import GoogleTranslator
import os
import random
from PIL import Image
import nltk
from cleantext import clean
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt_tab')

def text_description():
	
	st.write("### Preprocessing/Translation")

	with st.expander('Show details about preprocessing and translation'):

                work_dir = os.getcwd()
                # dataframe_dir = '\\datasets\\'
                # image_dir = '\\images\\'
                dataframe_dir = '/datasets/'
                image_dir = '/images/'

                df_translated = pd.read_csv( work_dir + dataframe_dir + 'rakuten_train_designation_translation.csv', encoding='utf8')
                df_translated['designation'] = df_translated['designation'].fillna(str())
                df_translated['designation'] = df_translated['designation'].astype('str')
                df_translated['designation_translation'] = df_translated['designation_translation'].fillna(str())
                df_translated['designation_translation'] = df_translated['designation_translation'].astype('str')
                df_translated['designation_filtered'] = df_translated['designation_filtered'].fillna(str())
                df_translated['designation_filtered'] = df_translated['designation_filtered'].astype('str')

                # filter functions to apply on string input

                # filter that does nothing, it is needed for the looping
                def filter_nofilter( text ):
                        return text

                # filter that does nothing, it is needed for the looping
                def filter_designation( text ):
                        return

                # filter that does nothing, it is needed for the looping
                def filter_description( text ):
                        return

                # use ftfy to remove/replace wrong character set encodings
                def filter_ftfy( text ):
                        fixed_text = ftfy.fix_text( text )
                        return fixed_text

                # use unicodedata to remove/replace no printing characters
                def filter_unicodedata( row ):
                        fixed_text = ''.join( char for char in row if unicodedata.category(char) not in ["Cf", "Cc", "Zl", "Zp"] )
                        return fixed_text

                def filter_bs4( text ):
                        return BeautifulSoup( text, features = 'html.parser' ).get_text( ' ' )

                # remove usernames from the end of the string, pattern is: @'only no space/tab characters'EOL
                def filter_username( text ):
                        return re.sub(r"@[^\s]*$", "", text )

                # find multiple space/tab and replace it with just a single space
                def filter_whitespace( text ):
                        re_combined_space = re.compile(r"\s+")
                        cleaned_text = re_combined_space.sub(" ", text).strip()
                        return cleaned_text

                def filter_norm(text):
                        return unicodedata.normalize('NFKC', text)

                def filter_charnorm(text):
                        # Step 1: Detect encoding with charset-normalizer
                        byte_data = str.encode( text )
                        detected = from_bytes(byte_data).best()
                        if detected:
                                try:
                                        # Step 2: Decode using the detected encoding
                                        #print( 'detect', detected )
                                        decoded_text = str( detected )
                                        return decoded_text
                                except Exception as e:
                                        print(f"Decoding or fixing failed: {e}")
                                        return None
                        else:
                                print("Encoding could not be detected.")
                                return None
                                
                filter_func_list = [ filter_nofilter, filter_ftfy, filter_unicodedata, filter_bs4, filter_norm, filter_username, filter_whitespace ]

                # define a function that will use all the functions in filter_func_list one after the other
                def filter_stacked( text ):
                        filtered = text
                        for filter_func in filter_func_list:
                                filtered = filter_func( filtered )
                        return filtered

                languages_list = [Language.FRENCH,
                                         Language.ENGLISH,  
                                         Language.GERMAN,
                                         Language.SPANISH,
                                         Language.ITALIAN,
                                         Language.SWEDISH,
                                         Language.POLISH,
                                         Language.DUTCH,
                                         Language.ROMANIAN,
                                         Language.PORTUGUESE,
                                         Language.JAPANESE
                                        ]
                languages_dict = { 'FRENCH': 'fr', 'ENGLISH': 'en', 'GERMAN': 'de', 'SPANISH': 'es',
                                                  'ITALIAN': 'it', 'SWEDISH': 'sv', 'POLISH': 'pl', 'ROMANIAN': 'ro', 'PORTUGUESE': 'pt', 'JAPANESE': 'ja', 'DUTCH': 'nl'  }




                def language_confidences( line ):


                        ########### lingua ##################
                        detector = LanguageDetectorBuilder.from_languages(*languages_list).with_preloaded_language_models().build()
                        confidences = detector.compute_language_confidence_values( line )
                        lingua_languages = []
                        lingua_confidences = []
                        for confidence in confidences:
                                lingua_languages.append( languages_dict[confidence.language.name] )
                                lingua_confidences.append( confidence.value )
                        ########### lingua ##################
                        
                        ############# langdetect #################
                        DetectorFactory.seed = 2
                        # expecting not more than 10 langauges
                        langdetect_languages = 10 * ['unknown']
                        langdetect_confidences = 10 * [0]
                        try:
                                languages = detect_langs( line )
                                for i, language in enumerate(languages):
                                        langdetect_languages[i] = language.lang
                                        langdetect_confidences[i] = language.prob
                                        if i == 3:
                                                break
                        except Exception as detect_langs_error:
                                langdetect_languages = 10 * ['unknown']
                                langdetect_confidences = 10 * [0]
                        
                        ############# langdetect #################
                        
                        ############### google handhelds ###########################
                        work_dir = os.getcwd()
                        model_path = work_dir + '\\language_detector.tflite'
                        # expecting not more than 10 langauges
                        google_languages = 10 * ['unknown']
                        google_confidences = 10 * [0]
                        base_options = python.BaseOptions(model_asset_path=model_path)
                        options = text.LanguageDetectorOptions(base_options=base_options)
                        with python.text.LanguageDetector.create_from_options(options) as detector:
                                google_language = detector.detect( line )
                        try:
                                google_languages[0] = google_language.detections[0].language_code
                                google_confidences[0] = google_language.detections[0].probability
                        except:
                                pass
                        
                        first_lang = [lingua_languages[0], langdetect_languages[0], google_languages[0]]
                        #print( first_lang )
                        first_conf = [np.round( lingua_confidences[0], 2 ), np.round( langdetect_confidences[0], 2 ), np.round( google_confidences[0], 2 )]
                        #print( first_conf )
                        second_lang = [lingua_languages[1], langdetect_languages[1], google_languages[1]]
                        #print( second_lang )
                        second_conf = [np.round( lingua_confidences[1], 2 ), np.round( langdetect_confidences[1], 2 ), np.round( google_confidences[1], 2 )]
                        #print( second_conf ) 
                        first_lang_sort = [x for _,x in sorted(zip(first_conf,first_lang) , reverse=True)]
                        #print( first_lang_sort )
                        first_conf_sort = sorted( first_conf, reverse=True )
                        #print( first_conf_sort )
                        second_lang_sort = [x for _,x in sorted(zip(second_conf,second_lang) , reverse=True)]
                        #print( second_lang_sort )
                        second_conf_sort = sorted( second_conf, reverse=True )
                        #print( second_conf_sort )
                        ############### google handhelds ###########################
                        
                        return ( first_lang_sort, first_conf_sort, second_lang_sort, second_conf_sort )



                def translate_to_en( text ):
                        #print( 'start translation' )
                        #print( text )
                        #print( '-----' )
                        translation_counter = 0
                        confidence_threshold_english = 0.95
                        translated_text = text
                        translation_error_text = str()
                        got_translated = False
                        before_trans = language_confidences( text )
                        #print( '0', before_trans )
                        after_trans = ()
                        # both detection found the same language
                        if( before_trans[0][0] == before_trans[0][1] == before_trans[0][2] == 'en'):
                                #print( 'all english')
                                if( min(before_trans[1]) < confidence_threshold_english ):
                                        if (before_trans[2][0] != 'unknown') and (before_trans[2][0] in languages_dict.values()):
                                                #print( 'english, first try', before_trans[2][0] )
                                                translation_counter += 1
                                                try:
                                                        translated_text = GoogleTranslator(source = before_trans[2][0], target='english').translate( text )
                                                except Exception as translation_error:
                                                        #print( 'translation error 1' )
                                                   # print( translation_error )
                                                        translation_error_text = 'translation error'
                                                after_trans = language_confidences( translated_text )
                                                #print( translated_text )
                                                
                                                if( before_trans == after_trans or before_trans[1][1] > 0.2 ):
                                                        if before_trans[2][1] != 'unknown' and (before_trans[2][1] in languages_dict.values()):
                                                                #print( 'english, second try', before_trans[2][1] )
                                                                translation_counter += 1
                                                                try:
                                                                        translated_text = GoogleTranslator(source = before_trans[2][1], target='english').translate( text )
                                                                except Exception as translation_error:
                                                                        #print( 'translation error 2' )
                                                                        translation_error_text = 'translation error'
                                                                after_trans = language_confidences( translated_text )
                                                                #print( translated_text )
                                                        
                                                        if( before_trans == after_trans ):
                                                                if before_trans[2][2] != 'unknown' and (before_trans[2][2] in languages_dict.values()):
                                                                        #print( 'english, third try', before_trans[2][2] )
                                                                        translation_counter += 1
                                                                        try:
                                                                                translated_text = GoogleTranslator(source = before_trans[2][2], target='english').translate( text )
                                                                        except Exception as translation_error:
                                                                                #print( 'translation error 3' )
                                                                                translation_error_text = 'translation error'
                                                                        after_trans = language_confidences( translated_text )
                                                                        #print( translated_text )
                        if translated_text != text:
                                        got_translated = True

                        else:
                                #print ( 'other language')
                                if before_trans[0][0] != 'unknown' and (before_trans[0][0] in languages_dict.values()):
                                        #print( 'other, first try', before_trans[0][0] )
                                        translation_counter += 1
                                        try:
                                                #print( before_trans[0][0] )
                                                #print( text )
                                                translated_text = GoogleTranslator(source = before_trans[0][0], target='english').translate( text )
                                        except Exception as translation_error:
                                                #print( 'translation error 4' )
                                                translation_error_text = 'translation error'
                                        after_trans = language_confidences( translated_text )
                                        #print( translated_text )
                                        
                                        if( before_trans == after_trans or before_trans[1][1] > 0.2 ):
                                                translation_error_text = str()
                                                if before_trans[0][1] != 'unknown' and (before_trans[0][1] in languages_dict.values()):
                                                        #print( 'other, second try', before_trans[0][1] )
                                                        translation_counter += 1
                                                        try:
                                                                translated_text = GoogleTranslator(source = before_trans[0][1], target='english').translate( text )
                                                        except Exception as translation_error:
                                                                #print( 'translation error 5' )
                                                                translation_error_text = 'translation error'
                                                        after_trans = language_confidences( translated_text )
                                                        #print( translated_text )
                                                        
                                                        if( before_trans == after_trans ):
                                                                if before_trans[0][2] != 'unknown' and (before_trans[0][2] in languages_dict.values()):
                                                                        #print( 'other, third try', before_trans[0][2] )
                                                                        translation_counter += 1
                                                                        try:
                                                                                translated_text = GoogleTranslator(source = before_trans[0][2], target='english').translate( text )
                                                                        except Exception as translation_error:
                                                                                #print( 'translation error 6' )
                                                                                translation_error_text = 'translation error'
                                                                        after_trans = language_confidences( translated_text )
                                                                        #print( translated_text )
                                if translated_text != text:
                                        got_translated = True
                                        

                        #last step always try French once again
                        #only try to translate if no translation error occured before
                        #if len(str(translation_error_text)) == 0:
                        #print ( 'double check french')
                        translation_counter += 1
                        try:
                                translated_text_temp = GoogleTranslator(source = 'fr', target='english').translate( translated_text )
                                if ( len( translated_text_temp ) / len( translated_text ) ) > 0.7:
                                        translated_text = translated_text_temp
                                else:
                                        #print( 'translation error 7' )
                                        translation_error_text = 'translation error'
                        except Exception as translation_error:
                                #print( 'translation error 8' )
                                translation_error_text = 'translation error'
                        #print( translated_text )
                                
                        if translated_text == text:
                                #print ( 'double all others')
                                for language in languages_dict.values():
                                        translation_counter += 1
                                        try:
                                                translated_text_temp = GoogleTranslator(source = 'fr', target='english').translate( translated_text )
                                                if ( len( translated_text_temp ) / len( translated_text ) ) > 0.7:
                                                        translated_text = translated_text_temp
                                                else:
                                                        #print( 'translation error 9' )
                                                        translation_error_text = 'translation error' 
                                        except Exception as translation_error:
                                                #print( 'translation error 10' )
                                                translation_error_text = 'translation error' 
                                #print( translated_text )
                        #print( 'last check' )
                        #print( translated_text )
                        if translation_error_text == str():
                                return translated_text
                        else:
                                return str()
                                        
                filtered = str()
                translated = str()
                
                st.write( 'In our text input columns \'designation\' and \'description\', there are several formatting issues we needed to address:'
                                        '\n'
                                        '1. fragments of HTML markup language. e.g. tags like: <br />Capacité de charge jusqu&#39;à 150 kg<br />'
                                        '\n'
                                        '2. non utf-8 characters, e.g. characters encoded in cp1252/Windows-1252 and others'
                                        '\n'
                                        '3. numerous characters that serve formatting, directional or layout purposes (invisible characters or non-printing characters), e.g. (\\u200e, \\u200b, \\xad).'
                                        '\n'
                                        '\n'
                                        'We used the following python packages for filtering our unput strings:'
                                        '\n'
                                        '1. beautiful soup'
                                        '\n'
                                        '2. ftfy'
                                        '\n'
                                        '3. unicodedata'
                                        )
                                        
                                                
                
                st.write( 'The filtered strings were afterwards translated with google translate using the deep-translator api. '
                                'In order to detect all languages of multi-languages strings, the confidences for the detected languages were evaluated. '
                                'With this method most possible parts of our input strings were translated to the target language English.'
                                'The distributions of the detected languages of our input strings are shown in the image below.'
                                )
                
                
                img = Image.open( work_dir + image_dir + 'languages.01.png' )
                st.image(img, use_container_width = True)
                
                
                st.write( 'Demonstration of the filtering and translation process:')
                # Initialize session state for selected string and results
                if 'selected_string' not in st.session_state:
                        st.session_state['selected_string'] = None
                if 'filtered' not in st.session_state:
                        st.session_state['filtered'] = None
                if 'translated' not in st.session_state:
                        st.session_state['translated'] = None

                # Button to select a random string and filter it
                if st.button('Select string from \'designation\' out of 20 examples'):
                        # Select a random string from the 'designation' column
                        #st.session_state['selected_string'] = random.choice(df_translated['designation'])
                        rows = range(0,20)
                        row = random.choice(rows)
                        print( df_translated.iloc[[row]]['designation'] )
                        st.session_state['selected_string'] = df_translated['designation'][row]
                        # Execute the filter function
                        st.session_state['filtered'] = filter_stacked(st.session_state['selected_string'])

                # Display the selected string and filtered result (if available)
                if st.session_state['selected_string']:
                        st.text_area("Selected string", value=st.session_state['selected_string'], height=100)
                if st.session_state['filtered']:
                        st.text_area("Result of filtering", value=st.session_state['filtered'], height=100)

                # Button to translate the filtered string
                if st.button("Translate filtered string"):
                        if st.session_state['filtered']:
                                # Execute the translate function
                                st.session_state['translated'] = translate_to_en(st.session_state['filtered'])

                # Display the translated result (if available)
                if st.session_state['translated']:
                        st.text_area("Translated result", value=st.session_state['translated'], height=100)
                
                st.write( 'The distributions of the detected languages of our translated strings are shown in the image below.' )
                img = Image.open( work_dir + image_dir + 'languages.02.png' )
                st.image(img, use_container_width = True)

	st.markdown("<h3>After Translation</h3>", unsafe_allow_html = True)
	with st.expander("click here for details"):
		st.markdown("<h3>Data Cleaning Approaches after Translation</h3>", unsafe_allow_html = True)
		st.write("1. Each word was further stemmed and lemmatized. \n"
                         "2. Stop words were updated and removed from the remaining text. \n")
		st.markdown("<h3>Word Cloud of Each Category</h3>", unsafe_allow_html = True)
		img = Image.open("images/text wordcloud.jpg")
		st.image(img, use_container_width = True)
		st.markdown("<h3>Top 10 Most Frequent Words of Each Category</h3>", unsafe_allow_html = True)
		img = Image.open("images/word top10.jpg")
		st.image(img, use_container_width = True)
		stop_words = {'a','about','above','after','again','against','ain','all','am','an','and','any','are','aren',"aren't",'as','at',
            'be','because','been','before','being','below','between','both','but','by','can','couldn',"couldn't",'d','did','didn',"didn't",
            'do','does','doesn',"doesn't",'doing','don',"don't",'down','during','each','few','for','from','further','had','hadn',"hadn't",
            'has','hasn',"hasn't",'have','haven',"haven't",'having','he','her','here','hers','herself','him','himself','his','how','i','if',
            'in','into','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",
            'my','myself','needn',"needn't",'no','nor','not','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out',
            'over','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than',
            'that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too',
            'under','until','up','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who',
            'whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself',
            'yourselves'}
		stop_words.update(['the', 'and','for', 'from', 'was', 'what', 'with', 'this', \
                            'that',  'don', 'pure', 'lot', 'are', 'who', 'more', 'will', 'tab', \
                            'each' , 'would', 'but', 'not','its','all','your', 'last','over', \
                                'are','you', 'can', 'above', 'his','she','ready', 'yes', \
                                'size'])
		stop_words.remove("no")
		stemmer = EnglishStemmer()
		wordnet_lemmatizer = WordNetLemmatizer()
		def lemmatization(words) :
			output = []
			for string in words :
				lemma = wordnet_lemmatizer.lemmatize(string)
				if (lemma not in output) : output.append(lemma)
			return output
		def unicode_to_ascii(s):
			return ''.join(c for c in unicodedata.normalize('NFD', s)
				  if unicodedata.category(c) != 'Mn')
		def clean_text(text):
			text = clean(
                    text = text,
                    fix_unicode = True,
                    to_ascii = True,
                    lower = True,
                    replace_with_url = ' ',
                    replace_with_email = ' ',
                    lang = 'en'
                )
			text = re.sub(r"[^A-Za-z0-9\s]+", " ", text)
			text = re.sub(r'(.)\1{3,}',r'\1', text)
			text = " ".join(text.split())
			w = str(text)
			w = unicode_to_ascii(w.lower().strip())
			w = re.sub(r"([?.!,¿])", r" \1 ", w)
			w = re.sub(r'[" "]+', " ", w)
			w = re.sub(r"[^a-zA-Z0-9?.!]+", " ", w)
			w = re.sub(r'\b\w{0,2}\b', '', w)
			words = word_tokenize(w.strip())
			words2 = [stemmer.stem(word) for word in words]
			words3 = lemmatization(words2)
			words = [word for word in words3 if word not in stop_words]
			if len(words) < 1: # sometimes, all words are removed
				return w
			else:
				return ' '.join(words).strip()
			
		user_input_word = st.text_input("Input a sentense to clean: ", 'Merry Christmas!')
		output_st = clean_text(user_input_word)
		st.write('Here is the sentence after cleaning :\n', output_st)
