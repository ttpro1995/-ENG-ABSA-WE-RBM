# CHOOSE LANGUAGE OF DATASET HERE
# Language eng or viet
LANGUAGE = "viet"

ENG_DATASET_FOLDER_DIR = '../dataset'
VIET_DATASET_FOLDER_DIR = '../dataset_viet'

ENG_RESULT_FOLDER_DIR = '../result'
VIET_RESULT_FOLDER_DIR = '../result_viet'

if LANGUAGE == "eng":
    DATASET_FOLDER_DIR = ENG_DATASET_FOLDER_DIR
    RESULT_FOLDER_DIR = ENG_RESULT_FOLDER_DIR
elif LANGUAGE == "viet":
    DATASET_FOLDER_DIR = VIET_DATASET_FOLDER_DIR
    RESULT_FOLDER_DIR = VIET_RESULT_FOLDER_DIR

Classified_Corpus = 'Classified_Corpus.xml'
Classified_Corpus_lower = "Classified_Corpus_lower.xml"
lower = "_lower"
full_sentiment_data_raw = 'full_sentiment_data_raw.txt'
full_sentiment_labels_raw = 'full_sentiment_labels_raw.txt'

full_aspect_data_raw = 'full_aspect_data_raw.txt'
full_aspect_labels_raw = 'full_aspect_labels_raw.txt'

data_vector = "data_vector.p"
normalize_data = "normalize_data.p"

Output_FSA = 'Output_FSA.txt'

result_SVM_sentiment = 'result_SVM_sentiment.txt'
Word2Vec_ENG_model = 'Word2Vec_ENG.model'

SentiWordNet = 'SentiWordNet_3.0.0.txt'

