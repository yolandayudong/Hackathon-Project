import re
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#nltk.download('wordnet')
#nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]
    
    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    
    text = text.split()

    text=[y.lower() for y in text]
    
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan
            
def normalize_text(df):
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : Removing_numbers(text))
    df.Text=df.Text.apply(lambda text : Removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : Removing_urls(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence

def train_model(model, data, targets):
    """
    Train a model on the given data and targets.
    
    Parameters:
    model (sklearn model): The model to be trained.
    data (list of str): The input data.
    targets (list of str): The targets.
    
    Returns:
    Pipeline: The trained model as a Pipeline object.
    """
    # Create a Pipeline object with a TfidfVectorizer and the given model
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('vect',TfidfVectorizer()),
                         ('clf', model)])
    # Fit the model on the data and targets
    text_clf.fit(data, targets)
    return text_clf

def get_F1(trained_model,X,y):
    """
    Get the F1 score for the given model on the given data and targets.
    
    Parameters:
    trained_model (sklearn model): The trained model.
    X (list of str): The input data.
    y (list of str): The targets.
    
    Returns:
    array: The F1 score for each class.
    """
    # Make predictions on the input data using the trained model
    predicted=trained_model.predict(X)
    # Calculate the F1 score for the predictions
    f1=f1_score(y,predicted, average=None)
    # Return the F1 score
    return f1

import pandas as pd
df_train = pd.read_csv('C:/Users/yud/Downloads/archive/train.csv', names=['Text','Emotion'], sep=',')
df_train.head()
print(df_train.shape)


import pandas as pd
df_valid = pd.read_csv('C:/Users/yud/Downloads/archive/valid.csv', names=['Text','Emotion'], sep=',')
df_valid.head()
print(df_valid.shape)

import pandas as pd
df_test = pd.read_csv('C:/Users/yud/Downloads/archive/test.csv', names=['Text','Emotion'], sep=',')
df_test.head()
print(df_test.shape)

df_train= normalize_text(df_train)
df_test= normalize_text(df_test)
df_valid= normalize_text(df_valid)

X_train = df_train['Text']
y_train = df_train['Emotion']

X_test = df_test['Text']
y_test = df_test['Emotion']

X_val = df_valid['Text']
y_val = df_valid['Emotion']

tokenizer = Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))
tokenizer.texts_to_matrix(X_train[0].split()).shape

from sklearn.linear_model import LogisticRegression
log_reg = train_model(LogisticRegression(solver='liblinear',random_state = 0), X_train, y_train)

text1 = "I want to go to Peru this summer ahhhhhhh! Hopefully! Yesyesyes! I miss it over there!!!"
text2 = "Hates headaches! Maybe I'm not ready to rock"

#Make a single prediction
print(log_reg.predict(["I want to go to Peru this summer ahhhhhhh! Hopefully! Yesyesyes! I miss it over there!!!"]))
print(log_reg.predict(["Hates headaches! Maybe I'm not ready to rock"]))

#test the model with the test data

#calculate the accuracy
log_reg_accuracy = accuracy_score(y_test, log_reg.predict(X_test))
print('Accuracy: ', log_reg_accuracy,'\n')

#calculate the F1 score
f1_Score = get_F1(log_reg,X_test,y_test)
pd.DataFrame(f1_Score, index=df_train.Emotion.unique(), columns=['F1 score'])

print(classification_report(y_test, log_reg.predict(X_test)))

from sklearn.tree import DecisionTreeClassifier
DT = train_model(DecisionTreeClassifier(random_state = 0), X_train, y_train)
DT_accuracy = accuracy_score(y_test, DT.predict(X_test))
print('Accuracy: ', DT_accuracy,'\n')

#calculate the F1 score
f1_Score = get_F1(DT,X_test,y_test)
print(classification_report(y_test, DT.predict(X_test)))

from sklearn.svm import SVC
SVM = train_model(SVC(random_state = 0), X_train, y_train)
SVM_accuracy = accuracy_score(y_test, SVM.predict(X_test))
print('Accuracy: ',SVM_accuracy,'\n')

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
#calculate the F1 score
f1_Score = get_F1(SVM,X_test,y_test)
print(f1_Score)
print(classification_report(y_test, DT.predict(X_test)))

from sklearn.ensemble import RandomForestClassifier
RF = train_model(RandomForestClassifier(random_state = 0), X_train, y_train)
RF_accuracy = accuracy_score(y_test, RF.predict(X_test))
print('Accuracy: ',RF_accuracy,'\n')

#calculate the F1 score
f1_Score = get_F1(RF,X_test,y_test)
print(f1_Score)
print(classification_report(y_test, RF.predict(X_test)))

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
# Load the pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2ForSequenceClassification.from_pretrained("gpt2")

def predict_sentiment(text):
    # Encode the text
    encoded_text = tokenizer.encode(text, return_tensors="pt")
    # Predict the sentiment
    sentiment = model(encoded_text)[0]
    # Decode the sentiment
    return sentiment.argmax().item()

# Test the model
text1 = "I want to go to Peru this summer ahhhhhhh! Hopefully! Yesyesyes! I miss it over there!!!"
text2 = "Hates headaches! Maybe I'm not ready to rock"

# 1 positive, 0 negative
print("Sentiment: ",predict_sentiment("I want to go to Peru this summer ahhhhhhh! Hopefully! Yesyesyes! I miss it over there!!!"))
print("Sentiment: ",predict_sentiment("Hates headaches! Maybe I'm not ready to rock"))

