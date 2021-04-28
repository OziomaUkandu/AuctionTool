import pandas as pd
import numpy as np
import streamlit as st  

import re
import joblib
import nltk

from nltk.corpus import stopwords
import string
pd.options.mode.chained_assignment = None

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer
import nltk_download_utils

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import sqlite3 




## Display the App description code
st.sidebar.title('Welcome to the Winning Bid Prediction Web App')


######## SQLITE3 DATABASE
# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

#Function to add username/password to the database
def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

#Function to logic the user
def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

#Functions to hash the password
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# SET THE USERNAME AND PASSWORD VARIABLES BOTH AS ADMIN
user1= 'admin'
password1='admin'

#Create a user
create_usertable()
add_userdata(user1,make_hashes(password1))




#Read our dataset
df=pd.read_csv('CombinedData.csv')
df=df.dropna().reset_index()

#Ask the user to introduce the Item's description
user_input = st.text_input("Please enter your item's description", 'DVD Film of a hollywood movie 1971')
#Create a test dataset with the user input
text_dic={'Auction Name':[user_input]}
input_text=pd.DataFrame(text_dic)
test_=input_text


#### Make select boxes of Currency/Cat 1/ Cat 2/Hierarchy
#### Ask the user to select from the available options
currencies= sorted(df['Currency'].unique().tolist())
currency_box = st.selectbox("Select a currency", options=currencies)


hierarchies= sorted(df['Hierarchy'].unique().tolist())
hierarchy_box = st.selectbox("Select a hierarchy", options=hierarchies)

starting_bids= st.slider('Select your Starting Bid', 0, 500, 25)
days= st.slider('Select your bidding day', 0, 10,1)



###Create a Dataframe from the user's selection of categorical/numerical variables
d = {'Days':[days],'Starting Bid':[starting_bids],'Currency': [currency_box], 'Hierarchy': [hierarchy_box]}
input_features2= pd.DataFrame(d)


### Ask the user for username and password
username= st.sidebar.text_input('Username')
password= st.sidebar.text_input('Password', type='password')

#Execute this code after clicking on the login button
if st.sidebar.button('Login to predict'):
     
     #Check if the password and username match in the database
     def check_user(username, password):
        create_usertable()
        hashed_pswd = make_hashes(password)
        result = login_user(username,check_hashes(password,hashed_pswd))
        return result
     result= check_user(username, password)
     #Open the App for the user if the authentication is successful
     if result:
        st.sidebar.success('Active session')
        ## Preprocessing function that includes all the text preprocessing steps
        def text_preprocessing(data): 
            #### what is df2
            df2=pd.DataFrame() 
            #Lower case
            df2["text_lower"] = data["Auction Name"].str.lower()
            
            #Remove punctuation
            PUNCT_TO_REMOVE = string.punctuation
            def remove_punctuation(text):
                """custom function to remove the punctuation"""
                return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
            df2["text_wo_punct"] = df2["text_lower"].apply(lambda text: remove_punctuation(text))
            
            #", ".join(stopwords.words('english'))
            STOPWORDS = set(stopwords.words('english'))
            STOPWORDS2 = set(stopwords.words('german'))
            def remove_stopwords(text):
                """custom function to remove the stopwords"""
                return " ".join([word for word in str(text).split() if word not in STOPWORDS])

            def remove_stopwords2(text):
                """custom function to remove the stopwords"""
                return " ".join([word for word in str(text).split() if word not in STOPWORDS2])

            df2["text_wo_stop"] = df2["text_wo_punct"].apply(lambda text: remove_stopwords(text))
            df2["text_wo_stop"] = df2["text_wo_stop"].apply(lambda text: remove_stopwords2(text))

            #Lemmatizer helps to get to words into their original format
            lemmatizer = WordNetLemmatizer()
            wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
            def lemmatize_words(text):
                pos_tagged_text = nltk.pos_tag(text.split())
                return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
            df2["text_lemmatized"] = df2["text_wo_stop"].apply(lambda text: lemmatize_words(text))
            # Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
            def remove_emoji(string):
                emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
                return emoji_pattern.sub(r'', string)

            df2["remove_emoji"] = df2["text_lemmatized"].apply(lambda text: remove_emoji(text))
            return df2

        #Apply text preprocessing on our user input data
        print('Start text preprocessing')
        train=text_preprocessing(df)
        test_=text_preprocessing(test_)

        print('Start encoding')
        #Encode categorical features in both train and test(user input)
        le = preprocessing.LabelEncoder()
        #Encode currency
        dummies=df.copy() 
        encode_cur=le.fit(dummies['Currency'])
        dummies['Currency']= encode_cur.transform(dummies['Currency'])
        input_features2['Currency']= encode_cur.transform(input_features2['Currency'])
        
        #Encode Hierarchy
        encode_hier=le.fit(dummies['Hierarchy'])
        dummies['Hierarchy']= encode_hier.transform(dummies['Hierarchy'])
        input_features2['Hierarchy']= encode_hier.transform(input_features2['Hierarchy'])
        #Drop unused columns
        dummies=dummies.drop(['index','Unnamed: 0','Auction Name', 'Date', 'Winning Bid', 'Category 1', 'Category 2'], axis=1)
        
        print('Start vectorizing')
        #Load the Tfidvectorizer that we fit on the train data
        tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1, 4), use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words = 'english')

        #Fit the vectorizer on both train and test(user input)
        tfv_f=tfv.fit(list(train['remove_emoji']),list(test_['remove_emoji']))

        #Transform our train text to vectors with tfidvectorizer, we convert all the words to vectors, 
        #so every word has its own numerical representation
        xtrain_tfv =  tfv_f.transform(train['remove_emoji'])
        xtrain_tfv = pd.DataFrame(xtrain_tfv.toarray())
        xtrain_tfv = pd.concat([xtrain_tfv, dummies], axis=1)
        #Transform our test (user input) text to vectors with tfidvectorizer, convert them to vectors based on the words in training,
        # so if the word doesn't exist in training, it will have a value of 0s in all the columns.
        x_test=tfv_f.transform(test_['remove_emoji'])
        x_test = pd.DataFrame(x_test.toarray())
        x_test = pd.concat([x_test, input_features2], axis=1)

            
        #Log transform the target for skewness
        ytrain=df['Winning Bid']
        ytrain = np.log1p(ytrain)


        
        print('Start predicting')
        from sklearn.linear_model import Ridge
        ridge_mod=Ridge(alpha=0.8)
        ridge_mod.fit(xtrain_tfv,ytrain)

        ridge_pred=ridge_mod.predict(x_test)
        #Then predict the ranges and cancel out the log transformation  
        rmse=0.72
        range1= ridge_pred - rmse
        range2= ridge_pred + rmse

        #Apply np.expm1 to cancel out the log transform with np.expm1, 
        #to get the original shape of the prediction price and the ranges
        # values become different but make more sense
        ridge_pred=np.expm1(ridge_pred)
        range1=np.expm1(range1)
        range2=np.expm1(range2)
        print(round(ridge_pred[0],3))

        #Display the prediction to the user
        st.markdown('## FINAL PREDICTION')
        st.write('Winning bid prediction for this item is around:', round(ridge_pred[0],3), currency_box)
       #st.write('In the range:', round(range1[0],3),currency_box, round(range2[0],3),currency_box)
        st.sidebar.write('Winning bid prediction for this item is around:', round(ridge_pred[0],3), currency_box)
        #st.sidebar.write('In the range:', round(range1[0],3),currency_box, round(range2[0],3),currency_box)
     #Display error if the entered username/password is not correct
     else:
        st.sidebar.warning('Wrong Username/Password')
