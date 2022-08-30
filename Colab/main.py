import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import nltk

import re
nltk.download('stopwords')
from nltk.corpus import stopwords

nlp=spacy.load("en_core_web_sm")
import pickle
import tensorflow
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import geopandas as gpd
import requests 
import json
import tweepy

consumer_key='vtpCktz8FTXUKEqbrmQyfitZ2'
consumer_secret='FDbBJ3gTQPvv2lA9jT23DDbnDJipV4NjmfEZMobi8Xl7zJO2L1'
access_token='1287085799339462656-yMJ0c9NzCgbwu7sd9QEVjyXUKkWieS'
access_token_secret='OPq0IiDTyGmhziS0EUNyYbupdr492tuB848rtGT8BxzLS'
#load model
file_to_read = open("Colab/stored_object.pickle", "rb")
z=pickle.load(file_to_read)
t=Tokenizer(num_words=50000,lower=True)
t.fit_on_texts(z)
x=t.texts_to_sequences(z)
x=sequence.pad_sequences(x,maxlen=250)
model=keras.models.load_model("Colab/my_model.h5")

st.set_page_config(layout="wide",menu_items=None)

def tweet_scrape(num_tweets,option): 
  try:
    option=option.split(" ")
    option=option[:2]
    option=" ".join(option)
  except:
    option=option
  text=st.markdown("<html><body><h4>Scrapping tweets ...</h4></body></html>", unsafe_allow_html=True)
          # scraped tweets
  bearer_token='AAAAAAAAAAAAAAAAAAAAAJbiNgEAAAAA9XfDS3YRCsEWv9GoitFHFTCuu9I%3DFdYzAbd2GvMgStnVTDVlnD1TVMARk4w306nGUt9YHNoT2AkUPQ'
  client=tweepy.Client(bearer_token=bearer_token)
  tweetDf={'data':[],'users':[]}

  def getUser(ids):
    users=client.get_users(ids=ids,user_fields=['username','name','location'])
    tweetDf['users']+=users.data

  def search_twitter(query, tweet_fields ,bearer_token):
    ids=[]
    tweets=client.search_recent_tweets(query=query,tweet_fields=tweet_fields,expansions=['author_id'],max_results=100) 
    tweetDf['data']=tweets.data
    since_id=tweetDf['data'][-1].id
    y=z
    
    for i in range(num_tweets-1):
      tweets=client.search_recent_tweets(query=query,tweet_fields=tweet_fields,expansions=['author_id'],since_id=since_id,max_results=100)      
      tweetDf['data']+=tweets.data
      since_id=tweetDf['data'][-1].id

    for i in tweetDf['data']:
      ids.append(i['author_id'])
    for i in range(num_tweets):
      try:
        getUser(ids[100*i:100*i+100])
      except:
        getUser(ids[500:len(ids)])
    text.empty()
    return tweetDf

  query="i want "+option+" -is:retweet OR i hate "+option+" -is:retweet OR i love "+option+" -is:retweet OR i wish for "+option+" -is:retweet"
  tweet_fields=['text','author_id','created_at','lang']
  json_response=search_twitter(query=query, tweet_fields=tweet_fields,bearer_token=bearer_token)
  return json_response

def tweet_df(json_response):
  data={'username':[],'name':[],'tweet':[],'location':[]}
  for i in range(len(json_response['users'])):
    try:
      doc=nlp(json_response['users'][i]['location'])
      k=0
      for token in doc:
        if token.ent_type_=="GPE":
          k=1
          location=str(doc).split(',')
          data['username'].append(json_response['users'][i]['username'])
          data['name'].append(json_response['users'][i]['name'])
          data['tweet'].append(json_response['data'][i]['text'])
          data['location'].append(" ".join(location))
          break
      if k==0:
        data['username'].append(json_response['users'][i]['username'])
        data['name'].append(json_response['users'][i]['name'])
        data['tweet'].append(json_response['data'][i]['text'])
        data['location'].append("")      
    except:
      data['username'].append(json_response['users'][i]['username'])
      data['name'].append(json_response['users'][i]['name'])
      data['tweet'].append(json_response['data'][i]['text'])
      data['location'].append("")

  pr=pd.DataFrame(data)
  count=0
  for i in pr['location']:
    if i!='':
      count+=1
  return pr

def final_tweet_df(tweetDf):
  coor={'geometry':[],'address':[]}
  head=coly3.markdown("<html><body><h3>Loading Locations ...</h3></body></html>", unsafe_allow_html=True)
  my_bar = coly3.progress(0)
  z=100/len(tweetDf)
  y=0
  k=0
  for i in tweetDf.location:
    k+=1
    if i=="":
      coor['geometry'].append(None)
      coor['address'].append(None)
    else:
      a=gpd.tools.geocode(i, provider='nominatim',timeout=100, user_agent="http")
      coor['geometry'].append(a['geometry'][0])
      coor['address'].append(a['address'][0])
    y=min(int((k)*z),100)
    my_bar.progress(y)


  a=pd.DataFrame(coor)
  final_data=pd.concat([tweetDf,a],axis=1)
  my_bar.empty()
  head.empty()
  return final_data

def preprocess(text):
  Stopwords=set(stopwords.words('english'))
  emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  
                                u"\U0001F300-\U0001F5FF"  
                                u"\U0001F680-\U0001F6FF"  
                                u"\U0001F1E0-\U0001F1FF"  
                                u"\U00002500-\U00002BEF"  
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  
                                u"\u3030"
                                "]+", flags=re.UNICODE)

  text=" ".join(word for word in str(text).split() if word not in Stopwords) # remove stopwords
  text=text.lower()                                                     #converting to lower
  text=re.sub(r'https?:\/\/\S+', '', text)                              #removing hyperlinks
  text=re.sub(r'@[a-zA-Z0-9_@./#&+-]+', '', text)                       #removing tags
  text=re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', text)          #removing web links
  text= re.sub(r'&[a-z]+;', '', text)                                   #removing html reference characters
  text=re.sub(r'\n', ' ', text)
  text=re.sub(r"(.)\1\1+",r"\1", text)                                  #removing repeated letters
  text=re.sub(r'\b(\w+)(\s+\1)+\b',r'\1', text)                         #removing repeated words
  text=emoji_pattern.sub('',text)                                       #remove emoticons
  doc,text=nlp(text),''                                                 # remove currency and punctuations
  for token in doc:
    if token.is_currency!=True and token.ent_type_!="MONEY" and token.is_punct!=True:
      text+=" "+str(token)
  return text
  
def predict(finalDf,tweetDf):
  text=st.markdown("<html><body><h4>predicting results ...</h4></body></html>", unsafe_allow_html=True)
  bar=st.progress(0)
  z=100/len(finalDf)
  y=0
  aa=finalDf
  aa=pd.Series(aa.tweet.values).to_dict()
  a=[]
  k={'username':[],'tweet':[]}
  ps=dict()
  for i in range(len(aa)):
    x=t.texts_to_sequences([aa[i]])
    x=sequence.pad_sequences(x,maxlen=250)
    p=model.predict(x)
    pred=np.argmax(p,axis=1)
    p=np.reshape(pred,(len(pred),1))
    if p==1:
      a.append('yes')
      ps[i]=aa[i]
      k['username'].append(tweetDf['username'][i])
      k['tweet'].append(tweetDf['tweet'][i])

    else:
      a.append('no')
    y=min(int((i)*z),100)
    bar.progress(y)
     
  x=a.count('yes')
  y=a.count('no')
  predictDf=pd.DataFrame(k)
  text.empty()
  bar.empty()
  return [x,y],predictDf


def get() :
  x=t.texts_to_sequences([title]) # pro
  x=sequence.pad_sequences(x,maxlen=250)
  p=model.predict(x)
  print(p)
  pred=np.argmax(p,axis=1)
  print(pred)
  if pred[0]==1:
    st.write('yes')
  else:
    st.write('no')

def send_msg(k,msg):
  count=0
  for i in k['username']:
    count+=1
  # authorization of consumer key and consumer secret
    try:
      auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        
      # set access to user's access key and access secret 
      auth.set_access_token(access_token, access_token_secret)
        
      # calling the api 
      api = tweepy.API(auth)
        
      # ID of the recipient
        
      # text to be sent
      text = msg
      user=api.get_user(screen_name=i)
      ID= user.id_str
      # sending the direct message
      direct_message = api.send_direct_message(ID, text)
    except:
      continue
  return count

st.title("Purchase Intention Analyzer")


col1,col2 = st.columns(2)
option = col1.selectbox(
     'Select Brand Name',
     ('iphone 11', 'Samsung galaxy s20'))
col1.title("Product")
num_tweets= col2.select_slider('Select number of tweets: ',options=['100', '200', '300', '400', '500', '600', '700'])
json_response=tweet_scrape(int(num_tweets)//100,option)
tweetDf=tweet_df(json_response)

finalDf=tweetDf
text2=st.markdown("<html><body><h4>Preprocessing tweets ...</h4></body></html>", unsafe_allow_html=True)  
finalDf['tweet']=finalDf['tweet'].apply(preprocess)
text2.empty()
result,k=predict(finalDf,tweetDf)



colx1,colx2,colx3,colx4=st.columns(4)
if option=='iphone 11': 
  colx1.image("https://i.ibb.co/wBnwsS1/71w3o-J7a-Wy-L-SX679-Photo-Room-Photo-Room-1.png",width=300)
else :
  colx1.image("https://i.ibb.co/7pH96cC/71-IX75jo-Wj-L-AC-SL1500-removebg.png",width=300)

col2.title("DataFrame", 297)
colx3.write(tweetDf[['username','tweet','location']])
colx4.write(k)

title = st.text_input('Send message via Twitter', option+' available at discounted price on Amazon and Flipkart, Grab the offer now!')
if st.button('Send message to '+str(len(k))+" interested users"):
     count=send_msg(k,title)
     st.success("Message successfully sent to "+str(count)+" interested users")

coly1,coly3=st.columns(2)
chart_data = pd.DataFrame(
     np.random.randn(50, 2),
     columns=["a", "b"])


labels = 'Yes', 'No'
sizes =[result[0],result[1]]
explode = (0.1, 0)
coly1.write('Pie Chart')
fig1, ax1 = plt.subplots()
fig1.set_facecolor("#0D1117")
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,textprops={'color':"white"})
ax1.axis('equal')



coly1.pyplot(fig1)
coly3.write('Map data')
finalDf=final_tweet_df(tweetDf)
d={'latitude':[],'longitude':[]}
for i in finalDf['geometry']:
  if str(i)[0]=='P':
    s=str(i).split('(')[1][:-1]
    p=s.split(" ")
    d['latitude'].append(float(p[1]))
    d['longitude'].append(float(p[0]))
data_of_map = pd.DataFrame(d)
coly3.map(data_of_map)      #-118.24544999999995 34.053570000000036
