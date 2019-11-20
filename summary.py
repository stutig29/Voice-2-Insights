import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
def preprocess_text(text):  
  stop_words = set(stopwords.words('english'))
  new_stop_words = ["also","use","bi","mi","al","go","et","fly","keep","video","let","see","watch","mr","like","alike","qlikview","actually","well","one"]
  stop_words = stop_words.union(new_stop_words)
  #text = re.sub(r'(\,|\?)(?=\s)','',text)
  text = re.sub(r'\'','',text)
  #text = re.sub(r'\$','',text)
  #text = re.sub(r'[0-9]','',text)
  #text = re.sub(r'\.',' .',text)
  #text = text.lower()
  #lemmatisation
  #l = WordNetLemmatizer()
  #text = [l.lemmatize(word,'v') for word in text.split() if word not in stop_words] 
  #text = ' '.join(text)
  #text = text.split(".")
  return text

def summarize(file):
  print("extracting summary . . . . ")
  with open(file, 'r') as myfile:
    data = myfile.readlines()

  from nltk.tokenize import sent_tokenize
  sentences = []
  for s in data:
    sentences.append(sent_tokenize(s))
  
  sentences = [y for x in sentences for y in x] 

  clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
  clean_sentences = [s.lower() for s in clean_sentences]

  clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

  word_embeddings = {}
  f = open('/home/stuti/Documents/PBL/glove.6B.100d.txt', encoding='utf-8')
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
  f.close()
  sentence_vectors = []
  for i in clean_sentences:
    if len(i) != 0:
      v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
      v = np.zeros((100,))
    sentence_vectors.append(v)

  sim_mat = np.zeros([len(sentences), len(sentences)])

  from sklearn.metrics.pairwise import cosine_similarity
  for i in range(len(sentences)):
    for j in range(len(sentences)):
      if i != j:
        sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
  import networkx as nx
  nx_graph = nx.from_numpy_array(sim_mat)
  scores = nx.pagerank(nx_graph)
  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
  return ranked_sentences


def remove_stopwords(sen):
  stop_words = stopwords.words('english')
  sen_new = " ".join([i for i in sen if i not in stop_words])
  return sen_new

if __name__ == '__main__':

  file = '/home/stuti/Documents/PBL/audio_files/DS/final_outputDS_text.txt'

    
  ranked_sentences = summarize(file)
  for i in range(2):
    print(ranked_sentences[i][1])