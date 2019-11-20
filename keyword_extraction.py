import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix

def preprocess_text(text):	
	stop_words = set(stopwords.words('english'))
	new_stop_words = ["also","use","bi","mi","al","th","give","go","et","fly","keep","video","let","see","watch","mr","like","alike","qlikview","actually","well","one"]
	stop_words = stop_words.union(new_stop_words)
	text = re.sub(r'(\,|\?)(?=\s)','',text)
	text = re.sub(r'\'','',text)
	text = re.sub(r'\$','',text)
	text = re.sub(r'[0-9]','',text)
	text = re.sub(r'\.',' .',text)
	text = text.lower()
	#lemmatisation
	l = WordNetLemmatizer()
	text = [l.lemmatize(word,'v') for word in text.split() if word not in stop_words]	
	text = ' '.join(text)
	text = text.split(".")
	return text

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items,topn=2000):
   
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    
    return results

def e_keyword(text,n):
	try:
		cv = CountVectorizer(min_df = 0.05,ngram_range=(n,n))
		X = cv.fit_transform(text)
		feature_names = cv.get_feature_names()
		tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
		tfidf_transformer.fit(X)
		tf_idf_vector = tfidf_transformer.transform(cv.transform(text))
		sorted_items = sort_coo(tf_idf_vector.tocoo())
		keywords = extract_topn_from_vector(feature_names,sorted_items)
	except:
		cv = CountVectorizer(min_df = 0.02,ngram_range=(n,n))
		X = cv.fit_transform(text)
		feature_names = cv.get_feature_names()
		tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
		tfidf_transformer.fit(X)
		tf_idf_vector = tfidf_transformer.transform(cv.transform(text))
		sorted_items = sort_coo(tf_idf_vector.tocoo())
		keywords = extract_topn_from_vector(feature_names,sorted_items)

	return keywords

def extract_keyword(file):
	print("Extracting Keywords . . . ")
	with open(file) as f:
		text = f.read()
	cleaned_text = preprocess_text(text)

	ngram = 1 #extract unigrams
	keywords = e_keyword(cleaned_text,ngram)

	uni_list = sorted(keywords.items(),key = lambda kv: (kv[1],kv[0]))
	#print(uni_list[:10])

	ngram = 2 #extract bigrams
	keywords = e_keyword(cleaned_text,ngram)

	bi_list = sorted(keywords.items(),key = lambda kv: (kv[1],kv[0]))
	#print(bi_list[:10])

	words = sorted((uni_list[:10] + bi_list[:10]), key = lambda kv: (kv[1]), reverse = True)

	return words

if __name__ == '__main__':

	
	file = '/home/stuti/Documents/PBL/audio_files/final_outputnews_text.txt'

	words = extract_keyword(file)
	print("TOP 5 KEYWORDS:")
	for w in words[:5]:
		print(w[0])

	
	





