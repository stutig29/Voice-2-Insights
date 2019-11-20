import spacy
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
words = set(nltk.corpus.words.words())

def check_eng_word(word):
	word = word.lower()
	if word in words:
		return True
	else:
		return False

def preprocess_text(text):	
	stop_words = set(stopwords.words('english'))
	new_stop_words = ["also","use","bi","mi","al","go","API","et","Movement","keep","Justice","Ayodhya","see","watch","III","qlikview","actually","Nahin","Hai","Chhodo Kal Ki Baatein Kal Ki Baat","Subscribe"]
	# add a list of HINGLISH words as stop words
	stop_words = stop_words.union(new_stop_words)

	text = re.sub(r'(\,|\?)(?=\s)','',text)
	text = re.sub(r'\'','',text)
	#text = re.sub(r'\$','',text)
	#text = re.sub(r'[0-9]','',text)
	text = re.sub(r'\.',' .',text)
	#text = text.lower()
	#lemmatisation
	l = WordNetLemmatizer()
	text = [l.lemmatize(word,'v') for word in text.split() if word not in stop_words]	
	text = ' '.join(text)
	text = text.split(".")
	return text

def extract_entities(file):
	with open(file) as f:
		text = f.read()
	cleaned_text = preprocess_text(text)

	nlp = spacy.load('en_core_web_sm') 

	person=[]
	org = []
	norp = []
	quantity = []
	product = []
	date = []
	gpe = []
	cardinal = []
	ordinal = []
	money = []
	for sentence in cleaned_text:
		doc = nlp(sentence)
		for ent in doc.ents:
			#if check_eng_word(ent.text) == True:
				#print(ent.text,ent.label_)
			if ent.label_ == "PERSON":
				if ent.text not in person:
					person.append(ent.text)
			if ent.label_ == "ORG":
				if ent.text not in org:
					org.append(ent.text)
			if ent.label_ == "QUANTITY":
				if ent.text not in quantity:
					quantity.append(ent.text)
			if ent.label_ == "NORP":
				if ent.text not in norp:
					norp.append(ent.text)
			if ent.label_ == "PRODUCT":
				if ent.text not in product:
					product.append(ent.text)
			if ent.label_ == "DATE":
				if ent.text not in date:
					date.append(ent.text)
			if ent.label_ == "GPE":
				if ent.text not in gpe:
					gpe.append(ent.text)
			if ent.label_ == "CARDINAL":
				if ent.text not in cardinal:
					cardinal.append(ent.text)
			if ent.label_ == "ORDINAL":
				if ent.text not in ordinal:
					ordinal.append(ent.text)
			if ent.label_ == "MONEY":
				if ent.text not in money:
					money.append(ent.text)

	return quantity,org,norp,product,person,cardinal,ordinal,money,date



if __name__ == '__main__':

	file = '/home/stuti/Documents/PBL/voice2_insights/files_uploaded/final_outputDS_text.txt'

	quantity,org,norp,product,person,cardinal,ordinal,money,date = extract_entities(file)

	print("PRODUCT:")
	print(product)
	print("ORG:")
	print(org)
	print("QUANTITY:")
	print(money)


	
  