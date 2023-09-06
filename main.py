import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from torch import positive
import utils as ut
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LarsCV, LogisticRegression
from dataset import Dataset, HornDataset, LiarDataset, KaggleDataset
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt')
from sentence_transformers import SentenceTransformer, util
  
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


import warnings
warnings.filterwarnings('ignore')

def dot(A,B): 
    return (sum(a*b for a,b in zip(A,B)))

#def cosine(u, v):
 #   return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

#def cosine_similarity(a,b):
 #   return dot(a,b) / ( (dot(a,a) **.5) * (dot(b,b) ** .5) )
def create_dataframe(matrix, tokens):

    doc_names = [f'doc_{i+1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return(df)


def classificationReport(prediction,test):
   
   cf_matrix = confusion_matrix(test, prediction)
   group_names = ['True Neg','False Pos','False Neg','True Pos']

   group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

   group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]

   labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

   labels = np.asarray(labels).reshape(2,2)

   ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

   ax.set_title('Seaborn Confusion Matrix with labels\n\n');
   ax.set_xlabel('\nPredicted Values')
   ax.set_ylabel('Actual Values ');
   ax.xaxis.set_ticklabels(['False','True'])
   ax.yaxis.set_ticklabels(['False','True'])
   plt.show()
   
   


def RandomForestMultipleFeatures(dataset):
    X = dataset.getData()["text"]
    Y = dataset.getData()["Label"]

    vectorizer            =  TfidfVectorizer()  
    train_tf_idf_features =  vectorizer.fit_transform(dataset.getData()['text']).toarray()
    train_tf_idf          = pd.DataFrame(train_tf_idf_features)

    features = ['NumberChars','NumberWords','NumberUniqueWords','NumberSentences','avg_wordlength','avg_sentlength']
    train = pd.merge(train_tf_idf,dataset.getData()[features],left_index=True, right_index=True)
    X_train, X_test, y_train, y_test = train_test_split(train, Y, test_size=0.2, random_state = 42)# Random Forest Classifier
    RandomForest_Classifier = RandomForestClassifier(n_estimators = 100, min_samples_split = 15, random_state = 42)
    RandomForest_Classifier.fit(X_train, y_train)
    RandomForestClassifier_prediction = RandomForest_Classifier.predict(X_test)
    print("Accuracy => ", round(accuracy_score(RandomForestClassifier_prediction, y_test)*100, 2))
    #print(classification_report(y_test, RandomForestClassifier_prediction, labels=[1,0]))
    classificationReport(RandomForestClassifier_prediction, y_test)
    
    
    
def LogisticRegressionMultipleFeatures(dataset):
    X = dataset.getData()["text"]
    Y = dataset.getData()["Label"]

    vectorizer            =  TfidfVectorizer()  
    train_tf_idf_features =  vectorizer.fit_transform(dataset.getData()['text']).toarray()
    train_tf_idf          = pd.DataFrame(train_tf_idf_features)

    features = ['NumberChars','NumberWords','NumberUniqueWords','NumberSentences','avg_wordlength','avg_sentlength']
    train = pd.merge(train_tf_idf,dataset.getData()[features],left_index=True, right_index=True)
    X_train, X_test, y_train, y_test = train_test_split(train, Y, test_size=0.3, random_state = 42)
    Logistic_Regression = LogisticRegression()
    Logistic_Regression.fit(X_train, y_train)
    LogisticRegression_prediction = Logistic_Regression.predict(X_test)
    print("Accuracy => ", round(accuracy_score(LogisticRegression_prediction, y_test)*100, 2))
    classificationReport(LogisticRegression_prediction, y_test)
    
    
    
def RandomForestPipeline(dataset):
    X = dataset.getData()["text"]
    Y = dataset.getData()["Label"]
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
    
    pipe2 = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
    model = pipe2.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
    #print(classification_report(y_test, prediction, labels=[1,0]))
    classificationReport(prediction, y_test)
    
    
    
def SVMPipeline(dataset):
    X = dataset.getData()["text"]
    Y = dataset.getData()["Label"]
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.1, random_state=42)
    param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf']}
    
    pipe2 = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                ('model', GridSearchCV(SVC(), param_grid, refit = True, verbose = 3))])
    model = pipe2.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
    #print(classification_report(y_test, prediction, labels=[1,0]))
    classificationReport(prediction, y_test)

    
    
    
def LogisticRegressionPipeline(dataset):
    X = dataset.getData()["text"]
    Y = dataset.getData()["Label"]
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
    pipe1 = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='word',preprocessor=ut.preprocess,tokenizer=ut.Tokenizer,stop_words=ut.stopwords_list)),
        ('norm2', TfidfTransformer(norm=None)),
        ('selector', SelectKBest(chi2, k=5)),
        ('clf', LogisticRegression(solver='liblinear', random_state=0)),
      ])
# Fitting the model
    model = pipe1.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
    #print(classification_report(y_test, prediction, labels=[1,0]))
    classificationReport(prediction, y_test)

def compute_similarity(input,strings):
    maxi = -1
    sentence =""
    label = "fake"
    for i in range(len(strings)):
        print(i)
        corpus = []
        corpus.append(input)
        corpus.append(strings[i][1])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus,y=None)
        vectorizer.get_feature_names_out()
        tokens = vectorizer.get_feature_names()
        create_dataframe(X.toarray(),tokens)
        cosine_similarity_matrix = cosine_similarity(X)
        create_dataframe(cosine_similarity_matrix,['doc_1','doc_2'])
        if maxi < cosine_similarity_matrix[0][1]:
            maxi = cosine_similarity_matrix[0][1]
            sentence = strings[i][1]
            label = strings[i][0]

    return label
    


def jaccard_similarity(input, strings):
    maxi = -1
    sentence =""
    label = "fake"
    for i in range(len(strings)):
       sentenceInput =set(input.lower().split()) 
       sentenceCompare = set(strings[i][1].lower().split()) 
       intersection = sentenceInput.intersection(sentenceCompare)
       union = sentenceInput.union(sentenceCompare)
       aux = float(len(intersection)) / len(union)
       if maxi < aux:
            maxi = aux
            sentence = strings[i][1]
            label = strings[i][0]
   
    return label
    
    
def bert_similarity(input, strings):
    maxi = -1
    sentence =""
    label = "fake"
    print("da")
    embedding_1= model.encode(input, convert_to_tensor=True)
    #query_vec = sbert_model.encode([input])[0]
    for i in range(len(strings)):
       #value = cosine(query_vec, sbert_model.encode([strings[i][1]])[0])
       print(i)      
       embedding_2 = model.encode(strings[i][1], convert_to_tensor=True)
       value = util.pytorch_cos_sim(embedding_1, embedding_2)
       if maxi < value:
            maxi = value
            sentence = strings[i][1]
            label = strings[i][0]

    #print(sentence)
    #print(maxi)
    #print(label)   
    return label

def doc2vec_similarity(input,strings):
    sentences = []
    for i in range(len(strings)):
        sentences.append(strings[i][1])
    tokenized_sent = []
    for s in sentences:
      tokenized_sent.append(word_tokenize(s.lower()))
    
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
    model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
    test_doc = word_tokenize(input.lower())
    test_doc_vector = model.infer_vector(test_doc)
    model.docvecs.most_similar(positive = [test_doc_vector])
    
    
def test_jaccard(datasetList,datasetTest):
    correct = 0
    for i in range(len(datasetTest)):
        label = jaccard_similarity(datasetTest[i][1],datasetList)
        if label == datasetTest[i][0]:
            correct = correct + 1
    print(correct)
    
def test_cosine(datasetList,datasetTest):
    correct = 0
    for i in range(len(datasetTest)):
        print(i)
        label = compute_similarity(datasetTest[i][1],datasetList)
        if label == datasetTest[i][0]:
            correct = correct + 1
    print(correct)

def test_bert(datasetList,datasetTest):
    correct = 0
    for i in range(len(datasetTest)):
        label = bert_similarity(datasetTest[i][1],datasetList)
        if label == datasetTest[i][0]:
            correct = correct + 1
    print(correct)
    
        
if __name__ == "__main__":

 if sys.argv[1] == 'LiarDataset':
   print("da")
   dataset = LiarDataset()
   
 elif sys.argv[1] == 'HornsDataset':
   dataset = HornDataset()
   
 else:
   dataset = KaggleDataset()
   
dataset.addingFeatures()
dataset.compressColumns()
data = dataset.getData()
dataset.WordCloud()
partDataset = data.sample(frac = 0.8)
partTest = data.drop(partDataset.index)


datasetList =  partDataset.apply(lambda row: (row['Label'],row['text']), axis=1).to_list()
datasetTest = partTest.apply(lambda row: (row['Label'],row['text']), axis=1).to_list()

mylist = data.apply(lambda row: (row['Label'],row['text']), axis=1).to_list()
SVMPipeline(dataset)
start = time.time()
RandomForestPipeline(dataset)
end = time.time()
print(end - start)
bert_similarity('Senate Majority Leader Harry Reid can not stand John McCain.',mylist)
end = time.time()


# Senate Majority Leader Harry Reid "said, quote, 'I can't stand John McCain.'
start = time.time()
compute_similarity('Senate Majority Leader Harry Reid can not stand John McCain.',mylist)
bert_similarity('Senate Majority Leader Harry Reid can not stand John McCain.',mylist)
end = time.time()
print(end - start)


start = time.time()
test_bert(datasetList,datasetTest)
jaccard_similarity('Senate Majority Leader Harry Reid can not stand John McCain.',mylist)
end = time.time()
print(end - start)

from gensim.models import Word2Vec
from gensim.test.utils import common_texts

model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.train([["hello", "world"]], total_examples=1, epochs=1)
dataset.addingFeatures()
RandomForestPipeline(dataset)
RandomForestMultipleFeatures(dataset)
LogisticRegressionPipeline(dataset)
LogisticRegressionMultipleFeatures(dataset)
print(dataset.getData().head())
SVMPipeline(dataset)



data = data.rename(columns={'Label': 'label'})
data['label'] = data['label'].apply(lambda x: 0 if x == 'fake' else 1)

part_1 = data.sample(frac = 0.8)
part_2 = data.drop(part_1.index)

part_1.to_csv('train.csv', index=False)
part_2.to_csv('test.csv', index=False)

