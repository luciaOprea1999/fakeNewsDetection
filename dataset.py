from abc import ABC,  abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import utils as ut
from wordcloud import STOPWORDS, WordCloud
import csv
import numpy as np
import nltk

class Dataset(ABC):
       
    @abstractmethod
    def compressColumns(self):
        pass
    
    @abstractmethod
    def sentencePlot(self):
       pass
   
    def addingFeatures(self):
        self.bigData['NumberChars'] = self.bigData['text'].apply(lambda x:ut.count_chars(x))
        self.bigData['NumberWords'] = self.bigData['text'].apply(lambda x:ut.count_words(x))
        self.bigData['NumberUniqueWords'] = self.bigData['text'].apply(lambda x:ut.count_unique_words(x))
        self.bigData['NumberSentences'] = self.bigData['text'].apply(lambda x:ut.count_sent(x))
        self.bigData['avg_wordlength'] = self.bigData['NumberChars']/self.bigData['NumberWords']
        self.bigData['avg_sentlength'] = self.bigData['NumberWords']/self.bigData['NumberSentences']
    
    def WordCloud(self):
        np.random.seed(0)
        trueStatements_dataset = self.bigData[self.bigData['label'].astype(str) == 'true']
        fakeStatements_dataset = self.bigData[self.bigData['label'].astype(str) == 'fake']
        np.random.seed(0)
        text_true = trueStatements_dataset['text'].values
        wordcloud_true = WordCloud(width=3000, height=2000, background_color='white',stopwords=STOPWORDS).generate(str(text_true))
        plt.imshow(wordcloud_true)
        plt.axis('off')
        plt.title("True Statements")
        plt.show()
        text_fake = fakeStatements_dataset['text'].values
        wordcloud_fake = WordCloud(width=3000, height=2000, background_color='white',stopwords=STOPWORDS).generate(str(text_fake))
        plt.imshow(wordcloud_fake)
        plt.axis('off')
        plt.title("Fake Statements")
        plt.show()
         

class LiarDataset(Dataset):
    
    def __init__(self):
        data1 = pd.read_csv("./Datasets/liar_dataset/train.tsv",sep='\t')
        data2 = pd.read_csv("./Datasets/liar_dataset/test.tsv",sep='\t')
        data1.columns=["ID", "label", "text", "subject", "speaker", "job", "state", "party", "barely_true_cts","false_cts", "half_true_cts", "mostly_true_cts", "pants_on_fire_cts", "context"]
        data2.columns=["ID", "label", "text", "subject", "speaker", "job", "state", "party", "barely_true_cts","false_cts", "half_true_cts", "mostly_true_cts", "pants_on_fire_cts", "context"]
        self.bigData = pd.concat([data1,data2],ignore_index = 1)
        
    def compressColumns(self):
        conditions = [
        (self.bigData['label'] == 'barely-true') | (self.bigData['label'] == 'pants-fire') | (self.bigData['label'] == 'false') ,
        (self.bigData['label'] == 'true') | (self.bigData['label'] == 'half-true') | (self.bigData['label'] == 'mostly-true')
        ]
        values = ["fake","true"]
        self.bigData['Label'] = np.select(conditions,values)
        columns=["ID", "label", "subject", "speaker", "job", "state", "party", "barely_true_cts",
       "false_cts", "half_true_cts", "mostly_true_cts", "pants_on_fire_cts", "context"]
        self.bigData.drop(columns, inplace=True, axis=1)
        
    def sentencePlot(self):
        self.bigData['char_count'] = self.bigData['text'].apply(lambda x:ut.count_chars(x))
        
    def getData(self):
        return self.bigData

class KaggleDataset(Dataset):
    
    def __init__(self):
        super().__init__()
        data1 = pd.read_csv("./Datasets/kaggle_dataset/Fake.csv")
        data2 = pd.read_csv("./Datasets/kaggle_dataset/True.csv")
        data1['Label'] = 'Fake'
        data2['Label'] = 'True'
        self.bigData = pd.concat([data1,data2],ignore_index = 1)
        
    def compressColumns(self):
        columns=["title", "subject", "date"]
        self.bigData.drop(columns, inplace=True, axis=1)
        
    def sentencePlot(self):
        self.bigData['char_count'] = self.bigData['text'].apply(lambda x:ut.count_chars(x))
        
    def getData(self):
        return self.bigData
    
class HornDataset(Dataset):
    
    def __init__(self):
        super().__init__()
        df = pd.read_csv("FinalFake.csv")
        df1 = pd.read_csv("FinalTrue.csv")
        df1.drop(df.columns[[0]], axis = 1, inplace = True)
        df.drop(df.columns[[0]], axis = 1, inplace = True)
        frames = [df, df1]
        self.bigData = pd.concat(frames)
        self.bigData.dropna(subset = ["Label"], inplace=True)
        print(self.bigData['text'].dtype)
        self.bigData['text'] = self.bigData['text'].astype(str)
        self.bigData['Label'] = self.bigData['Label'].astype(str)
        print(self.bigData['Label'].dtype)

        
    def compressColumns(self):
        pass
        
    def sentencePlot(self):
        self.bigData['char_count'] = self.bigData['text'].apply(lambda x:ut.count_chars(x))
        
    def getData(self):
        return self.bigData
        