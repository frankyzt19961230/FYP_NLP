from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import tree
import pymongo
import numpy as np
import database_test as database
import pandas,string,os,csv
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

def Originfile_insert(outfile,collection):
    collection.delete_many({})
    database.csv_insert(outfile,collection)

def syllabus():
    csvDir = "/Users/frank/PycharmProjects/FYP_classification/Dataset/articles/csv/"
    fileoutput ='/Users/frank/PycharmProjects/FYP_classification/Dataset/articles/EIE_outfile.csv'
    os.remove(fileoutput)
    for fileinput in os.listdir(csvDir):
        with open(csvDir+fileinput) as f:
            reader = csv.reader(f)
            #print(list(reader))
            title = fileinput.strip('/Users/frank/PycharmProjects/FYP_classification/Dataset/articles/csv/.')
            content= ''
            for row in reader:
                #print(row)
                rowcontent = ''.join(row).replace('\n','')
                content = content+rowcontent
            print(content)
        with open(fileoutput,mode='a') as f:
            f.write(title+' '+content+'\n')
            #content = ''.join(list(reader))

def feature_keywords(outfile,collection,vectorizer):
    collection.delete_many({})
    data = open(outfile).read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        if len(line.split()) > 0:
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))
    wordslist = texts
    titlelist = labels

    transformer = TfidfTransformer()
    tfidf = vectorizer.fit_transform(wordslist)
    #print(tfidf)
    #print(vectorizer.fit_transform(wordslist))
    words = vectorizer.get_feature_names()  #所有文本的关键字
    weight = tfidf.toarray()

    n = 5 # 前五位
    for (title, w, text) in zip(titlelist, weight, texts):
        sw = stopwords.words('english')
        wordsdet= ['sessions mini','final','12','supervisor','18','development','management','model','based','grading','grading citeria','expectation grading','2013','li','case study','semester','short quizzes','book','introduce','mini','mini project','help','self','assignment homework','students','project','subject','assessment','hours']
        list = sw + wordsdet
        print (u'{}:'.format(title))
        # 排序
        loc = np.argsort(-w)
        keywordsList = []
        #Keywords = ''
        i,j=0,0
        while j < n:
            if words[loc[i]] in list:
                i += 1
                continue
            keywordsList.append(words[loc[i]]+',')

            print (u'-{}: {} {}'.format(str(j + 1), words[loc[i]], w[loc[i]]))
            i +=1
            j +=1
        Keywords_long = ''.join(keywordsList)
        Keywords = Keywords_long[:-1]
        url = "http://www.eie.polyu.edu.hk/prog/syllabus/"+title+".pdf"
        if title == 'EIE4432':
            maincontent = 'website design'
        elif title == 'EIE3312':
            maincontent = 'linear system'
        else:
            maincontent = 'TBD'

        post = {
            'Label': title,
            'MainContent':maincontent,
            'KeyWords': Keywords,
            'URL':url,
            'Content': text

        }
        collection.insert_one(post)
        print ('\n')

def main():
    nltk.download('stopwords')
    # Set up database
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient['SUBJECT_INFO']
    collection_set = {
        'Acct' : db['Student_Account'],
        'Sbj_Info': db['Subject_Info'],
        'Keywords': db['Subject_Keywords'],
    }
    vectorizer_set = {
        'Tdidf_diagram': TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english'),
        'Count': CountVectorizer(),
        'Tdidf': TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', stop_words='english')

    }
    # Preprocess

    outfile = "/Users/frank/PycharmProjects/FYP_classification/Dataset/articles/EIE_outfile.csv"
    #TSNE(outfile)

    # Origin CSV insert to database
    Originfile_insert(outfile, collection_set['Keywords'])
    # keywords
    feature_keywords(outfile,collection_set['Keywords'],vectorizer_set['Tdidf_diagram'])
    #feature_keywords(outfile,collection_set['BBC_countV'],vectorizer_set['Count'])
    #feature_keywords(outfile,collection_set['BBC_tdidf'],vectorizer_set['Tdidf'])

    # Classify
    classifier_set = {
        'NB': MultinomialNB(),
        'SVM': SVC(kernel='linear'),
        'DT': tree.DecisionTreeClassifier()

    }
    # Database Input




if __name__=='__main__':
    main()


