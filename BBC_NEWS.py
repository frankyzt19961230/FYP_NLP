from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import tree
import pymongo
import numpy as np
import database_test as database
import pandas,string,os
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from nlp_rake import rake

def Originfile_insert(outfile,collection):
    collection.delete_many({})
    database.csv_insert(outfile,collection)


def KeyWords_combine(db):
    pipeline = [{
        '$lookup': {
            'from': 'BBC_News_FeatureKeyWords_CountVectorizer',
            'localField': 'Content',
            'foreignField' : 'Content',
            'as': 'Keywords_CountV'
        }},{
        '$lookup': {
            'from': 'BBC_News_FeatureKeyWords_tdidf',
            'localField': 'Content',
            'foreignField' : 'Content',
            'as': 'Keywords_tdidf'
        }},{
        '$lookup': {
            'from': 'BBC_News_FeatureKeyWords_tdidf_Biagram',
            'localField': 'Content',
            'foreignField' : 'Content',
            'as': 'Keywords_tdidf_Biagram'
        }},{
        '$project': {
            'Label':{'$arrayElemAt':['$Keywords_CountV.Label',0]},
            'Keywords_tdidf': {'$arrayElemAt':['$Keywords_tdidf.KeyWords', 0]},
            'Keywords_tdidf_Biagram': {'$arrayElemAt':['$Keywords_tdidf_Biagram.KeyWords', 0]},
            'Keywords_CountV': {'$arrayElemAt':['$Keywords_CountV.KeyWords', 0]},
            'Content':{'$arrayElemAt':['$Keywords_CountV.Content',0]}}}

    ]
    update_result = db.aggregate(pipeline)
    """
    for x in update_result:
        print(x)
        db.update_one({
            '_id':x['_id']
        }, {
            'Label':x['Label'],
            'Keywords_tdidf':x['Keywords_tdidf'],
            'Keywords_tdidf_Biagram':x['Keywords_tdidf_Biagram'],
            'Keywords_CountV':x['Keywords_CountV'],
            'Content':x['Content']
        })
    """


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
        wordsdet= ['and','of','the','to','in','will','students','project','subject','assessment','hours','with','on','be','for','you','he','she','her','his']
        print (u'{}:'.format(title))
        # 排序
        loc = np.argsort(-w)
        keywordsList = []
        #Keywords = ''
        i,j=0,0
        while j < n:
            if words[loc[i]] in wordsdet:
                i += 1
                continue
            keywordsList.append(words[loc[i]]+',')

            print (u'-{}: {} {}'.format(str(j + 1), words[loc[i]], w[loc[i]]))
            i +=1
            j +=1
        Keywords = ''.join(keywordsList)
        post = {
            'Label': title,
            'KeyWords': Keywords,
            'Content': text
        }
        collection.insert_one(post)
        print ('\n')
def TSNE(outfile):
    data = open(outfile).read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        if len(line.split()) > 0:
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))
    #创建一个dataframe，列名为text和label
    trainDF = pandas.DataFrame()
    trainDF['seriesNum'] = range(0, 2225)
    trainDF['label'] = labels
    trainDF['text'] = texts
    trainDF['category_id'] = trainDF['label'].factorize()[0]
    labels = trainDF['category_id']
    category_id_df = trainDF[['label', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'label']].values)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(trainDF['text']).toarray()

    SAMPLE_SIZE = int(len(features) * 0.3)
    np.random.seed(0)
    indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
    projected_features = manifold.TSNE(n_components=2, random_state=0).fit_transform(features[indices])
    colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']
    for category, category_id in sorted(category_to_id.items()):
        points = projected_features[(labels[indices] == category_id).values]
        plt.scatter(points[:, 0], points[:, 1], s=15, c=colors[category_id], label=category)
    plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",
              fontdict=dict(fontsize=15))
    plt.legend()
    plt.show()


def multiple_classify(outfile, collection,classifier):
    collection.delete_many({}) #重新输入
    data = open(outfile).read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        if len(line.split()) > 0:
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))
    #创建一个dataframe，列名为text和label
    trainDF = pandas.DataFrame()
    trainDF['seriesNum'] = range(0, 2225)
    trainDF['label'] = labels
    trainDF['text'] = texts

    # print(trainDF)
    pipeline = Pipeline([
        ('tdidf_vectorizer',   TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)),
        ('classifier',         classifier)
    ])

    k_fold = model_selection.KFold(n_splits=5, shuffle=True)
    scores = []
    confusion = np.zeros((5,5))
    for train_indices, test_indices in k_fold.split(trainDF):

        sub_seriesNum = trainDF['seriesNum'][test_indices].tolist()
        train_text = trainDF['text'][train_indices]
        train_y = trainDF['label'][train_indices]
        test_text = trainDF['text'][test_indices]
        test_text_list = test_text.tolist()
        test_y = trainDF['label'][test_indices]
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        test_y = encoder.fit_transform(test_y)

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)
        for i in range(len(predictions)):
            post = {
                'Num': int(sub_seriesNum[i]),
                'Predict_Label': int(predictions[i]),
                'Actual_Label': int(test_y[i]),
                'Content': test_text_list[i]
            }
            collection.insert_one(post)
        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, average='macro')
        scores.append(score)

    print('Total news classified:', len(trainDF))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)
    print('\n')

def main():
    # Set up database
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient['BBC_NEWS']
    collection_set = {
        #'Act' : db['Student_Account'],
        #'Sbj_Info': db['Subject_Info'],
        'BBC': db['BBC_News'],
        'BBC_Result': db['BBC_News_ClassificationResult'],
        'BBC_Biagram':db['BBC_News_FeatureKeyWords_tdidf_Biagram'],
        'BBC_countV':db['BBC_News_FeatureKeyWords_CountVectorizer'],
        'BBC_tdidf':db['BBC_News_FeatureKeyWords_tdidf']
    }
    vectorizer_set = {
        'Tdidf_diagram': TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english'),
        'Count': CountVectorizer(),
        'Tdidf': TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', stop_words='english')

    }
    # Preprocess
    targetfile = "Dataset/bbc-text.csv"
    outfile = database.preprocess(targetfile)
    #TSNE(outfile)

    # Origin CSV insert to database
    Originfile_insert(outfile, collection_set['BBC'])
    # keywords
    #feature_keywords(outfile,collection_set['BBC_Biagram'],vectorizer_set['Tdidf_diagram'])
    #feature_keywords(outfile,collection_set['BBC_countV'],vectorizer_set['Count'])
    #feature_keywords(outfile,collection_set['BBC_tdidf'],vectorizer_set['Tdidf'])
    # Classify
    classifier_set = {
        'NB': MultinomialNB(),
        'SVM': SVC(kernel='linear'),
        'DT': tree.DecisionTreeClassifier()

    }
    multiple_classify(outfile, collection_set['BBC_Result'],classifier_set['NB'])
    multiple_classify(outfile, collection_set['BBC_Result'],classifier_set['SVM'])
    #multiple_classify(outfile, collection_set['BBC_Result'],classifier_set['DT'])
    # Database Input
    database.LabelDecoder(collection_set['BBC_Result'])

    pipeline = [{
        '$lookup': {
            'from': 'BBC_News_FeatureKeyWords_CountVectorizer',
            'localField': 'Content',
            'foreignField' : 'Content',
            'as': 'Keywords_CountV'
        }},{
        '$lookup': {
            'from': 'BBC_News_FeatureKeyWords_tdidf',
            'localField': 'Content',
            'foreignField' : 'Content',
            'as': 'Keywords_tdidf'
        }},{
        '$lookup': {
            'from': 'BBC_News_FeatureKeyWords_tdidf_Biagram',
            'localField': 'Content',
            'foreignField' : 'Content',
            'as': 'Keywords_tdidf_Biagram'
        }},{
        '$project': {
            'Label':{'$arrayElemAt':['$Keywords_CountV.Label',0]},
            'Keywords_tdidf': {'$arrayElemAt':['$Keywords_tdidf.KeyWords', 0]},
            'Keywords_tdidf_Biagram': {'$arrayElemAt':['$Keywords_tdidf_Biagram.KeyWords', 0]},
            'Keywords_CountV': {'$arrayElemAt':['$Keywords_CountV.KeyWords', 0]},
            'Content':{'$arrayElemAt':['$Keywords_CountV.Content',0]}}}

    ]
    update_result = db['BBC_News'].aggregate(pipeline)

    for x in update_result:
        #print(x)
        db['BBC_News'].update({
            '_id':x['_id']
        }, {
            'Label':x['Label'],
            'Keywords_tdidf':x['Keywords_tdidf'],
            'Keywords_tdidf_Biagram':x['Keywords_tdidf_Biagram'],
            'Keywords_CountV':x['Keywords_CountV'],
            'Content':x['Content']
        })

    query = {
        "$where": "this.Predict_Label != this.Actual_Label"
    }
    answer = collection_set['BBC_Result'].find(query).sort('Num')
    #for x in answer:
     #   print(x)
    #feature_keywords("/Users/frank/PycharmProjects/FYP_classification/RAKE-tutorial/articles/txt/EIE3105.pdf.txt")


if __name__=='__main__':
    main()
