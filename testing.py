from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
import pymongo
import numpy as np
import database_test as database
from sklearn import decomposition, ensemble
import pandas, numpy, string,os
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from nlp_rake import rake


#加载数据集
def pre_access():
    data = open("Dataset/corpus/corpus").read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        content = line.split()
        labels.append(content[0])
        texts.append(" ".join(content[1:]))
    #创建一个dataframe，列名为text和label
    trainDF = pandas.DataFrame()
    trainDF['label'] = labels
    trainDF['text'] = texts
    #print(trainDF)

    #将数据集分为训练集和验证集
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],shuffle=False)# test_size default=0.25
    # label编码为目标变量
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
    #print (train_x)

    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xvalid_tfidf = tfidf_vect.transform(valid_x)

    print (xtrain_tfidf)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)
    #外面
    # SVM on Ngram Level TF IDF Vectors
    word_accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
    NGram_accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print ("SVM, word Vectors: ", word_accuracy)
    print ("SVM, N-Gram Vectors: ", NGram_accuracy)

def create_model(d_train , d_test):
    print("训练样本 = %d" % len(xtrain_tfidf))
    print("测试样本 = %d" %len(xvalid_tfidf))
    vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2 ) #tf-idf特征抽取ngram_range=(1,2)
    features = vectorizer.fit_transform(d_train.title)
    print("训练样本特征表长度为 " + str(features.shape))
    # print(vectorizer.get_feature_names()[3000:3050]) #特征名展示
    test_features = vectorizer.transform(d_test.title)
    print("测试样本特征表长度为 "+ str(test_features.shape))
    #支持向量机
    #C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0
    svmmodel = svm.SVC(C = 1.0 , kernel= "linear") #kernel：参数选择有rbf, linear, poly, Sigmoid, 默认的是"RBF";

    nn = svmmodel.fit(features , d_train.sku)
    print(nn)
    # predict = svmmodel.score(test_features ,d_test.sku)
    # print(predict)
    pre_test = svmmodel.predict(test_features)
    d_test["pre_skuno"] = pre_test
    d_test.to_excel("wr60_svm_pre1012.xlsx", index=False)
    #外面
    print("对新样本进行60个车型预测")
    d_train = pd.read_excel("wr60_train1012.xlsx") #训练

    df = pd.read_excel("wr机器学习分析报告.xlsx",sheetname="01预测") #测试
    d_test = df[df.pre_category == 1]
    create_model(d_train, d_test)

def binary_classify():
    data = open("Dataset/corpus/corpus").read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        content = line.split()
        labels.append(content[0])
        texts.append(" ".join(content[1:]))
    #创建一个dataframe，列名为text和label
    trainDF = pandas.DataFrame()
    trainDF['label'] = labels
    trainDF['text'] = texts

    pipeline = Pipeline([
        ('tdidf_vectorizer',   TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)),
        ('classifier',         MultinomialNB())
    ])

    k_fold = model_selection.KFold(n_splits=5, shuffle=True)
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold.split(trainDF):

        train_text = trainDF['text'][train_indices]
        train_y = trainDF['label'][train_indices]
        test_text = trainDF['text'][test_indices]
        test_y = trainDF['label'][test_indices]
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        test_y = encoder.fit_transform(test_y)
        print(test_y)

        pipeline.fit(train_text, train_y)

        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, average='binary')
        scores.append(score)

    print('Total emails classified:', len(trainDF))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)

def RAKE():
    stoppath = '/RAKE-tutorial/data/stoplists/SmartStoplist.txt'

    rake_object = rake.Rake(stoppath, 5, 3, 4)

    sample_file = open("data/docs/fao_test/w2167e.txt", 'r', encoding="iso-8859-1")
    text = sample_file.read()

    keywords = rake_object.run(text)

    # 3. print results
    print("Keywords:", keywords)

def feature_keywords(outfile):
    data = open(outfile).read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        if len(line.split()) > 0:
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))
    wordslist = texts
    titlelist = labels

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(wordslist))

    words = vectorizer.get_feature_names()  #所有文本的关键字
    weight = tfidf.toarray()

    n = 5 # 前五位
    for (title, w) in zip(titlelist, weight):
        wordsdet= ['and','of','the','to','in','will','students','project','subject','assessment','hours','with','on','be','for','you','he','she','her','his']
        print (u'{}:'.format(title))
        # 排序
        loc = np.argsort(-w)
        i,j=0,0
        while j < n:
            if words[loc[i]] in wordsdet:
                i += 1
                continue
            print (u'-{}: {} {}'.format(str(j + 1), words[loc[i]], w[loc[i]]))
            i +=1
            j +=1
        print ('\n')

def multiple_classify(outfile, collection):
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
        ('classifier',         MultinomialNB())
    ])

    k_fold = model_selection.KFold(n_splits=5, shuffle=True)
    scores = []
    confusion = numpy.zeros((5,5))
    for train_indices, test_indices in k_fold.split(trainDF):

        sub_seriesNum = trainDF['seriesNum'][test_indices].tolist()
        train_text = trainDF['text'][train_indices]
        train_y = trainDF['label'][train_indices]
        test_text = trainDF['text'][test_indices]
        test_text_list= test_text.tolist()
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


def readfile(filename):
    wordsdet= ['and','of','the','to','in','will','students','project','subject','assessment','hours','with','on','be','for']
    titlelist,results = [],[]
    content = ''
    f = open(filename,encoding='utf-8',errors='replace')
    filecontents = f.readlines()
    for line in filecontents:
        content += line.strip('\n')
    title = "E"+filename.strip("/Users/frank/PycharmProjects/FYP_classification/RAKE-tutorial/articles/csv.")
    titlelist.append(title)
    results.append(content)
    print(results)
    print(titlelist)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(results))

    words = vectorizer.get_feature_names()  #所有文本的关键字
    weight = tfidf.toarray()

    n = 10 # 前五位
    for (title, w) in zip(titlelist, weight):
        print (u'{}:'.format(title))
        # 排序
        loc = np.argsort(-w)
        i=0
        j=0
        while j < n:
            if words[loc[i]] in wordsdet:
                i += 1
                continue
            print (u'-{}: {} {}'.format(str(j + 1), words[loc[i]], w[loc[i]]))
            i += 1
            j += 1
        print ('\n')

def main():
    # Set up database
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient['BBC_NEWS']
    collection_set = {
        #'Act' : db['Student_Account'],
        #'Sbj_Info': db['Subject_Info'],
        'BBC': db['BBC_News'],
        'BBC_Result': db['BBC_News_ClassificationResult']
    }
    # Preprocess
    targetfile = "Dataset/bbc-text.csv"
    outfile = database.preprocess(targetfile)
    # keywords
    feature_keywords(outfile)
    # Classify
    multiple_classify(outfile, collection_set['BBC_Result'])
    # Database Input
    database.LabelDecoder(collection_set['BBC_Result'])

    query = {
        "$where": "this.Predict_Label != this.Actual_Label"
    }
    answer = collection_set['BBC_Result'].find(query).sort('Num')
    #for x in answer:
     #   print(x)
    #feature_keywords("/Users/frank/PycharmProjects/FYP_classification/RAKE-tutorial/articles/txt/EIE3105.pdf.txt")
    csvDir = "/Users/frank/PycharmProjects/FYP_classification/RAKE-tutorial/articles/csv/"
    for filename in os.listdir(csvDir):
        readfile(csvDir+filename)

if __name__=='__main__':
    main()
