import pandas as pd
import numpy as np
import re, collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2,SelectKBest

pd.set_option('display.max_columns', 3000)
pd.set_option('display.width', 3000)
news = pd.read_csv("Dataset/bbc-text.csv")
news['category_ID'] = news['category'].factorize()[0]

tech_news = news[news['category_ID'] == 0]
bus_news = news[news['category_ID'] == 1]
sport_news = news[news['category_ID'] == 2]
enter_news = news[news['category_ID'] == 3]
pol_news = news[news['category_ID'] == 4]
news_dict = [tech_news,bus_news,sport_news,enter_news,pol_news]

text = news.pop('text')
news.insert(2, 'Text', text)
category_array = news.pop('category').values #DataFrame to Array
category_ID_array = news.pop('category_ID').values #DataFrame to Array
CategoryFrame=pd.DataFrame({
    'cate':category_array,
    'cate_id':category_ID_array
})
#print(news)
#print(CategoryFrame)
def wordsbag(news):
    news.drop(['category', 'category_ID'], axis=1)
    words_box=[]
    for index, row in news.iterrows():
        words_box.extend(row['text'].strip().split())
    result = collections.Counter(words_box)
    dataframe = pd.DataFrame.from_dict(result, orient='index')
    dataframe = dataframe.rename(columns={'index':'word', 0:'count'}).sort_values(by=['count'],ascending=False)
    print(result)
    #print(dataframe)

def tfidf_generator(news):
    category_ID_array = news.pop('category_ID').values
    category_array = news.pop('category').values #DataFrame to Array
    #news.drop(['category', 'category_ID'], axis=1)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(news.text)
    word = vectorizer.get_feature_names() #feature names
    #print(X.toarray())
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X) #查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
    weight = tfidf.toarray()
    print(weight)
    weight_frame = pd.DataFrame(weight,columns=word)
    lb = preprocessing.LabelBinarizer()
    category_array_Binarized = lb.fit_transform(category_ID_array)
    observed = np.dot(category_array_Binarized.T, weight)

    print(observed)
    #List = np.array(observed[1])
    chiscore = chi2(weight, category_ID_array)
    kbest = SelectKBest(score_func=chi2,k=4)
    #weight_kbest = kbest.fit_transform(observed, category_array)
    frame2 = pd.DataFrame({
        'Word':word,
        'Chi':chiscore[0]
    }).sort_values(by=['Chi'],ascending=False)
    print(frame2)

def test():
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(news.Text)
    word = vectorizer.get_feature_names() #feature names
    #print(X.toarray())
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X) #查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
    weight = tfidf.toarray()
    weight_frame = pd.DataFrame(weight,columns=word)
    # print(weight_frame)


    lb = preprocessing.LabelBinarizer()
    category_array_Binarized = lb.fit_transform(category_ID_array)
    print(category_array_Binarized)
    observed = np.dot(category_array_Binarized.T, weight)
    print(observed)
    List_cate_0 = np.array(observed[1])

    frame3 = pd.DataFrame({
        'Word':word,
        'Count of terms':List_cate_0}).sort_values(by=['Count of terms'],ascending=False)
    print(frame3)
    # the second row of the observed array refers to the total count of the terms that belongs to class 1.
    # Then we compute the expected frequencies of each term for each class.
    chiscore = chi2(weight,category_ID_array)
    kbest = SelectKBest(score_func=chi2,k=4)
    #weight_kbest = kbest.fit_transform(observed, category_array)
    frame2 = pd.DataFrame({
        'Word':word,
        'Chi':chiscore[0]
    }).sort_values(by=['Chi'],ascending=False)
    print(frame2)

    """
    tfidf2 = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf2.fit_transform(news.Text).toarray()
    labels = news.category_ID
    features.shape
    """


    frame1 = pd.DataFrame(weight,columns=word)
    #print(frame1)

    #print (features)
    """
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
            print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
            for j in range(len(word)):
                if (weight[i][j] > 0.1):
                    print(word[j], weight[i][j])
                    print(word[j], weight2[i][j]) 
    """


def main():
    for new in news_dict:
        wordsbag(new)
    #tfidf_generator(tech_news)

if __name__=='__main__':
    main()
