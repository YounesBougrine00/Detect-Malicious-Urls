from django.shortcuts import render

# Create your views here.

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# I . Data exploration and sanitization
def home(request):
    url=request.GET.get('url')
    df = pd.read_csv("C:/Users/infoo/Desktop/securityApp/scanApp/static/test.csv",error_bad_lines=False,encoding='latin1') 
    df = pd.DataFrame(df)
    df = df.sample(n=10000)
    from io import StringIO
    col = ['label','url']
    df = df[col]


    #Deleting nulls

    df = df[pd.notnull(df['url'])]

    #more settings for our data manipulation

    df.columns = ['label', 'url']
    df['category_id'] = df['label'].factorize()[0]
    category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'label']].values)


    #Verify : tokenizer function for URL
    def getTokens(input):
        tokensBySlash = str(input.encode('utf-8')).split('/')
        allTokens=[]
        for i in tokensBySlash:
            tokens = str(i).split('-')
            tokensByDot = []
            for j in range(0,len(tokens)):
                tempTokens = str(tokens[j]).split('.')
                tokentsByDot = tokensByDot + tempTokens
            allTokens = allTokens + tokens + tokensByDot
        allTokens = list(set(allTokens))
        if 'com' in allTokens:
            allTokens.remove('com')
        return allTokens

    y = [d[1]for d in df] #labels
    myUrls = [d[0]for d in df] #urls
    vectorizer = TfidfVectorizer( tokenizer=getTokens ,use_idf=True, smooth_idf=True, sublinear_tf=False)
    features = vectorizer.fit_transform(df.url).toarray()
    labels = df.label
    features.shape


    model = LogisticRegression(random_state=0)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.20, random_state=0)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    clf = LogisticRegression(random_state=0) 
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print ('train accuracy =', train_score)
    print ('test accuracy =', test_score)

    # X_predict = ['https://stackoverflow.com','www.radsport-voggel.de/wp-admin/includes/log.exe','https://www.kdnuggets.com']
    # X_predict = vectorizer.transform(X_predict)
    # y_Predict = clf.predict(X_predict)
    # print(y_Predict)


    y_Predict=""
    if url:
        X_predict = [url]
        X_predict = vectorizer.transform(X_predict)
        y_Predict = clf.predict(X_predict)
        print(y_Predict)
    context={"y_predict":y_Predict}
    return render(request,'index.html',context)