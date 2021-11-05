import pandas as pd
import re

trending = pd.read_csv("trending.csv", encoding = "UTF-8",index_col='video_id')
non_trending = pd.read_csv("non_trending.csv", encoding = "UTF-8",index_col='V_id')

stop_words={'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn','1','2','3','4','5','6','7','8','9','0','vs','le','la','de','none'}

def processTag(columns,tagRanking):         #Splits the given tags into words, removes nonascii characters, stop words and calculates the average rank of given tag
    tag=columns[0]
    trending=columns[1]
    tagAfterSplit=[]
    if(type(tag)==str):
        if(trending==1):
            tagFirstSplit=tag.split('|')
        else:
            tagFirstSplit=tag.split(',')
        for t in tagFirstSplit:
            tagSecondSplit=t.split()
            for t in tagSecondSplit:
                t=re.sub('[^^A-Za-z0-9]+','',t).lstrip()
                if(t):
                    
                    tagAfterSplit.append(t.lower())
                    
        filteredTag=[w for w in tagAfterSplit if not w in stop_words]
        filteredUniqueTag=set(filteredTag)
        sum=0
        for f in filteredUniqueTag:
            try:
                sum+=int(tagRanking[f])
            except Exception as e:
                sum+=1
        if(len(filteredUniqueTag)>0):
            average=(sum/len(filteredUniqueTag))
            return average
        else:
            return 1
    else:
        return tag

def processTitle(columns,wordRanking):          #Splits the given title into words, removes nonascii characters, stop words and calculates the average rank of given title
    title=columns[0]
    titleAfterSplit=[]
    if(type(title)==str):
        temp=title.split()
        for t in temp:
            t=re.sub('[^^A-Za-z0-9]+','',t).lstrip()
            if(t):
                titleAfterSplit.append(t.lower())
        filteredTitle=[w for w in titleAfterSplit if not w in stop_words]
        filteredUniqueTitle=set(filteredTitle)
        sum=0
        for f in filteredUniqueTitle:
            try:
                sum+=int(wordRanking[f])
            except Exception as e:
                sum+=1
        if(len(filteredUniqueTitle)>0):
            average=(sum/len(filteredUniqueTitle))
            return average
        else:
            return 1
    else:
        return title

def prepareTagRanking(youTubeTrendingData):               #Splits the Title and tags from trending data into words,removes nonascii characters, removes stop words and calculates the number occurences these words for ranking  
    tags=[]
    for tag in youTubeTrendingData['tags']:
        if(type(tag) == str):
            temp=tag.split('|')
            for t in temp:
                temp1=t.split()
                for t1 in temp1:
                    t1=re.sub('[^A-Za-z0-9]+', '', t1).lstrip()
                    if (t1):
                        tags.append(t1.lower())
    
    for title in youTubeTrendingData['title']:
        if(type(title)==str):
            temp=title.split()
            for t in temp:
                t=re.sub('[^A-Za-z0-9]+', '', t).lstrip()
                if(t):
                    tags.append(t.lower())
    

    filtered_tags = [w for w in tags if not w in stop_words]
    if(len(filtered_tags)):
        tags=pd.DataFrame(data=filtered_tags)
        tags.columns=['tags']
        tagCount=tags['tags'].value_counts()
        return tagCount
    return []

#trending.info()
#non_trending.info()

#cleaning

trending.drop(['categoryId','description','trending_date','publishedAt','thumbnail_link','channelTitle','comments_disabled','ratings_disabled'],axis=1,inplace=True)
non_trending.drop(['categoryId','description','publishedAt','thumbnail','definition','privacyStatus','dimension','projection','caption','license','embeddable','licencedContent','defaultAudioLanguage','duration'],axis=1,inplace=True)

non_trending.rename(columns={'likeCount': 'likes','viewCount': 'view_count','dislikeCount': 'dislikes','commentCount':'comment_count'}, inplace=True)

# trending.info()
# print('---------------------')
# non_trending.info()
trending['isTrending'] = 1
non_trending['isTrending'] = 0

#removing rows
#trending = trending[:1000]
#non_trending = non_trending[:1000]





yt_data = pd.concat([trending, non_trending])
yt_data.fillna(0, inplace=True) #nullto0
#yt_data.info()
#print(yt_data.describe())



#tag_title_processing
tagRanking=prepareTagRanking(trending)
yt_data['tags']=yt_data[['tags','isTrending']].apply(processTag,axis=1,args=(tagRanking,))
yt_data['title']=yt_data[['title']].apply(processTitle,axis=1,args=(tagRanking,))

print("Success")

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = yt_data.iloc[:,2:7]  
y = yt_data.iloc[:,-1]    

# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import MinMaxScaler
# scalar = MinMaxScaler()
# scalar.fit(X)
# X = scalar.transform(X)
# # define and fit the final model
# model = Sequential()
# model.add(Dense(4, input_dim=2, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam')
# model.fit(X, y, epochs=500, verbose=0)
# # new instances where we do not know the answer
# Xnew, _ = pd.read_csv("data1.csv", encoding = "UTF-8")
# Xnew = scalar.transform(Xnew)
# # make a prediction
# ynew = model.predict_classes(Xnew)
# # show the inputs and predicted outputs
# for i in range(len(Xnew)):
# 	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import sklearn.preprocessing as pre
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier

def createROCCurve(y_test,y_pred,heading):      #plots ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_pred = label_binarize(y_pred,classes=[0,1])
    y_test = label_binarize(y_test,classes=[0,1])
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(heading)
    plt.legend(loc="lower right")
    plt.show()

# featuresUsed=list(yt_data.columns.values)
# print('Features Used\n',featuresUsed)
# print('Total',len(featuresUsed))
# #Before Feature Selection
# #Splits the data into training and testing data. One-third of the data is selected for testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=25)
# #Logistic Regression is used for traning the model
# LogReg = LogisticRegression()
# LogReg.fit(X_train, y_train)
# y_pred = LogReg.predict(X_test)
# #Confusion matrix is used to calculate the accuracy of the model
# from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(y_test, y_pred)
# accuracy=((confusion_matrix[0][0]+confusion_matrix[1][1])/len(y_test)*100)
# print('\nAccuracy',accuracy,'\n')
# print('Classfication Report\n',classification_report(y_test, y_pred))
# createROCCurve(y_test,y_pred,'Before Applying Linear SVC For Feature Selection')

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
featuresUsed=list(yt_data.columns.values)
print('Features Used\n',featuresUsed)
print('Total',len(featuresUsed))


l=[5,10,15,25,40,50,100,200,500]
print('Value of K\t   Accuracy')
for i in l:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,)
    #K Neasrest Classifier with different 
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    actual = y_test
    accuracy = (np.sum(predictions == actual)/len(actual))*100
    print(i,'\t\t',accuracy)