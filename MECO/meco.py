import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect, SelectFromModel, chi2, SelectFpr, SelectFdr, RFE, RFECV,VarianceThreshold,SelectKBest
from sklearn.svm import SVR, LinearSVC, LinearSVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score,confusion_matrix, accuracy_score
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time


def funVarianceThreshold(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

    rfc_model = LogisticRegression(solver='lbfgs')
    rfc_model.fit(X_train, y_train)
    rfc_prediction = rfc_model.predict(X_test)
    print("Accuracy: ", accuracy_score(rfc_prediction, y_test))

    for i in range(8):
        threshold = i * 0.01
        start_time = time.time()
        selector = VarianceThreshold(threshold=threshold)
        fit = selector.fit(X_train, y_train)
        X_new_train = X_train[X_train.columns[selector.get_support(indices=True)]]
        finish_time=time.time()
        X_new_test = X_test[X_test.columns[selector.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_new_train, y_train)
        rfc_prediction = rfc_model.predict(X_new_test)
        print("Threshold: ", threshold, "Accuracy: ", accuracy_score(rfc_prediction, y_test), "  Time = ", finish_time-start_time)
        print(X_new_train.shape[1])
    return X

def funSelectKBest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    fit = SelectKBest(chi2, k=10).fit(X_train, y_train)
    X_train_new = X_train[X_train.columns[fit.get_support(indices=True)]]
    finish_time=time.time()
    print("time = ", finish_time-start_time)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores)

    acc_score = [0]*16
    for i in range(1,17):
        fit = SelectKBest(score_func=chi2, k=i).fit(X_train,y_train)
        X_train_new = X_train[X_train.columns[fit.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[fit.get_support(indices=True)]]
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train_new, y_train)
        prediction = model.predict(X_test_new)
        acc_score[i -1] = accuracy_score(prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])

    plt.figure(1, figsize=(10, 6))
    plt.clf()
    plt.plot([i+1 for i in range(16)], acc_score, linewidth=2)
    plt.axis('tight')
    plt.xlabel('Amount of features')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title("SelectKBest method")
    plt.show()

def funPCA(X, y):
    start_time = time.time()
    pca = PCA().fit(X)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        fit = PCA(n_components=i).fit(X)
        X_new = pd.DataFrame(fit.transform(X))
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=27)
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc_score[i -1] = accuracy_score(prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])

    plt.figure(1, figsize=(10, 6))
    plt.clf()
    # plt.axes('tight')
    plt.plot([i+1 for i in range(16)], acc_score, linewidth=2)
    plt.axis('tight')
    plt.xlabel('Amount of features')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title("PCA")
    plt.show()

def mySelectFromModel(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    if model == 1:
        model = LogisticRegression(solver='lbfgs')
    elif model == 2:
        model = RandomForestClassifier(n_estimators = 30)
    elif model == 3:
        model = ExtraTreesClassifier(n_estimators = 30)
    else:
        print("Invalid number of model")
        return
    smf = SelectFromModel(model)#, prefit=True)
    smf.fit(X_train, y_train)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        smf = SelectFromModel(model, max_features=i, threshold=-np.inf)#, prefit=True)
        smf.fit(X_train, y_train)
        X_train_new = X_train[X_train.columns[smf.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[smf.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_train_new, y_train)
        rfc_prediction = rfc_model.predict(X_test_new)
        acc_score[i - 1] = accuracy_score(rfc_prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])
    return acc_score

def myRFE(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    if model == 1:
        model = LogisticRegression(solver='lbfgs')
    elif model == 2:
        model = RandomForestClassifier(n_estimators = 30)
    elif model == 3:
        model = ExtraTreesClassifier(n_estimators = 30)
    else:
        print("Invalid number of model")
        return
    fit = RFE(model, 1).fit(X_train, y_train)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        fit = RFE(model, i).fit(X_train, y_train)
        X_train_new = X_train[X_train.columns[fit.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[fit.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_train_new, y_train)
        rfc_prediction = rfc_model.predict(X_test_new)
        acc_score[i - 1] = accuracy_score(rfc_prediction, y_test)
        print("Features:", i, ". Accuracy:", acc_score[i -1])
    return acc_score

def diagramOfMethods(title, model1, model2, model3 = 'null'):
    plt.figure(1, figsize=(10,6))
    plt.clf()
    features = [i+1 for i in range(16)]
    plt.plot(features, model1, 'r-')
    plt.plot(features, model2, 'b--')
    if model3 != 'null':
        plt.plot(features, model3, 'g-.')
    plt.axis('tight')
    plt.xlabel('Amount of features')
    plt.ylabel('Accuracy')
    plt.grid()
    if model3 != 'null':
        plt.legend(['LogisticRegression', 'RandomForestClassifier', 'ExtraTreesClassifier'])
    else:
        plt.legend(['SelectFromModel', 'RFE'])
    plt.title("Accuracy of  " + title)
    plt.show()

def scatterDiagram(X, y):
    X_new = X.drop(['3'], axis='columns').drop(['4'], axis='columns').drop(['5'], axis='columns').drop(['6'], axis='columns').drop(['7'], axis='columns').drop(['8'], axis='columns').drop(['9'], axis='columns').drop(['11'], axis='columns').drop(['15'], axis='columns').drop(['16'], axis='columns')
    # X_new = pd.DataFrame({X['1'],X['2'],X['10'],X['12'],X['13'],X['14']})
    pd.plotting.scatter_matrix(X_new, c=y, figsize=(6, 6))
    plt.show()

def preparationData():
    data = pd.read_csv('data/data1_train.csv')

    data = data.drop(['ID'], axis=1).loc[data['pdays'] != -1]

    data['subscribed'] = data['subscribed'].replace(to_replace=['no', 'yes'], value=[0, 1])
    data['job'] = data['job'].replace(to_replace=['admin.', 'entrepreneur', 'blue-collar', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'], value=[1,2,3,4,5,6,7,8,9,10,11,0])
    data['marital'] = data['marital'].replace(to_replace=['divorced', 'married', 'single'], value=[1,2,3])
    data['education'] = data['education'].replace(to_replace=['primary', 'secondary', 'tertiary', 'unknown'], value=[1, 2, 3, 0])
    data['default'] = data['default'].replace(to_replace=['no', 'yes'], value=[0, 1])
    data['housing'] = data['housing'].replace(to_replace=['no', 'yes'], value=[0, 1])
    data['loan'] = data['loan'].replace(to_replace=['no', 'yes'], value=[0, 1])
    data['contact'] = data['contact'].replace(to_replace=['cellular', 'telephone', 'unknown'], value=[1, 2, 0])
    data['month'] = data['month'].replace(to_replace=['apr', 'aug', 'dec', 'feb', 'jan', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep'], value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    data['poutcome'] = data['poutcome'].replace(to_replace=['failure', 'success', 'other', 'unknown'], value=[1, 2, 3, 0])

    X = data.drop(['subscribed'], axis=1)
    Y = data['subscribed']
    data_nm = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
    X_nm = data_nm.drop(['subscribed'], axis=1)
    Y_nm = data_nm['subscribed']
    X.columns = X_nm.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
    return X, Y, X_nm, Y_nm

if __name__ == '__main__':
    x, y, x_nm, y_nm = preparationData()

    # funVarianceThreshold(x_nm, y_nm)
    # funSelectKBest(x_nm, y_nm)
    # funPCA(x_nm, y_nm)
    #
    # #Logical regression
    acc_score_SFM_LR = mySelectFromModel(x_nm, y_nm, 1)
    acc_score_RFE_LR = myRFE(x_nm, y_nm, 1)
    #
    # #Random forest classification
    acc_score_SFM_RFC = mySelectFromModel(x_nm, y_nm, 2)
    acc_score_RFE_RFC = myRFE(x_nm, y_nm, 2)
    #
    # #Extra trees clas
    acc_score_SFM_ETC = mySelectFromModel(x_nm, y_nm, 3)
    acc_score_RFE_ETC = myRFE(x_nm, y_nm, 3)
    #
    # #Comparison SFM
    diagramOfMethods('SelectFromModel method', acc_score_SFM_LR, acc_score_SFM_RFC, acc_score_SFM_ETC)
    diagramOfMethods('RFE method', acc_score_RFE_LR, acc_score_RFE_RFC, acc_score_RFE_ETC)
    diagramOfMethods('Extra trees classifier', acc_score_SFM_LR, acc_score_RFE_LR)
    diagramOfMethods('Extra trees classifier', acc_score_SFM_RFC, acc_score_RFE_RFC)
    diagramOfMethods('Extra trees classifier', acc_score_SFM_ETC, acc_score_RFE_ETC)
    #
    # summ1 = 0
    # summ2 = 0
    # summ3 = 0
    # summ4 = 0
    # summ5 = 0
    # for i in range(16):
    #     summ1 += acc_score_SFM_LR[i] + acc_score_RFE_LR[i]
    #     summ2 += acc_score_SFM_RFC[i] + acc_score_RFE_RFC[i]
    #     summ3 += acc_score_SFM_ETC[i] + acc_score_RFE_ETC[i]
    #     summ4 += acc_score_SFM_ETC[i]
    #     summ5 += acc_score_RFE_ETC[i]
    # print(summ1/summ2)
    # print(summ1/summ3)
    # print(summ4/summ5)
