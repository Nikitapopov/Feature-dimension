import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect, SelectFromModel, chi2, SelectFpr, SelectFdr, RFE, RFECV, VarianceThreshold, SelectKBest
from sklearn.svm import SVR, LinearSVC, LinearSVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score,confusion_matrix, accuracy_score
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time


def diagramOfMethods(title, model1, model2, model3 = 'null'):
    plt.figure(1, figsize=(14, 13))
    plt.clf()
    features = [i+1 for i in range(16)]
    plt.plot(features, model1, 'x-')
    plt.plot(features, model2, '*-')
    if model3 != 'null':
        plt.plot(features, model3, 'p-')
    plt.axis('tight')
    plt.xlabel('n features')
    plt.ylabel('accuracy')
    plt.grid()
    if model3 != 'null':
        plt.legend(['LogisticRegression', 'RandomForestClassifier', 'ExtraTreesClassifier'])
    else:
        plt.legend(['SelectFromModel', 'RFE'])
    plt.title("Accuracy of  " + title)
    plt.show()

    X_1 = sm.add_constant(X)
    # Fitting sm.OLS model
    model = sm.OLS(y, X_1).fit()
    res = model.pvalues[1:13]

def scatterDiagram(X, y):
    X_new = X.drop(['3'], axis='columns').drop(['4'], axis='columns').drop(['5'], axis='columns').drop(['6'], axis='columns').drop(['7'], axis='columns').drop(['8'], axis='columns').drop(['9'], axis='columns').drop(['11'], axis='columns').drop(['15'], axis='columns').drop(['16'], axis='columns')
    # X_new = pd.DataFrame({X['1'],X['2'],X['10'],X['12'],X['13'],X['14']})
    pd.plotting.scatter_matrix(X_new, c=y, figsize=(6, 6))
    plt.show()

def preparationData(dataset, numDS):
    data = pd.read_csv(dataset)

    if numDS == 0:
        data = data.drop(['ID'], axis=1).loc[data['pdays'] != -1]
        label = LabelEncoder()
        dicts = {}
        for i in data:
            if type(i) != int or type(i) != double:
                label.fit(data[i].drop_duplicates())
                dicts[i] = list(label.classes_)
                data[i] = label.transform(data[i])

        X = data.drop(['subscribed'], axis=1)
        Y = data['subscribed']
        data_nm = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
        X_nm = data_nm.drop(['subscribed'], axis=1)
        Y_nm = data_nm['subscribed']
        X.columns = X_nm.columns = [i for i in range(1,17)]

    elif(numDS == 1):
        label = LabelEncoder()
        dicts = {}
        for i in data:
            if type(i) != int or type(i) != double:
                label.fit(data[i].drop_duplicates())
                dicts[i] = list(label.classes_)
                data[i] = label.transform(data[i])

        X = data.drop(['y'], axis=1)
        Y = data['y']
        data_nm = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
        X_nm = data_nm.drop(['y'], axis=1)
        Y_nm = data_nm['y']
        X.columns = X_nm.columns = [i for i in range(1,21)]

    elif numDS == 2:
        data = data.loc[data['pdays'] != -1]
        label = LabelEncoder()
        dicts = {}
        for i in data:
            if type(i) != int or type(i) != double:
                label.fit(data[i].drop_duplicates())
                dicts[i] = list(label.classes_)
                data[i] = label.transform(data[i])

        X = data.drop(['deposit'], axis=1)
        Y = data['deposit']
        data_nm = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
        X_nm = data_nm.drop(['deposit'], axis=1)
        Y_nm = data_nm['deposit']
        X.columns = X_nm.columns = [i for i in range(1,17)]


    return X, Y, X_nm, Y_nm


def funVarianceThreshold(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    # dfvariances = pd.DataFrame(fit.variances_)
    # dfcolumns = pd.DataFrame(X.columns)
    # featureVar = pd.concat([dfcolumns, dfvariances], axis=1)
    # featureVar.columns = ['Specs', 'Variances']
    # print(featureVar)
    # print(X[X.columns[selector.get_support(indices=True)]])
    # print(cross_val_score(LogisticRegression(solver='lbfgs'), X, y, scoring='neg_log_loss', cv=5).mean())

    rfc_model = LogisticRegression(solver='lbfgs')
    rfc_model.fit(X_train, y_train)
    rfc_prediction = rfc_model.predict(X_test)
    print("Accuracy: ", accuracy_score(rfc_prediction, y_test))

    start_time = time.time()
    selector = VarianceThreshold(threshold=0.0)
    fit = selector.fit(X_train, y_train)
    X_new_train = X_train[X_train.columns[selector.get_support(indices=True)]]
    finish_time=time.time()
    print("time = ", finish_time-start_time)
    X_new_test = X_test[X_test.columns[selector.get_support(indices=True)]]
    rfc_model = LogisticRegression(solver='lbfgs')
    rfc_model.fit(X_train, y_train)
    rfc_prediction = rfc_model.predict(X_test)
    print("Accuracy: ", accuracy_score(rfc_prediction, y_test))

    start_time = time.time()
    selector = VarianceThreshold(threshold=0.005)
    fit = selector.fit(X_train, y_train)
    X_new_train = X_train[X_train.columns[selector.get_support(indices=True)]]
    finish_time=time.time()
    print("time = ", finish_time-start_time)
    X_new_test = X_test[X_test.columns[selector.get_support(indices=True)]]
    rfc_model = LogisticRegression(solver='lbfgs')
    rfc_model.fit(X_new_train, y_train)
    rfc_prediction = rfc_model.predict(X_new_test)
    print("Accuracy: ", accuracy_score(rfc_prediction, y_test))

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

    # feat_importances = pd.Series(fit.scores_, index=X.columns)
    # feat_importances.plot(kind='barh')
    # plt.xlabel('Rate')
    # plt.ylabel('Feature')
    # plt.grid()
    # plt.title("Importance rating")
    # plt.xscale('log')
    # plt.show()

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

    plt.figure(1, figsize=(14, 13))
    plt.clf()
    plt.plot([i+1 for i in range(16)], acc_score, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n features')
    plt.ylabel('accuracy')
    plt.grid()
    plt.title("Accuracy")
    plt.show()

    # plt.figure(1, figsize=(14, 13))
    # plt.clf()
    # plt.plot(val_scores, linewidth=2)
    # plt.axis('tight')
    # plt.xlabel('n_components')
    # plt.ylabel('val_scores')
    # plt.show()

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

    # feature_coef = smf.estimator_.coef_[0]
    # feature_importance = pd.DataFrame({'features': list(X_train.columns),
    #                                    'coef': abs(feature_coef)})
    # print(feature_importance)

    # summa = 0
    # for j in range(0, 100):
    #     rfc_model = LogisticRegression(solver='lbfgs')
    #     rfc_model.fit(X_train, y_train)
    #     rfc_prediction = rfc_model.predict(X_test)
    #     summa += accuracy_score(rfc_prediction, y_test)
    # print(summa)
    # print("Accuracy: ", accuracy_score(rfc_prediction, y_test))

    acc_score = [0]*16
    # for j in range(0, 10):
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
    # feat_useless = pd.Series(fit.ranking_, index=X.columns)
    # feat_importances = -feat_useless + 17
    # print(feat_importances)
    # feat_importances.plot(kind='barh')
    # plt.xlabel('Rate')
    # plt.ylabel('Feature')
    # plt.title("Importance rating")
    # plt.show()

    acc_score = [0]*16
    for i in range(1,17):
        fit = RFE(model, i).fit(X_train, y_train)
        X_train_new = X_train[X_train.columns[fit.get_support(indices=True)]]
        X_test_new = X_test[X_test.columns[fit.get_support(indices=True)]]
        rfc_model = LogisticRegression(solver='lbfgs')
        rfc_model.fit(X_train_new, y_train)
        print(X_test_new)
        rfc_prediction = rfc_model.predict(X_test_new)
        acc_score[i - 1] = accuracy_score(rfc_prediction, y_test)
        print("Features:", i, ". Accuracy:", acc_score[i -1])
    return acc_score


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
        # fit_tr = PCA(n_components=i).fit(X_train)
        # fit_te = PCA(n_components=i).fit(X_test)
        # X_train_new = fit_tr.transform(X_train)
        # X_train_new = fit_te.transform(X_test)
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc_score[i -1] = accuracy_score(prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])

    plt.figure(1, figsize=(14, 13))
    plt.clf()
    # plt.axes('tight')
    plt.plot([i+1 for i in range(16)], acc_score, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n features')
    plt.ylabel('accuracy')
    plt.grid()
    plt.title("Accuracy")
    plt.show()

    # plt.figure(1, figsize=(14, 13))
    # plt.clf()
    # plt.axes([.2, .2, .7, .7])
    # plt.plot(pca.explained_variance_ratio_, linewidth=2)
    # plt.axis('tight')
    # plt.xlabel('n components')
    # plt.ylabel('explained_variance_ratio_')
    # plt.show()


def funICA(X, y):  #Independent Component Analysis
    start_time = time.time()
    fastICA = FastICA().fit(X)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        fit = FastICA(n_components=i).fit(X)
        X_new = pd.DataFrame(fit.transform(X))
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=27)
        # fit_tr = PCA(n_components=i).fit(X_train)
        # fit_te = PCA(n_components=i).fit(X_test)
        # X_train_new = fit_tr.transform(X_train)
        # X_train_new = fit_te.transform(X_test)
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc_score[i -1] = accuracy_score(prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])

    plt.figure(1, figsize=(14, 13))
    plt.clf()
    # plt.axes('tight')
    plt.plot([i+1 for i in range(16)], acc_score, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n features')
    plt.ylabel('accuracy')
    plt.grid()
    plt.title("Accuracy")
    plt.show()


def funLDA(X, y):  #ILinear Discriminant Analysis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    start_time = time.time()
    fastICA = LinearDiscriminantAnalysis().fit(X, y)
    finish_time=time.time()
    print("time = ", finish_time-start_time)

    acc_score = [0]*16
    for i in range(1,17):
        fit = LinearDiscriminantAnalysis(n_components=i).fit(X_train, y_train)
        X_train_new = fit.transform(X_train)
        X_test_new = fit.transform(X_test)
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test_new)
        acc_score[i -1] = accuracy_score(prediction, y_test)
        print("Features: ", i, ". Accuracy: ", acc_score[i - 1])

    plt.figure(1, figsize=(14, 13))
    plt.clf()
    # plt.axes('tight')
    plt.plot([i+1 for i in range(16)], acc_score, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n features')
    plt.ylabel('accuracy')
    plt.grid()
    plt.title("Accuracy")
    plt.show()

if __name__ == '__main__':
    x1, y1, x_nm1, y_nm1 = preparationData('data/data1_train.csv', 0)
    x2, y2, x_nm2, y_nm2 = preparationData('data/data2.csv', 1)
    x3, y3, x_nm3, y_nm3 = preparationData('data/data3.csv', 2)

    print(np.shape(x1))
    print(np.shape(x2))
    print(np.shape(x3))

    # funVarianceThreshold(x_nm, y_nm)
    # funSelectKBest(x_nm, y_nm)
    #
    # ---------- Logical regression ----------
    # acc_score_SFM_LR = mySelectFromModel(x_nm, y_nm, 1)
    # acc_score_RFE_LR = myRFE(x_nm, y_nm, 1)
    #
    # ------ Random trees classification -----
    # acc_score_SFM_RFC = mySelectFromModel(x_nm, y_nm, 2)
    # acc_score_RFE_RFC = myRFE(x_nm, y_nm, 2)
    #
    # ------ Extra trees classification ------
    # acc_score_SFM_ETC = mySelectFromModel(x_nm, y_nm, 3)
    # acc_score_RFE_ETC = myRFE(x_nm, y_nm, 3)
    #
    # ------------ Comparison SFM ------------
    # diagramOfMethods('method SelectFromModel', acc_score_SFM_LR, acc_score_SFM_RFC, acc_score_SFM_ETC)
    # diagramOfMethods('method RFE', acc_score_RFE_LR, acc_score_RFE_RFC, acc_score_RFE_ETC)
    # diagramOfMethods('model logistic regression', acc_score_SFM_RFC, acc_score_RFE_RFC)
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

    # ----------------------- FEATURE EXTRACTION -----------------------

    # funPCA(x_nm, y_nm)
    # funICA(x_nm, y_nm)
    # funLDA(x_nm, y_nm)

