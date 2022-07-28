from turtle import pd
import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def show_rate(train_df, test_df):
    #객실 등급별 생존 확률 보기
    Surv_class = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values("Survived", ascending=False)
    print(Surv_class)

    #성별에 따른 생존 확률
    Surv_sex = train_df[['Sex', "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values("Survived",ascending=False)
    print(Surv_sex)

    #승선한 가족수에 따른 생존확률
    Surv_sib = train_df[['SibSp', "Survived"]].groupby(['SibSp'], as_index=False).mean()
    print(Surv_sib)

    #승선한 부모와 자식의 수에 따른 생존 확률
    Surv_par = train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
    print(Surv_par)

    #나이에 따른 생존확률
    graph = sns.FacetGrid(train_df, col='Survived')
    graph.map(plt.hist, 'Age', bins=3) # bins = 나이를 얼마만큼의 수의 군집으로 나눌 것인가?
    plt.show()

    #객실 등급과 생존 여부에 따른 연령 분포
    graph = sns.FacetGrid(train_df, col='Survived', row="Pclass", hue="Pclass", height=2.2, aspect=1.6)
    graph.map(plt.hist, "Age", alpha=0.5, bins=20)
    graph.add_legend()
    plt.show()

    #승선지와 객실 등급에 따른 생존률
    graph = sns.FacetGrid(train_df, row='Embarked')
    graph.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order=[1, 2, 3], hue_order=["male", 'female'])
    graph.add_legend()
    plt.show()

    #승선지, 생존여부, 성별에 따른 요금
    g = sns.FacetGrid(train_df, row = "Survived", col="Embarked")
    g.map(sns.barplot, 'Sex', "Fare", ci=None, order=["male", 'female'])
    g.add_legend()
    plt.show()

def preprocess(train_df, test_df):
    train_df.drop(["PassengerId", "Name", 'Ticket', "Cabin"], axis=1, inplace=True)
    test_df.drop(["PassengerId", "Name", 'Ticket', "Cabin"], axis=1, inplace=True)

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_df['Sex'] = train_df['Sex'].map({"female": 0, "male": 1}).astype(int)
    test_df['Sex'] = test_df['Sex'].map({"female": 0, "male": 1}).astype(int)

    train_df['Embarked']  = train_df["Embarked"].map({"S":0, "C": 1, "Q": 2}).astype(int)
    test_df['Embarked'] = test_df["Embarked"].map({'S':0, 'C': 1, 'Q':2})

    return train_df, test_df

def logistic_regression(train_x, train_y, test_x, test_y):
    logreg = LogisticRegression()
    logreg.fit(train_x, train_y)
    y_pred = logreg.predict(test_x)
    acc = round(logreg.score(train_x, train_y)*100, 2)
    test_acc = accuracy_score(test_y, y_pred)
    print("----------Logistic Regression Accuracy----------")
    print("acc: ", acc)
    print("test acc: ", test_acc)

def svm(train_x, train_y, test_x, test_y):
    svc = SVC()
    svc.fit(train_x, train_y)
    y_pred = svc.predict(test_x)
    acc = round(svc.score(train_x, train_y)*100, 2)
    test_acc = accuracy_score(test_y, y_pred)
    print("-------------Support Vector Machine-------------")
    print("acc: ", acc)
    print("test acc: ", test_acc)
    

def svc_linear(train_x, train_y, test_x, test_y):
    svc_li = LinearSVC()
    svc_li.fit(train_x, train_y)
    y_pred = svc_li.predict(test_x)
    acc = round(svc_li.score(train_x, train_y)*100, 2)
    test_acc = accuracy_score(test_y, y_pred)
    print("---------------Linear SVC Accuracy--------------")
    print("acc: ", acc)
    print("test acc: ", test_acc)
    
if __name__ == "__main__":
    train = pd.read_csv("titanic/train.csv")
    test_x = pd.read_csv("titanic/test.csv")
    test_y = pd.read_csv("titanic/gender_submission.csv")

    temp = pd.concat([test_x, test_y], axis=1)

    train_df, test_df = preprocess(train, temp)

    train_x = train_df.drop("Survived", axis=1)
    train_y = train_df['Survived']
    test_x = test_df.drop("Survived", axis=1)
    test_y = test_df['Survived']

    logistic_regression(train_x, train_y, test_x, test_y)
    svm(train_x, train_y, test_x, test_y)
    svc_linear(train_x, train_y, test_x, test_y)