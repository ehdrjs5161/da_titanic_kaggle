from tkinter import E
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def set_pandas_display():
    pd.set_option("display.max_rows", 100)  # 출력할 최대 행 갯수를 100으로 설정
    pd.set_option("display.max_columns", 500)  # 출력할 최대 열 갯수를 500개로 설정
    pd.set_option("display.width", 500)  # 글자 수 기준 출력할 넓이 설정
    pd.set_option("max_info_columns", 500)  # 열 기반 info가 주어질 경우, 최대 넓이


def Decision_tree():
    return -1

if __name__ == "__main__":
    set_pandas_display()
    train = pd.read_csv("da_titanic_kaggle/titanic/train.csv")
    test_x = pd.read_csv("da_titanic_kaggle/titanic/test.csv")
    test_y = pd.read_csv("da_titanic_kaggle/titanic/gender_submission.csv")

    train.drop(['PassengerId', "Name", "Cabin", "Ticket"], axis='columns', inplace=True)
    test_x.drop(['PassengerId', "Name", "Cabin", "Ticket"], axis='columns', inplace=True)
    train.dropna(inplace=True)
    test_x.dropna(inplace=True)

    train.reset_index(inplace=True)
    train.drop(['index'], axis=1, inplace=True)
    train_y = train.pop("Survived")
    train_x = train

    sex_temp = []
    for data in train_x['Sex']:
        if data == "male":
            sex_temp.append(0)
        else:
            sex_temp.append(1)
    train_x['Sex'] = sex_temp

    tsex_temp = []
    for data in test_x['Sex']:
        if data == "male":
            tsex_temp.append(0)
        else:
            tsex_temp.append(1)
    test_x["Sex"] = tsex_temp

    embark_temp = []
    for data in train_x['Embarked']:
        if data == 'S':
            embark_temp.append(0)
        elif data == 'C':
            embark_temp.append(1)
        else:
            embark_temp.append(2)
    train_x["Embarked"] = embark_temp        

    tembark_temp = []    
    for data in test_x['Embarked']:
        if data == 'S':
            tembark_temp.append(0)
        elif data == 'C':
            tembark_temp.append(1)
        else:
            tembark_temp.append(2)
    test_x['Embarked'] = tembark_temp

    print(train_x.head(5))

    titan_tree = DecisionTreeClassifier(max_depth=6, random_state=13)
    titan_tree.fit(train_x, train_y)
    print("Score: ", titan_tree.score(train_x, train_y))
    y_predict = titan_tree.predict(test_x)
    score = 0
    for value, predict in zip(test_y, y_predict):
            if value == predict:
                score += 1
    print(score)