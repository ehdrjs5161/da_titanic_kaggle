import pandas as pd
import tensorflow as tf
from tensorflow import keras

def set_pandas_display():
    pd.set_option("display.max_rows", 100)  # 출력할 최대 행 갯수를 100으로 설정
    pd.set_option("display.max_columns", 500)  # 출력할 최대 열 갯수를 500개로 설정
    pd.set_option("display.width", 500)  # 글자 수 기준 출력할 넓이 설정
    pd.set_option("max_info_columns", 500)  # 열 기반 info가 주어질 경우, 최대 넓이

def modeling():
    input = keras.layers.Input(4)
    hidden1 = keras.layers.Dense(10)(input)
    hidden2 = keras.layers.Dense(10)(input)
    output = keras.layers.Dense(2, activation='sigmoid')(hidden2)

    model = keras.models.Model(input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    set_pandas_display()
    print(tf.__version__)

    train = pd.read_csv("titanic/train.csv")
    test = pd.read_csv("titanic/test.csv")
    result = pd.read_csv("titanic/gender_submission.csv")

    train_y = train.pop("Survived")
    train_x = train
    train.drop(['PassengerId', "Name", "Cabin"], axis='columns', inplace=True)
    train_x.dropna(inplace=True)
    print(train_x)


