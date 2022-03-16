import pandas as pd


train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")
data = pd.read_csv("titanic/gender_submission.csv")

print(train)
print(test)
print(data)