from sklearn.impute import SimpleImputer as SI
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import pandas as pd

def drop_col(df):
    col_with_missing_val = [col for col in df.columns if df[col].isnull().any()]
    print(col_with_missing_val)
    df.drop(col_with_missing_val, axis=1) 
    return df

def impute_col(df):
    impute = SI()
    imputed_df = pd.DataFrame(impute.fit_transform(df))
    imputed_df.columns = df.columns
    
    return imputed_df

def label_with_OE(df):
    object_col = [col for col in df.columns if df[col].dtype == "object"]
    ordinal_encoder = OrdinalEncoder()
    df[object_col] = ordinal_encoder.fit_transform(df[object_col])

    return df

def group_age(df):
    age_group = []
    for i in range(len(df)):
        age_group.append(int(df['Age'].iloc[i]/10))

    df['Age'] = age_group
    return df

def find_best_model(x_data, y_data):
    X_train, X_valid, y_train, y_valid = train_test_split(x_data, y_data,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)
    score = []
    for i in range(100, 1001, 100):
        model = RandomForestClassifier(n_estimators=i, random_state=0)
        model.fit(X_train, y_train)
        predicts = model.predict(X_valid)
        score.append((i, mean_absolute_error(y_valid, predicts)))

if __name__ == "__main__":
    X_train = pd.read_csv("titanic/train.csv")
    X_train.drop(["PassengerId", 'Name', "Cabin", 'Ticket'], axis=1, inplace=True)
    X_train.dropna(subset=['Embarked'], axis=0, inplace=True)

    categorical_col = [col for col in X_train.columns if X_train[col].dtype == 'object']
    print(categorical_col)

    label_OE_df = label_with_OE(X_train)
    imputed_X_train = impute_col(label_OE_df)
    imputed_X_train['Age'] = imputed_X_train['Age'].astype(int)

    final_X_train = group_age(imputed_X_train)    
    final_Y_train = final_X_train.pop("Survived")

    # find_best_model(final_X_train, final_Y_train)
    X_test = pd.read_csv("titanic/test.csv")
    Y_test = pd.read_csv("titanic/gender_submission.csv")

    test_data = X_test.set_index('PassengerId').join(Y_test.set_index("PassengerId"))
    test_data.drop(['Name', "Cabin", 'Ticket'], axis=1, inplace=True)

    label_test_data = label_with_OE(test_data)
    imputed_test_data = impute_col(test_data)
    final_X_test = group_age(imputed_test_data)
    final_Y_test = final_X_test.pop("Survived")

    best_model = RandomForestClassifier(n_estimators=100, random_state=0)
    best_model.fit(final_X_train, final_Y_train)
    predicts = best_model.predict(final_X_test)

    score = accuracy_score(final_Y_test, predicts)
    print(score)