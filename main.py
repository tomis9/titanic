import pandas as pd
from pathlib import Path

train_raw = pd.read_csv(Path("./data/train.csv"))

y_train = train_raw['Survived']

mean_age = train_raw['Age'].mean()

def process_data(data, mean_age=mean_age):
    data = data[["Pclass", "Sex", "Age", "SibSp", "Parch"]]
    data.columns = [col.lower() for col in data.columns]
    data.rename(
        {"pclass": "class", "sibsp": "sib"}, axis=1, inplace=True
    )
    data.loc[data['age'].isna(), "age"] = mean_age
    data['is_male'] = (data['sex'] == 'male').astype(int)
    data.drop('sex', axis=1, inplace=True)
    return data


X_train = process_data(train_raw)


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

test_raw = pd.read_csv(Path("./data/test.csv"))

test = process_data(test_raw)

dtc.predict(test)