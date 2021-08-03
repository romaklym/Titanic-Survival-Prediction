import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Reading csv files with train and test data
train_data = pd.read_csv("C:/Users/Roman Klym/Desktop/Titanic ML Prediction/train.csv")
test_data = pd.read_csv("C:/Users/Roman Klym/Desktop/Titanic ML Prediction/test.csv")

# Can see first 5 column of data in csv files
print(train_data.head())
print(test_data.head())

# Checking which percentage of women survived
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("{}% of women who survived".format(rate_women))

# Checking which percentage of men survived
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("{}% of men who survived".format(rate_men))


# Building a model
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# Outputing result into a csv file, where 1 is survived and 0 did not
output.to_csv('C:/Users/Roman Klym/Desktop/Titanic ML Prediction/result.csv', index=False)

# Reading results
result_data = pd.read_csv("C:/Users/Roman Klym/Desktop/Titanic ML Prediction/result.csv")
print(result_data.head(10))
