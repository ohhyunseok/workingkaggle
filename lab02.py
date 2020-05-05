import numpy as np
import pandas as pd

train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")


# Embarked

train['Embarked'].isnull().sum()
# 2

train['Embarked'].value_counts()
# S    644
# C    168
# Q     77


test['Embarked'].isnull().sum()
# 0

test['Embarked'].value_counts()
# S    270
# C    102
# Q     46

# null 모두 'S'
train['Embarked'].fillna('S', inplace=True)
# train['Embarked'].value_counts()
# S    644
# C    168
# Q     77


train['Title'] = train['Name'].str.extract(
    '([A-Za-z]+)\.', expand=False)

# print(train.head(5))

# print(pd.crosstab(train_data['Title'], train_data['Sex']))
