import numpy as np 
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")
subm_data = pd.read_csv("./submission.csv")

y_train = train_data["Survived"]
y_test = subm_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
x_train_oh = pd.get_dummies(train_data[features])
x_test_oh = pd.get_dummies(test_data[features])


model = RandomForestClassifier(n_estimators = 1000, max_depth=5, random_state=1)
model.fit(x_train_oh, y_train)
result = model.score(x_test_oh, y_test)
print(result)
predictions = model.predict(x_test_oh)


# mlp = MLPClassifier(hidden_layer_sizes=(20,), activation='relu', solver='sgd', 
# alpha=0.001, batch_size=64, learning_rate_init=0.01, max_iter=500)
# mlp.fit(x_train_oh, y_train)
# result = mlp.score(x_test_oh, y_test)
# print("loss : ", mlp.loss_)
# print("score : ", result)
# predictions = mlp.predict(X_test)



# sgd = SGDClassifier(loss='log',penalty='l1', alpha=0.01, random_state=42)
# #pipe = make_pipeline(StandardScaler(), sgd)
# #scores = cross_validate(pipe, x_train_oh, y_train, return_train_score=True)
# sgd.fit(x_train_oh, y_train)
# result = sgd.score(x_test_oh, y_test)
# #result = (np.mean(scores['train_score']))
# #predictions = sgd.predict(x_test_oh)
# print(result)


# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('my_submission_rf_04.csv', index=False)
# print('success~!')

