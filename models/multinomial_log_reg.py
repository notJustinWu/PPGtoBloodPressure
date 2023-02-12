from models.splitted_sbp_dbp_features_1 import getFeatures_SBP_DBP, normalize, add_intercept
from sklearn.model_selection import train_test_split

data = getFeatures_SBP_DBP()
features = data["features"]
targets = data['targets']
sbp = data["sbp"]
hypertensive = data["hypertensive"]
labels = data["one-hot-labels"]

train_x, test_x, train_targets, test_targets = train_test_split(features, labels, train_size=0.7)

train_x = normalize(train_x)
test_x = normalize(test_x)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, penalty='l2')
model.fit(train_x,train_targets)
target_pred = model.predict(test_x)

from sklearn import metrics
print("Model Accuracy (Multinomial Logistic Regression):",metrics.accuracy_score(test_targets,target_pred))
print(target_pred)
print(test_targets)

