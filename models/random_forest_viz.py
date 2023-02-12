from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from splitted_sbp_dbp_features import *
from feature_select import *
import pydot

# Load the data
data = getFeatures_SBP_DBP(all_ppg=True)
features = data["features"]
sbp = data["sbp"]
dbp = data["dbp"]

features = get_filtered_features_random_forest_reg(features, sbp)

X, y = features, sbp

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest regressor
rf = RandomForestRegressor(n_estimators=1, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Get the first tree from the random forest
tree = rf.estimators_[0]

# Export the tree as a .dot file
export_graphviz(tree, out_file="tree.dot",
                feature_names=range(1, X.shape[1] + 1),
                rounded=True,
                filled=True)

# Convert the .dot file to a .png image
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

# Display the image
Image(filename='tree.png')
