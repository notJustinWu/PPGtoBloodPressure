from splitted_sbp_dbp_features import *
from sklearn.feature_selection import r_regression, mutual_info_regression, SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm



def get_filtered_features_pearsons(features, y, max_features, target):
    correlations = np.abs(r_regression(features, y))
    corr_min = np.min(np.sort(np.abs(correlations))[features.shape[1] - max_features:])
    isSelected = np.full(correlations.shape[0], True)
    for i in range(correlations.shape[0]):
        if correlations[i] < corr_min:
            isSelected[i] = False
    np.savetxt(f"selected_features_pearson_reg_{target}_{max_features}.csv", isSelected,delimiter=",")
    return features[:,isSelected]


def get_filtered_features_reg(features, y):
    selector = SelectFromModel(estimator=LinearRegression()).fit(features, y)
    coeffs = selector.estimator_.coef_
    selected = selector.get_support()
    return features[:,selected]


def get_filtered_features_linear_svm(features, y):
    lsvc = svm.LinearSVC(C=0.01, penalty="l1", dual=False)
    selector = SelectFromModel(lsvc.fit(features, y),prefit=True, max_features=10)
    selected = selector.get_support()
    return features[:,selected]


def get_filtered_features_f_classif(features, y, max_features):
    new_features =  SelectKBest(f_classif, k=max_features).fit(features, y)
    selected = new_features.get_support()
    np.savetxt(f"selected_features_ANOVA_class_{max_features}.csv", selected,delimiter=",")
    return new_features.transform(features)


def get_filtered_features_random_forest_cl(features, y, max_features):
    rfc = RandomForestClassifier(n_estimators = 100, random_state = 5)
    selector = SelectFromModel(rfc, threshold=-np.inf, max_features=max_features)
    selected = selector.fit(features, y).get_support()
    np.savetxt(f"selected_features_rf_class_{max_features}.csv", selected, delimiter=",")
    return features[:,selected]


def get_filtered_features_random_forest_reg(features, y, max_features, target):
    rfc = RandomForestRegressor(n_estimators = 100, random_state = 5)
    selector = SelectFromModel(rfc, threshold=-np.inf, max_features=max_features)
    selected = selector.fit(features, y).get_support()
    np.savetxt(f"selected_features_rf_reg_{target}_{max_features}.csv", selected, delimiter=",")
    return features[:,selected]