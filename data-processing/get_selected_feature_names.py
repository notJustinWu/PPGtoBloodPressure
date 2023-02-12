import numpy as np

feature_names = ["IH", "IL", "ai", "lasi", "S01", "S02", "S03", "S04", "d10", "d10+s10", "d10/s10", "d25", "d25+s25", "d25/s25", "d33", "d33+s33", "d33/s33", "d50", "d50+s50", "d50/s50", "d66", "d66+s66", "d66/s66", "d75", "d75+s75", "d75/s75", "st", "dt"]

for i in range(28):
    name = "a"+str(i)
    feature_names.append(name)


def get_selected_feature_names(bool_arr_path):
    selected_feature_names = []
    bool_arr = np.loadtxt(bool_arr_path, delimiter=",", dtype=np.double)
    for i in range(len(bool_arr)):
        isSelected = bool_arr[i]
        feature_name = feature_names[i]
        if isSelected:
            selected_feature_names.append(feature_name)

    return selected_feature_names

print("============================")

print(get_selected_feature_names("selected_features_pearson_reg_sbp_3.csv"))
print(get_selected_feature_names("selected_features_pearson_reg_dbp_3.csv"))
print(get_selected_feature_names("selected_features_rf_reg_sbp_3.csv"))
print(get_selected_feature_names("selected_features_rf_reg_dbp_3.csv"))
print(get_selected_feature_names("selected_features_ANOVA_class_3.csv"))
print(get_selected_feature_names("selected_features_rf_class_3.csv"))

print("============================")

print(get_selected_feature_names("selected_features_pearson_reg_sbp_6.csv"))
print(get_selected_feature_names("selected_features_pearson_reg_dbp_6.csv"))
print(get_selected_feature_names("selected_features_rf_reg_sbp_6.csv"))
print(get_selected_feature_names("selected_features_rf_reg_dbp_6.csv"))
print(get_selected_feature_names("selected_features_ANOVA_class_6.csv"))
print(get_selected_feature_names("selected_features_rf_class_6.csv"))

print("============================")

print(get_selected_feature_names("selected_features_pearson_reg_sbp_10.csv"))
print(get_selected_feature_names("selected_features_pearson_reg_dbp_10.csv"))
print(get_selected_feature_names("selected_features_rf_reg_sbp_10.csv"))
print(get_selected_feature_names("selected_features_rf_reg_dbp_10.csv"))
print(get_selected_feature_names("selected_features_ANOVA_class_10.csv"))
print(get_selected_feature_names("selected_features_rf_class_10.csv"))

print("============================")

print(get_selected_feature_names("selected_features_pearson_reg_sbp_16.csv"))
print(get_selected_feature_names("selected_features_pearson_reg_dbp_16.csv"))
print(get_selected_feature_names("selected_features_rf_reg_sbp_16.csv"))
print(get_selected_feature_names("selected_features_rf_reg_dbp_16.csv"))
print(get_selected_feature_names("selected_features_ANOVA_class_16.csv"))
print(get_selected_feature_names("selected_features_rf_class_16.csv"))

print("============================")
