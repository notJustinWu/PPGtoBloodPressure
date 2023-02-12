from models.pytorch_nn import run_classification
from splitted_sbp_dbp_features import getFeatures_SBP_DBP, normalize, add_intercept
from feature_select import get_filtered_features_reg, get_filtered_features_pearsons, get_filtered_features_linear_svm, get_filtered_features_f_classif
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


data = getFeatures_SBP_DBP(all_ppg=True)
features = data["features"]
targets = data["targets"]
sbp = data["sbp"]
hypertensive = data["hypertensive"]

num_features = [28, 16, 10, 6, 3]
accuracies_train = []
accuracies_test = []
features = normalize(features)

for num_feature in num_features:
    n_features = get_filtered_features_f_classif(features, targets, max_features=num_feature)
    train_x, test_x, train_targets, test_targets = train_test_split(features, targets, train_size=0.7)

    train_x = normalize(train_x)
    test_x = normalize(test_x)

    y_predict_train, y_predict_test = run_classification(train_x, train_targets, test_x, test_targets, num_classes=5, num_epochs=1000, num_hidden=30, verbose=True)

    accuracy_train = 1 - (np.count_nonzero(np.abs(y_predict_train-train_targets))/train_targets.shape[0])
    accuracy_test = 1 - (np.count_nonzero(np.abs(y_predict_test-test_targets))/test_targets.shape[0])

    accuracies_train.append(accuracy_train)
    accuracies_test.append(accuracy_test)

    class_labels = ["Low", "Normal", "Elevated", "Hyp I", "Hyp II"]
    predicted_labels_train = []
    predicted_labels_test = []
    true_labels_train = []
    true_labels_test = []

    for i in range(train_targets.shape[0]):
        predicted_labels_train.append(class_labels[y_predict_train[i]])
        true_labels_train.append(class_labels[train_targets[i]])

    for i in range(test_targets.shape[0]):
        predicted_labels_test.append(class_labels[y_predict_test[i]])
        true_labels_test.append(class_labels[test_targets[i]])

    cf_train = confusion_matrix(true_labels_train, predicted_labels_train, labels=class_labels)
    cf_test = confusion_matrix(true_labels_test, predicted_labels_test, labels=class_labels)

    disp_train = ConfusionMatrixDisplay(cf_train, display_labels=class_labels)
    disp_test = ConfusionMatrixDisplay(cf_test, display_labels=class_labels)

    disp_train.plot()
    disp_test.plot()

    fig_train = disp_train.figure_
    fig_test = disp_test.figure_

    fig_train.savefig(f"nn_train_confusion_{num_feature}.png")
    fig_test.savefig(f"nn_test_confusion_{num_feature}.png")

figure2, axis2 = plt.subplots()


axis2.plot(np.array(num_features), np.array(accuracies_train), label="Train Accuracy")
axis2.plot(np.array(num_features), np.array(accuracies_test), label="Test Accuracy")

axis2.set_xlabel("# features")
axis2.set_ylabel("Accuracy")

axis2.set_xticks(np.array(num_features))

figure2.tight_layout()
figure2.legend()
figure2.savefig(f"nncl_accuracies.png")

np.savetxt("nncl_train_accuracies.csv", np.array(accuracies_train), delimiter=",")
np.savetxt("nncl_test_accuracies.csv", np.array(accuracies_test), delimiter=",")






