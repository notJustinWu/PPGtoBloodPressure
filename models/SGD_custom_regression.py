import numpy as np
from splitted_sbp_dbp_features import getFeatures_SBP_DBP
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def compute_cost(X, y, params):
    n_samples = len(y)
    h = X @ params
    return (1/(2*n_samples))*np.sum((h-y)**2)


def abs_diff_error(X, y, params):
    n_samples = len(y)
    h = X @ params
    return np.sum(abs(y-h))/n_samples



def gradient_descent(X, y, params, learning_rate, n_iters):
    n_samples = len(y)
    J_history = np.zeros((n_iters,1))
    print(params.shape)

    for i in range(n_iters):
        grad = X.T @ (X @ params - y)
        params = params - (learning_rate/n_samples) * grad
        J_history[i] = compute_cost(X, y, params)
        

    return (J_history, params)


data = getFeatures_SBP_DBP()
features = data["features"]
sbp = data["sbp"]

# Getting rid of some features by applying Lasso
# features_1 = features[:,1:2]
# features_2 = features[:,4:5]
# features = np.concatenate((features_1, features_2), axis=1)
###########

X, X_test, y, Y_test = train_test_split(features, sbp, train_size=0.7)
y = y.reshape((-1,1))
Y_test = Y_test.reshape((-1,1))

print(X.shape, y.shape)


n_samples = len(y)

mu = np.mean(X, 0)
sigma = np.std(X, 0)

X = (X-mu) / sigma

X = np.hstack((np.ones((n_samples,1)),X))
n_features = np.size(X,1)
print(n_features)
params = np.zeros((n_features,1))



n_iters = 1000
learning_rate = 0.1

initial_cost = compute_cost(X, y, params)

print("Initial cost is: ", initial_cost, "\n")

(J_history, optimal_params) = gradient_descent(X, y, params, learning_rate, n_iters)

print("Optimal parameters are: \n", optimal_params, "\n")
print(optimal_params.shape)

print("Final cost is: ", J_history[-1])

print("Final absolute avg diff for train data is", abs_diff_error(X, y, optimal_params))


mu = np.mean(X_test, 0)
sigma = np.std(X_test, 0)

X_test = (X_test-mu) / sigma

X_test = np.hstack((np.ones((X_test.shape[0],1)),X_test))

print("Final absolute avg diff for test data is", abs_diff_error(X_test, Y_test, optimal_params))

plt.scatter(X_test @ optimal_params, Y_test)
# plt.show()


# plt.plot(range(len(J_history)), J_history, 'r')

# plt.title("Convergence Graph of Cost Function")
# plt.xlabel("Number of Iterations")
# plt.ylabel("Cost")
# plt.show()