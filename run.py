from pandas import read_csv
import numpy as np
from RegressionTree import DTree

def load_data(path_to_csv, has_header=True):
    """
    Loads a csv file of diamonds dataset
    Values can be both numerical and string (class type)
    returns: X - numpy array of size (n,m) of input features
             Y - numpy array of output features
    """
    if has_header:
        data = read_csv(path_to_csv, header='infer')
    else:
        data = read_csv(path_to_csv, header=None)
    Y = data["price"].as_matrix()
    data = data.drop('price', axis=1)
    mask = [np.issubdtype(i, np.number) for i in data.dtypes.tolist()][1:]
    data = data.as_matrix()
    X = data[:, 1:]
    return X, Y, mask

def train_test_split(X, Y, fraction):
    """
    perform the split of the data into training and testing sets
    input:
        X: numpy array of size (n,m)
        Y: numpy array of size (n,)
        fraction: number between 0 and 1, specifies the size of the training
                data

    returns:
        X_train
        Y_train
        X_test
        Y_test
    """
    if fraction < 0 or fraction > 1:
        raise Exception("Fraction for split is not valid")

    # do random sampling for splitting the data
    arr = np.column_stack([X, Y])
    perm = np.random.permutation(arr)
    r_fraction = int(len(Y) * fraction)
    return perm[:r_fraction, :-1], perm[:r_fraction, -1].astype(np.float64), perm[r_fraction:, :-1], perm[r_fraction:, -1].astype(np.float64)

def mse(y, b):
    return (y - b) ** 2


def mse_derivative(y, b):
    return y - b


X, Y, mask = load_data("data/diamonds.csv")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, .8)

d_tree = DTree(mask)
d_tree.fit(X_train, Y_train)
Y_pred = d_tree.predict(X_test)

print("Usual DTree  MSE: {0:.1f}, RMSE: {1:.1f}, MAE: {2:.1f}, Relative RMSE: {3:.3f}, Relative MAE: {4:.3f}".
          format(((Y_pred - Y_test) ** 2).mean(),
                 np.sqrt(((Y_pred - Y_test) ** 2).mean()), np.abs(Y_pred - Y_test).mean(),
                 np.sqrt(((Y_pred - Y_test) ** 2).mean()) / Y_train.mean(), np.abs(Y_pred - Y_test).mean() / Y_train.mean()))




learn_rate = 0.1
print("learn rate: ", learn_rate)
iterations = 100
u = np.zeros(len(Y_train))
ensemble = []
for i in range(iterations):
    # small tree, so we don't overfit
    tree = DTree(mask, max_depth=6, min_leaves=1)
    # our antigradient
    pr = learn_rate * mse_derivative(Y_train, u)
    tree.fit(X_train, pr)
    t = tree.predict(X_train)
    u += t
    ensemble.append(tree)
    Y_pred = np.sum([tree.predict(X_test) for tree in ensemble], axis=0)
    print("Iteration ", i)
    print("std: ", np.std(pr))
    print("Train set MSE: {0:.1f}, RMSE: {1:.1f}, MAE: {2:.1f}, Relative RMSE: {3:.3f}, Relative MAE: {4:.3f}".
          format(((Y_train - u) ** 2).mean(),
                 np.sqrt(((Y_train - u) ** 2).mean()), np.abs(Y_train - u).mean(),
                 np.sqrt(((Y_train - u) ** 2).mean()) / Y_train.mean(), np.abs(Y_train - u).mean() / Y_train.mean()))
    print("Test set MSE: {0:.1f}, RMSE: {1:.1f}, MAE: {2:.1f}, Relative RMSE: {3:.3f}, Relative MAE: {4:.3f}".
          format(((Y_pred - Y_test) ** 2).mean(),
                 np.sqrt(((Y_pred - Y_test) ** 2).mean()), np.abs(Y_pred - Y_test).mean(),
                 np.sqrt(((Y_pred - Y_test) ** 2).mean()) / Y_train.mean(), np.abs(Y_pred - Y_test).mean() / Y_train.mean()))
    
    
                                                                                                                      
    # print("Test set error: ", ((Y_pred - Y_test) ** 2).mean(), np.divide(np.abs(Y_pred - Y_test), Y_test).mean())

