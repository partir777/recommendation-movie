import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import time, math
from sklearn.preprocessing import scale
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
import _pickle as cPickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# NMF에 들어갈 Shape을 만들어 준다.
def GetShape(filename):
    df = pd.read_csv(filename, sep='|')
    n_users = len(df['user_id'].unique())
    n_items = len(df['movie_id'].unique())
    return (n_users, n_items)
  
# R= (X,y), iin sparse format
def ConvertToDense(X, y, shape):
    row = X[:, 0]
    col = X[:, 1]
    data = y
    matrix_sparse = sparse.csr_matrix((data, (row, col)), shape=(shape[0]+1, shape[1]+1))
    R = matrix_sparse.todense()
    R = R[1:, 1:]
    R = np.asarray(R)
    return R

  # 데이터를 로드한다.
def LoadData(filename, R_shape):
    df = pd.read_csv(filename, sep='|')
    X = df[['user_id', 'movie_id', 'rating']].values
    y = df['rating'].values
    # y = df['user_id'].values
    return X, y, ConvertToDense(X, y, R_shape)
  
def get_rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()     # Ignore nonzero terms
    actual = actual[actual.nonzero()].flatten() # Ignore nonzero terms
    return mean_squared_error(pred, actual)
  
def make_recommendation_activeuser(item_info, R, prediction, user_idx, k=5):

    rated_items_df_user = pd.DataFrame(R).iloc[user_idx, :]                 # get the list of actual ratings of user_idx (seen movies)
    user_prediction_df_user = pd.DataFrame(prediction).iloc[user_idx,:]     # get the list of predicted ratings of user_idx (unseen movies)
    reco_df = pd.concat([rated_items_df_user, user_prediction_df_user, item_info], axis=1)   # merge both lists with the movie's title
    reco_df.columns = ['rating', 'prediction','title']
    print('Preferred movies for user #', user_idx)
    print(reco_df.sort_values(by='rating', ascending=False)[:k])          # returns the 5 seen movies with the best actual ratings
    print('Recommended movies for user #', user_idx)
    reco_df = reco_df[ reco_df['rating'] == 0]
    print (reco_df.sort_values(by='prediction', ascending=False)[:k])        # returns the 5 unseen movies with the best predicted ratings
    # return np.sqrt(mean_squared_error(pred, actual)
    
  # Load the data set
R_shape = GetShape('./files/mp.data')
X, y, R = LoadData('./files/mp.data', R_shape)
item_info = pd.read_csv('./files/u.item', sep='|', header=None, usecols=[1], engine='python')   # Information about the item
item_info.columns = ['title']

# Choose a model: NMF
parametersNMF = {
    'n_components': 15,  # number of latent factors
    'init': 'random',
    'random_state': 0,
    'alpha': 0.001,  # regularization term
    'l1_ratio': 0,  # set regularization = L2
    'max_iter': 15
}

estimator = NMF(**parametersNMF)

err = 0
n_iter = 0.
n_folds = 5
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X[:, 0]):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Converting sparse array to dense array
    R_train = ConvertToDense(X_train, y_train, R_shape)
    R_test = ConvertToDense(X_test, y_test, R_shape)

    # Training (matrix factorization)
    t0 = time.time()
    estimator.fit(R_train)
    Theta = estimator.transform(R_train)  # user features
    M = estimator.components_.T  # item features
    n_iter += estimator.n_iter_
    
    # Making the predictions
    R_pred = M.dot(Theta.T)
    R_pred = R_pred.T

    # Computing the error on the validation set
    err = get_rmse(R_pred, R_test)
    print("*** RMSE Error : ", err / n_folds)
    print("Mean number of iterations:", n_iter / n_folds)

import time
start = time.time()
make_recommendation_activeuser(item_info, R, R_pred, user_idx=57, k=1000)
print(time.time()-start)
