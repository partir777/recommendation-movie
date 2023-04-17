
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
from matplotlib import pyplot as plt
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def hitRatio(t1, t2):
    cnt = 0
    for i in t1:
        for j in t2:
            if i == j:
                cnt = cnt + 1
    return int(cnt/len(t1) *100)
 
def totalHitRatio(_nmf_rank, _persona):
    hit_per = []
    for p in _persona:
        hit = 0
        for i in range(len(p)):
            hit = hit + hitRatio(p[i], _nmf_rank[i])
        hit_per.append(round(hit / len(p),2))
    return hit_per
  
  def hitYn(t1, t2):
    for i in t1:
        for j in t2:
            if i == j:
                return 1
    return 0
  
  def hitAtK(_nmf_rank, _persona):
    hit_per = []
    for p in _persona:
        hit = 0
        for i in range(len(p)):
            hit = hit + hitYn(p[i], _nmf_rank[i])
        hit_per.append(round(hit / len(p),4))
    return hit_per
  
  # NMF에 들어갈 Shape을 만들어 준다.
def GetShape(filename):
    df = pd.read_csv(filename, sep='|')
    n_users = len(df['persona'].unique())
    n_items = len(df['movie_id'].unique())
    return (n_users, n_items)
  
  # R= (X,y), iin sparse format
def ConvertToDense(X, y, shape):
    row = X[:, 2]
    col = X[:, 1]
    data = y
    matrix_sparse = sparse.csr_matrix((data, (row, col)))
    R = matrix_sparse.todense()
    R = R[0:, 0:]
    R = np.asarray(R)
    return R
  
  # 데이터를 로드한다.
def LoadData(filename, R_shape):
    df = pd.read_csv(filename, sep='|')
    
    X = df[['user_id', 'movie_id', 'persona', 'rating']].values
    y = df['rating'].values
    return X, y, ConvertToDense(X, y, R_shape)
  
  def get_rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()     # Ignore nonzero terms
    actual = actual[actual.nonzero()].flatten() # Ignore nonzero terms
    return np.sqrt(mean_squared_error(pred, actual))
  
  def make_recommendation_activeuser(item_info, R, prediction, user_idx, k=5):
    rated_items_df_user = pd.DataFrame(R).iloc[user_idx, :]                 # get the list of actual ratings of user_idx (seen movies)
    user_prediction_df_user = pd.DataFrame(prediction).iloc[user_idx,:]     # get the list of predicted ratings of user_idx (unseen movies)
    reco_df = pd.concat([rated_items_df_user, user_prediction_df_user, item_info], axis=1)   # merge both lists with the movie's title
    reco_df.columns = ['rating', 'prediction','title']
    #print('Preferred movies for user #', user_idx)
    #print(reco_df.sort_values(by='prediction', ascending=False)[:k])          # returns the 5 seen movies with the best actual ratings
    # print('Recommended movies for user #', user_idx)
    reco_df = reco_df[ reco_df['rating'] == 0]
    return reco_df.sort_values(by='prediction', ascending=False)[:k]
    # print (reco_df.sort_values(by='prediction', ascending=False)[:k])        # returns the 5 unseen movies with the best predicted ratings
    
  def make_recommendation_activeuser_persona(item_info, R, prediction, ignore_movies, persona, k=5):
    
    rated_items_df_user = pd.DataFrame(R).iloc[persona, :]                 
    user_prediction_df_user = pd.DataFrame(prediction).iloc[persona,:]     
    reco_df = pd.concat([rated_items_df_user, user_prediction_df_user, item_info], axis=1)   
    reco_df.columns = ['rating', 'prediction','title']
    reco_df = reco_df.drop(reco_df.index[ignore_movies.values])
    reco_df = reco_df[ reco_df['rating'] == 0]
    
    return reco_df.sort_values(by='prediction', ascending=False)[:k]
  
  
_PERSONA_NUM_ = ['23']
_TOP_N_ = 12

for i in _PERSONA_NUM_:   
    # Load the data set
    R_shape = GetShape('c')
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
        err += get_rmse(R_pred, R_test)
        print("*** RMSE Error : ", err / n_folds)
        print("Mean number of iterations:", n_iter / n_folds)
        
make_recommendation_activeuser(item_info, R, R_pred, 22, k=100)
    
nmf_rank = np.load('./files/nmf_top50_rank.npy')
persona23 = np.load('./files/nmf_top50persona_p_23_rank.npy')
#persona30 = np.load('./files/nmf_top'+str(_TOP_N_)+'persona_p_30_rank.npy')
#persona40 = np.load('./files/nmf_top'+str(_TOP_N_)+'persona_p_40_rank.npy')
#persona50 = np.load('./files/nmf_top'+str(_TOP_N_)+'persona_p_50_rank.npy')
persona = [] 
persona.append(persona23)
precision_atk = totalHitRatio(nmf_rank, persona)
hit_atk = hitAtK(nmf_rank, persona)
print(precision_atk)
print(hit_atk)
#persona.append(persona30)
#persona.append(persona40)

# NMF, DNN 전체 사용자의 적중률 구하기
plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True
x_values = [23, 30, 40, 50]
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(x_values, hit_persona_nmf, 'b')
ax1.set_xlabel('number of persona')
ax1.set_title('Total User NMF')
ax1.set_ylabel('hit ratio (%)')
ax1.set_xticks([23, 30, 40, 50])
# 942 / 23 = 40
plt.show()


#persona.append(persona50)
#hit_persona_nmf = totalHitRatio(nmf_rank, persona)

