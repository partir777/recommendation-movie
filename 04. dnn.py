
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import f1_score 
import time
warnings.filterwarnings('ignore')

# normalize
def normalize_col(df,col_name):
    df[col_name] = (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
    return df
  
def getRecommended(user, model, movies, data, topN):
    user_id = user
    i_persona = data['user_id'] == user
    persona = data[i_persona]
    persona = persona.persona.unique()[0]
    i_user_profile = data['user_id'] == user
    user_profile = data[i_user_profile]
    user_profile = user_profile.user_profile.iloc[0]
    iw_movies = data['user_id'] == user
    w_movies = data[iw_movies]
    
    target_movies = copy.deepcopy(movies)
    for i in w_movies.movie_id:
        indexNames = target_movies[ target_movies['movie_id'] == i ].index
        target_movies.drop(indexNames , inplace=True)
    
    movie_size = len(target_movies)
    top_n = topN

    tmp_movie_data = target_movies.movie_id.values
    tmp_persona = np.array([persona for i in range(movie_size)])
    tmp_user_profile = np.array([user_profile for i in range(movie_size)])
    w_movies = w_movies.movie_id
    start = time.time()

    predictions = model.predict([tmp_movie_data, tmp_persona])
    predictions = np.array([p[0] for p in predictions])
    predictions = (-predictions).argsort()[:top_n] + 1
    print(time.time()-start)
    return predictions
  
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
  
_PERSONA_NUM_ = ['23']
_TOP_N_ = 50

for z in _PERSONA_NUM_:
    # data Load
    data = pd.read_csv('./files/mp_'+z+'.data', sep='|')
    #idx_data = data[data['rating'] < 3].index
    #data = data.drop(idx_data)
    #print(len(data))
    movies = pd.read_csv('./files/u.item', header=None, sep='|', encoding='latin-1' )
    movies.columns = ['movie_id', 'title', 'release_date', 'video_release_date',
                      'imdb_url', 'genre_unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    # example_age
    movies['example_age'] = np.round((pd.to_datetime("now") - pd.to_datetime(movies['release_date']))/np.timedelta64(1, 'Y'))
    movies['example_age'].fillna(0, inplace=True)
    movies['example_age'] = movies.example_age.astype('int64')
    
    user_df = pd.read_csv('./files/u.user', header=None, sep='|', engine='python')   
    user_df.columns = ['user_id','age','gender','job','time']
    user_df['gender_n'] = user_df['gender'].apply(lambda x:1 if x == 'M' else 2)
    profiles = ['age', 'gender_n']
    user_df['user_profile'] = [list(gs) for gs in zip(*[user_df[profile] for profile in profiles])]
    
    # Merge
    data = pd.merge(data, movies, how='left', on="movie_id")
    data = pd.merge(data, user_df, on='user_id')
    train, test = train_test_split(data, test_size = 0.3)
    number_of_unique_user = len(data.user_id.unique())
    number_of_unique_movie_id = len(data.movie_id.unique())
    number_of_unique_persona = len(data.persona.unique())
    number_of_unique_example_age = len(data.example_age.unique())
   
    
    # layer 쌓기
    movie_input = Input(shape=(1, ), name='movie_input_layer')
    user_input = Input(shape=(1, ), name='user_input_layer')
    persona_input = Input(shape=(1, ), name='persona_input_layer')
    nput_profile = Input(shape=(2, ), name='user_profile')

    movie_embedding_layer = Embedding(number_of_unique_movie_id + 1, 16, name='movie_embedding_layer')
    #user_embedding_layer = Embedding(number_of_unique_user + 1, 16, name='user_embedding_layer')
    persona_embedding_layer = Embedding(number_of_unique_persona + 1, 16, name='persona_embedding_layer')
  

    movie_vector_layer = Flatten(name='movie_vector_layer')
    #user_vector_layer = Flatten(name='user_vector_layer')
    persona_vector_layer = Flatten(name='persona_vector_layer')


    concate_layer = Concatenate()

    dense_layer1 = Dense(128, activation='relu')
    dense_layer2 = Dense(32, activation='relu')

    result_layer = Dense(1)
    
    # 쌓기
    movie_embedding = movie_embedding_layer(movie_input)
    #user_embedding = user_embedding_layer(user_input)
    persona_embedding = persona_embedding_layer(persona_input)


    movie_vector = movie_vector_layer(movie_embedding)
    # user_vector = user_vector_layer(user_embedding)
    persona_vector = persona_vector_layer(persona_embedding)
 
    
    # concat = concate_layer([movie_vector, persona_vector, input_profile])
    concat = concate_layer([movie_vector, persona_vector])
    dense1 = dense_layer1(concat)
    dense2 = dense_layer2(dense1)

    result = result_layer(dense2)

    #model = Model(inputs=[movie_input, persona_input, input_profile], outputs=result)
    model = Model(inputs=[movie_input, persona_input], outputs=result)
    model.summary()
    model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
  
    history = model.fit([
                         train.movie_id, 
                         train.persona
                         #,tf.keras.preprocessing.sequence.pad_sequences(train['user_profile'])
                        ], train.rating, epochs=50, verbose=1)
    # plt.plot(history.history['loss'])
    # plt.xlabel('epochs')
    # plt.ylabel('training error')
    model.evaluate([
                test.movie_id,
                test.persona
                #,tf.keras.preprocessing.sequence.pad_sequences(test['user_profile'])
           ], test.rating)
    dnn_persona = []
    '''
    for i in range(942):
        r = getRecommended((i+1),model, movies, data, _TOP_N_)
        user = []
        for j in range(len(r)):
            user.append(r[j])
        dnn_persona.append(user)
    # np.save('./files/dnn_v3_persona_top'+str(_TOP_N_)+'_rank', dnn_persona)
    # print("................ Persona "+z)
    '''
print("End!")

getRecommended(22,model, movies, data, 1000)

nmf_rank = np.load('./files/nmf_top50_rank.npy')
dnn_12 = np.load('./files/dnn_v3_persona_top12_rank.npy')
dnn_20 = np.load('./files/dnn_v3_persona_top20_rank.npy')
dnn_30 = np.load('./files/dnn_v3_persona_top30_rank.npy')
dnn_40 = np.load('./files/dnn_v3_persona_top40_rank.npy')
dnn_50 = np.load('./files/dnn_v3_persona_top50_rank.npy')

dnn_persona = [] 
dnn_persona.append(dnn_12)
dnn_persona.append(dnn_20)
dnn_persona.append(dnn_30)
dnn_persona.append(dnn_40)
dnn_persona.append(dnn_50)
hit_persona_dnn = totalHitRatio(nmf_rank, dnn_persona)
print(hit_persona_dnn)
hit_atk = hitAtK(nmf_rank, dnn_persona)
print(hit_atk)

# NMF, DNN 전체 사용자의 적중률 구하기
plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True
x_values = [23, 30, 40, 50]
fig = plt.figure()
ax2 = fig.add_subplot(212)
ax2.plot(x_values, hit_persona_dnn, 'r')
ax2.set_xlabel('number of persona')
ax2.set_ylabel('hit ratio (%)')
ax2.set_title('Total User DNN')
ax2.set_xticks([23, 30, 40, 50])
# 942 / 23 = 40
plt.show()
