
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from gensim import corpora
import gensim
import pickle
from sklearn.cluster import KMeans

_MODEL_FILE_PATH_ = './models/dnn_v2/'
_MODEL_FILE_TOPIC_ = './models/topic_model'
_MODEL_FILE_KM_ = './models/user_persona'
_MODEL_FILE_CORPUS_ = './models/corpus'
_PERSONA_FILE_PATH_ = './files/mp.data'
_USERS_FILE_PATH_ = './files/up.data'
_MOVIE_FILE_PATH_ = './files/u.item'
_DICTIONARY_FILE_PATH = './models/dictionary'
_TEST_USER_ID_ = 100
_TOP_N_ = 5

def load():
    persona = pd.read_csv(_PERSONA_FILE_PATH_, sep='|', engine='python')
    movies = pd.read_csv(_MOVIE_FILE_PATH_, sep='|', engine='python')
    movies.columns = ['movie_id', 'title', 'release_date', 'video_release_date',
                  'imdb_url', 'genre_unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    topic = gensim.models.ldamodel.LdaModel.load(_MODEL_FILE_TOPIC_)
    dnn = tf.keras.models.load_model(_MODEL_FILE_PATH_)
    km = pickle.load(open(_MODEL_FILE_KM_, "rb"))
    dictionary = corpora.Dictionary.load_from_text(_DICTIONARY_FILE_PATH)
    corpus = pickle.load(open(_MODEL_FILE_CORPUS_, "rb"))
    users = pd.read_csv(_USERS_FILE_PATH_, sep='|', engine='python')
    return persona, movies, users, topic, dnn, km, dictionary, corpus
  
  
def getWatchMovies(data, userId):
    idx = data['user_id'] == userId
    return data[idx]['movie_id']
  
  
def getUserTopic(corpus, topic_model):
    user_topic = np.zeros(shape=(20,), dtype=np.int8)
    t = topic_model.get_document_topics(corpus)
    for i in range(len(t)):
        idx = t[i][0]
        user_topic[idx] = round(t[i][1] * 10)
    return user_topic
  
  
def genreCount(_movie_df):
    
    genre = {'count':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] }
    genre_list = [ 'genre_unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    genre_df = pd.DataFrame(genre, index=genre_list)

    for i in genre_list:
        genre_df.loc[i] = _movie_df[i].sum()
    genre_df = genre_df.sort_values(by=['count'] ,ascending=False)

    return genre_df
  
  
def getRecommendation(_user_id, _movie_df, df, _topN, model):
    
    movie_ids = df["movie_id"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    
    # Let us get a user and see the top recommendations.
    user_id = _user_id
    movies_watched_by_user = df[df.user_id == user_id]
    all_movies = _movie_df["movie_id"]
    all_movies = list(
        set(all_movies).intersection(set(movie2movie_encoded.keys()))
    )
    all_movies= [[movie2movie_encoded.get(x)] for x in all_movies]
    movies_not_watched = _movie_df[
        ~_movie_df["movie_id"].isin(movies_watched_by_user.movie_id.values)
    ]["movie_id"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    
    idx_up = df["user_id"] == user_id
    up = df[idx_up]
    ps = up.persona.unique()
    persona = up.persona.unique()[len(ps)-1]
    user_encoder = persona
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )
    '''
    user_movie_array = np.hstack(
        ([[persona]] * len(movies_not_watched), movies_not_watched)
    )
    '''
    dnn = tf.keras.models.load_model(_MODEL_FILE_PATH_)
    # ratings = model.predict(user_movie_array).flatten()
    ratings = dnn.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-_topN:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    recommended_movies = _movie_df[_movie_df["movie_id"].isin(recommended_movie_ids)]
    
    return recommended_movie_ids
  
def printMovie(movies, movie_ids):
    for i in movie_ids:
        print(movies[movies['movie_id'] == i])
        
 def makeUserCorpus(userMovie, dictionary):
    nMovie_data = np.array(userMovie)
    c = nMovie_data[0][0]
    topic = []
    topics_c = []
    for i in range(len(nMovie_data)):
        rc = nMovie_data[i][0]
        if c != rc :
            c = rc
            topics_c.append(topic)
            topic = []
        if i == len(nMovie_data)-1:
            topics_c.append(topic)
        rt = nMovie_data[i][2]
        for j in range(int(rt)):
            topic.append(str(nMovie_data[i][1])+'movie')
    corpus = [dictionary.doc2bow(text) for text in topics_c]
    return corpus
  
  
if __name__ == "__main__":
    # 1. 서비스에 사용하는 데이터 모델 로드
    persona_data, movie_data, users, topic_model, dnn_model, km_model, dictionary, corpus = load()
    # 2. 사용자의 프로파일 생성
    user = users[users['user_id'] == _TEST_USER_ID_]
    user_persona = user['persona'].values[0]
    w = getWatchMovies(persona_data, _TEST_USER_ID_)
    w = pd.merge(w, movie_data, on='movie_id')
    
    #print('영화 개수 : ',len(w), '사용자 페르소나 : ', user_persona, '장르 정보',genreCount(w) )
    
    # 3. 토픽 및 페르소나 확인
    # user_topics = getUserTopic(corpus[_TEST_USER_ID_-1], topic_model)
    # x = km_model.predict([user_topics])
    for i in range(10):    

        # 4. 본영화 리스트
        #watch_movice = getWatchMovies(persona_data, _TEST_USER_ID_)
        
        # 5. 영화 추천
        p_movice = (getRecommendation(_TEST_USER_ID_, movie_data, persona_data, _TOP_N_, dnn_model))

        # 6. 추천 받은 영화 데이터 추가
        for j in p_movice:
            pt = pd.DataFrame({'user_id':_TEST_USER_ID_, 
                                                'movie_id':j, 
                                                'rating':4, 
                                                'time':0,  
                                                'topic1':user['topic1'],  'topic2':user['topic2'],  'topic3':user['topic3'], 'topic4':user['topic4'], 
                                                'topic5':user['topic5'],  'topic6':user['topic6'],  'topic7':user['topic7'], 'topic8':user['topic8'], 
                                                'topic9':user['topic9'],  'topic10':user['topic10'],  'topic11':user['topic11'], 'topic12':user['topic12'], 
                                                'topic13':user['topic13'],  'topic14':user['topic14'],  'topic15':user['topic15'], 'topic16':user['topic16'], 
                                                'topic17':user['topic17'],  'topic18':user['topic18'],  'topic19':user['topic19'], 'topic20':user['topic20'], 
                                                'persona': user_persona})
            persona_data = pd.concat([persona_data, pt]) 
            

        # 7. 사용자 코퍼스 변경
        t_movies = persona_data[persona_data['user_id'] == _TEST_USER_ID_]
    
        new_corpus = makeUserCorpus(t_movies, dictionary)

        # 8. 토픽 및 페르소나 확인
        user_topics = getUserTopic(new_corpus[0], topic_model)
        user_persona = km_model.predict([user_topics])[0]
        
        w = getWatchMovies(persona_data, _TEST_USER_ID_)
        w = pd.merge(w, movie_data, on='movie_id')
            
        # 9. 성향 별 페르소나 출력 
        print('[',str(i),']','영화 개수 : ',len(t_movies), '사용자 페르소나 : ', user_persona,'토픽 : ', user_topics,'장르 정보', genreCount(w))
        
