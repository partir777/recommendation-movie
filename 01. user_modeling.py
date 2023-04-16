import pandas as pd
import numpy as np
from gensim import corpora
import gensim
import pickle
import pyLDAvis.gensim
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Coherence values 계산
def compute_coherence_values(dictionary, corpus, text, limit, start=2, step=3): 
    coherence_values = [] 
    model_list = [] 
    for num_topics in range(start, limit, step): 
        print(num_topics)
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=30)
        coherence_model_lda = gensim.models.CoherenceModel(model=model, texts=text, dictionary=dictionary, coherence='c_v')
        model_list.append(model) 
        coherence_values.append(coherence_model_lda.get_coherence())   
    return model_list, coherence_values 
  
# Coherence score 별 토픽 갯수 확인
def find_optimal_number_of_topics(dictionary, corpus, text): 
    limit = 100; 
    start = 10; 
    step = 10; 
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, text=text, start=start, limit=limit, step=step) 
    x = range(start, limit, step) 
    plt.plot(x, coherence_values) 
    plt.xlabel("Num Topics") 
    plt.ylabel("Coherence score") 
    plt.legend(("coherence_values"), loc='best') 
    plt.show()
    
# 가장 성능이 좋은 모델 선택
def selecte_model(model_list, coherence_values):
    tmp_c = 0
    index = 0
    for i in range(len(coherence_values)):
        print(coherence_values[i])
        if tmp_c < coherence_values[i]:
            tmp_c = coherence_values[i]
            index = i
    return model_list[index]

# 사용자의 영화 평점 정보 불러오기
movie_data = pd.read_csv('./files/u.data', sep='|')
movie_data.columns = ['user_id', 'movie_id', 'rating', 'time']
sMovie_data = movie_data.sort_values(by='user_id')
nMovie_data = np.array(sMovie_data)

# 토픽(사용자)별 단어 집합 만들기
# 사용자별로 movie * rating으로 영화의 빈도를 조정한다.
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
    for j in range(rt):
        topic.append(str(nMovie_data[i][1])+'movie')
        
# 단어 사전 만들기
dictionary = corpora.Dictionary(topics_c)
corpus = [dictionary.doc2bow(text) for text in topics_c]
dictionary.save_as_text('dictionary')
pickle.dump(corpus, open("corpus", "wb"))

# 검사하고자 할 토픽의 범위 지정
limit = 100; 
start = 10; 
step = 3; 
find_optimal_number_of_topics(dictionary=dictionary, corpus=corpus, text=topics_c)
# model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, text=topics_c, start=start, limit=limit, step=step) 
#가장 좋은 모델 선택
#ldamodel = selecte_model(model_list, coherence_values)


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 60, id2word=dictionary, passes=15)

# 토픽 확인
topics = ldamodel.print_topics(60, num_words=5)
for topic in topics:
    print(topic)
    
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis)


#4) 문서 별 토픽 분포 보기

for i, topic_list in enumerate(ldamodel[corpus]):
    if i==5:
        break
    print(i,'번째 문서의 topic 비율은',topic_list)

#def make_topictable_per_doc(ldamodel, corpus, texts):
topic_table = pd.DataFrame()
texts = topics

#print(corpus[:2]) #[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 2), (22, 2), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 4), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1), (35, 1), (36, 1), (37, 1), (38, 1), (39, 1), (40, 1), (41, 1), (42, 2), (43, 1), (44, 1), (45, 1), (46, 1), (47, 1), (48, 1), (49, 1), (50, 1), (51, 1), (52, 1), (53, 1), (54, 1)], [(52, 1), (55, 1), (56, 1), (57, 1), (58, 1), (59, 1), (60, 1), (61, 1), (62, 1), (63, 1), (64, 1), (65, 1), (66, 2), (67, 1), (68, 1), (69, 1), (70, 1), (71, 2), (72, 1), (73, 1), (74, 1), (75, 1), (76, 1), (77, 1), (78, 2), (79, 1), (80, 1), (81, 1), (82, 1), (83, 1), (84, 1), (85, 2), (86, 1), (87, 1), (88, 1), (89, 1)]]
#print(ldamodel[corpus][:2]) #<gensim.interfaces.TransformedCorpus object at 0x7fed1abf9898>
#print(len(ldamodel[corpus])) #11314

# 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
for i, topic_list in enumerate(ldamodel[corpus]):
    print(topic_list) #[(0, 0.49480823), (3, 0.042265743), (16, 0.3513999), (17, 0.0986229)]
    doc = topic_list[0] if ldamodel.per_word_topics else topic_list 
    print(doc) #[(0, 0.49480823), (3, 0.042265743), (16, 0.3513999), (17, 0.0986229)] 
    # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
    # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
    # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
    # 48 > 25 > 21 > 5 순으로 정렬이 된 것.
    doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
    print(doc) #[(0, 0.49480823), (16, 0.3513999), (17, 0.0986229), (3, 0.042265743)]

    # 모든 문서에 대해서 각각 아래를 수행
    for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
        if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
            wp = ldamodel.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])         
            print(int(topic_num), round(prop_topic, 4), topic_list, topic_keywords) #15 0.3419 [(5, 0.14677979), (11, 0.1351353), (15, 0.34185725), (16, 0.2243457), (17, 0.1397852)] people, christian, many, church, also, christians, religion, world, believe, would
            topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list, topic_keywords]), ignore_index=True)
            # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
        else:
            break
#return(topic_table)

#topictable = make_topictable_per_doc(ldamodel, corpus, tokenized_doc)
topic_table = topic_table.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topic_table.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중', 'topic_keywords']
topic_table[:10]


ldamodel.save('topic_model_60')

# 사용자 별 토픽 구하기 - 업그레이 int-hot vector
# 사용자 별 페르소나 메트릭스 생성
user_topics = np.array([])

for c in corpus:
    user_topic = np.zeros(shape=(60,), dtype=np.int8)
    t = ldamodel.get_document_topics(c)
    
    for i in range(len(t)):
        idx = t[i][0]
        user_topic[idx] = round(t[i][1] * 10)
    user_topics = np.append(user_topics, user_topic)
user_topics = user_topics.reshape(943, 60)
print(user_topics[99])

# k-means clustering로 사용자 성향별 군집화 만들기 
from sklearn.cluster import KMeans
#클러스터의 개수 지정(n개)
num_clusters = 30
#알맞은 매트릭스 Z 삽입
km = KMeans(n_clusters=num_clusters).fit(user_topics)

# 최적의 K 구하기
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
distortions = []
K = range(1,100)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(user_topics)
    kmeanModel.fit(user_topics)
    distortions.append(sum(np.min(cdist(user_topics, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / user_topics.shape[0])
#Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

y_kmeans = km.fit_predict(user_topics)
X = user_topics

# 사용자별 persona를 만들어보장
cols = ['topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6', 'topic7', 'topic8','topic9', 'topic10',
        'topic11', 'topic12', 'topic13', 'topic14', 'topic15', 'topic16', 'topic17', 'topic18','topic19', 'topic20',
       'topic21', 'topic22', 'topic23', 'topic24', 'topic25', 'topic26', 'topic27', 'topic28','topic29', 'topic30',
        'topic31', 'topic32', 'topic33', 'topic34', 'topic35', 'topic36', 'topic37', 'topic38','topic39', 'topic40',
       'topic41', 'topic42', 'topic43', 'topic44', 'topic45', 'topic46', 'topic47', 'topic48','topic49', 'topic50',
        'topic51', 'topic52', 'topic53', 'topic54', 'topic55', 'topic56', 'topic57', 'topic58','topic59', 'topic60']
user_persona = pd.DataFrame(user_topics, columns=cols)
user_persona['persona'] = km.labels_
user_persona = user_persona.astype(int)
user_persona = user_persona.reset_index().rename(columns={'index': 'user'})

aa = user_persona['user'] + 1
user_persona['user_id'] = aa
user_persona = user_persona.drop('user', axis=1)

# 기존에 있는 u.data와 사용자 persona를 연결해준다.
user_profiles = pd.merge(movie_data, user_persona, how='left', on='user_id')

# 데이터 저장
user_profiles.to_csv("./files/mp_30.data", sep='|', index=False)

# 데이터 저장
user_persona.to_csv("./files/up_30.data", sep='|', index=False)

user_profiles = user_profiles.drop(['time','topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9','topic10','topic11','topic12','topic13','topic14','topic15','topic16','topic17','topic18','topic19','topic20'], axis=1)

pickle.dump(km, open("user_persona_30", "wb"))

