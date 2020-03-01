#!/usr/bin/env python
# coding: utf-8

# In[1]:
from time import *
import pickle
from gensim.models import Word2Vec
import jieba
import scipy.spatial.distance as distance
import re
import numpy as np
import heapq
from sklearn.decomposition import TruncatedSVD


# In[8]:


def token(string):
    # we will learn the regular expression next course.
    return re.findall('\w+', string)

def cut(string): return list(jieba.cut(string))


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")



def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    #x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        #x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x_mask

def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(mask.shape).astype('float32')
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] > 0:
                try:
                    weight[i,j] = weight4ind[seq[i][j]]
                except:
                    weight[i,j] = 0
    weight = np.asarray(weight, dtype='float32')
    return weight

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX




def get_weighted_average(model,Token, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_sentence = len(Token)
    emb = np.zeros((n_sentence, 128))
    for i in range(n_sentence):
        #w.shape[1]最长的单词数目
        words_matrix=np.zeros((w.shape[1], 128)).astype('float64')
        for j in range(w.shape[1]):
            if w[i,j]>0:
                try:
                    words_matrix[j,:]=model.wv[Token[i][j]]
                except:
                    pass
        emb[i,:] =  w[i,:].dot(words_matrix) / np.count_nonzero(w[i,:])
        
        if emb[i,:].sum()==0:
            print(Token[i])
    return emb


# In[100]:


news='''据韩联社消息，韩国首尔3月1日下午以杀人罪、伤害罪、违反传染病预防管理相关法律为由，向首尔中央地方监察厅起诉“新天地教会”教主李万熙，以及教会十二支派的支派长. 首尔方面表示，李万熙等人拒绝病毒检测，并在给政府提交的教徒名单中故意遗漏和隐瞒，妨碍防疫部门工作。这在刑法上属于杀人罪及伤害罪，并涉嫌违反传染病预防法。

　　目前李万熙在其靠近首尔、位于京畿道的家中进行自我隔离。“新天地”相关人士称李万熙已接受检测，而首尔尚未正式确认这一消息。

　　现年89岁的李万熙于1984年创建“新天地教会”，他自称“耶稣转世再临之主”，一贯宣扬只有信奉“新天地教”才能实现“末日拯救”。“新天地”是在基督教界被普遍认为属于邪教，我国多地政府主管部门都已依法取缔“新天地”教会。

　　有韩国媒体今天报道，一些韩国“新天地”教会的信徒曾在2020年1月去过中国武汉，但没有透露此行包含了多少人。韩国司法部则在2月29日称，据韩方追踪和推测，在过去8个月里，有至少42名来自武汉的“新天地”教派信徒入境韩国。

　　长安街知事注意到，今天上午，首尔市长朴元淳在社交媒体要求李万熙道歉，并表示教会方面如不采取必要措施，将以过失杀人罪等罪名提起诉讼。

　　朴元淳还喊话检察总长尹锡悦称，逮捕李万熙是目前检方应该发挥的作用。

　　近日，韩国检方已对李万熙正式展开调查，水原检察厅将此案分配至刑事六部展开调查，理由是在新冠肺炎确诊者大幅度增加的情况下，李万熙却阻碍保健当局的防疫活动。

　　韩国疫情持续恶化，与“新天地教会”大邱教会暴发的“超级感染事件”直接相关。韩国卫生当局认为，超过一半的病例都与“新天地”有关。 

　　触发“超级感染事件”的，是韩国第31号病例、一名61岁的“新天地”教会女教徒，她在出现症状后仍数次参加教会活动。此后，大邱确诊病例激增，超过70%的患者为“新天地”大邱教会信徒。
'''


# In[3]:

print("Loading model and data")
model = Word2Vec.load('model')

weight_dic=pickle.load(open("./weight_data.pkl",'rb'))

print("Starting")
begin_time=time()
# In[101]:


#数据清洗

sentence_raw=cut_sent(news)

sentence_clean=[]
for asent in sentence_raw:
    if len(asent)>0:
        sentence_clean.append(asent)

sentence_ready = [''.join(token(str(a)))for a in sentence_clean]

Token=[]
for sent in sentence_ready:
    Token.append(cut(sent))


# In[106]:


#生成句子向量

x_mask=prepare_data(Token)
w = seq2weight(Token, x_mask, weight_dic) #weight matrix
emb = get_weighted_average(model, Token, w)


# In[107]:


# 生成文章向量

article_ready = [''.join(token(str(news)))]

article_Token=[]
for sent in article_ready:
    article_Token.append(cut(sent))

article_x_mask=prepare_data(article_Token)
article_w = seq2weight(article_Token, article_x_mask, weight_dic) #weight matrix
article_emb = get_weighted_average(model, article_Token, article_w)


# In[110]:


# 合并文章向量和句子向量，找到相似度最大的几个句子

new_emb=np.concatenate((article_emb,emb),axis=0)

new_emb[np.isnan(new_emb)]=0

new_emb=remove_pc(new_emb,1)

title_or_not=0
simlarity=[]
for i in range(1+title_or_not,new_emb.shape[0]):
    c = 1 - distance.cosine(new_emb[0,:], new_emb[i,:])
    simlarity.append(c)

max_index = map(simlarity.index, heapq.nlargest(5, simlarity)) 
abstract=[]
for i in list(set(max_index)):
    abstract.append(sentence_clean[i])


# In[120]:

end_time=time()
print("The toal runing time is ",end_time-begin_time)
print("".join(abstract))


# In[ ]:




