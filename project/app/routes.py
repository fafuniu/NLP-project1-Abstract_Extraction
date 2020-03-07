from app import app
from flask import render_template,request
from time import *
import pickle
from gensim.models import Word2Vec
import jieba
import scipy.spatial.distance as distance
import re
import numpy as np
import heapq
from sklearn.decomposition import TruncatedSVD


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

print("Loading model and data")
model = Word2Vec.load('../models/model')

weight_dic=pickle.load(open("../data/weight_data.pkl",'rb'))

print("Starting")



@app.route("/")
@app.route('/index')
def index():
     return render_template('master.html')

@app.route('/abs')
def abs():
    query = request.args.get('query', '') 
    #print("thst")
    news=query

    sentence_raw=cut_sent(news)

    sentence_clean=[]
    for asent in sentence_raw:
        if len(asent)>2:
            sentence_clean.append(asent.strip())

    sentence_ready = [''.join(token(str(a)))for a in sentence_clean]

    Token=[]
    for sent in sentence_ready:
        Token.append(cut(sent))

    #生成句子向量

    x_mask=prepare_data(Token)
    w = seq2weight(Token, x_mask, weight_dic) #weight matrix
    emb = get_weighted_average(model, Token, w)


    # 生成文章向量

    article_ready = [''.join(token(str(news)))]

    article_Token=[]
    for sent in article_ready:
        article_Token.append(cut(sent))

    article_x_mask=prepare_data(article_Token)
    article_w = seq2weight(article_Token, article_x_mask, weight_dic) #weight matrix
    article_emb = get_weighted_average(model, article_Token, article_w)


    # 合并文章向量和句子向量，找到相似度最大的几个句子

    new_emb=np.concatenate((article_emb,emb),axis=0)

    new_emb[np.isnan(new_emb)]=0

    new_emb=remove_pc(new_emb,1)

    title_or_not=0
    simlarity=[]
    for i in range(1+title_or_not,new_emb.shape[0]):
        c = 1 - distance.cosine(new_emb[0,:], new_emb[i,:])
        simlarity.append(c)

    print(simlarity)
    max_index = map(simlarity.index, heapq.nlargest(5, simlarity)) 
    abstract=[]
    for i in list(set(max_index)):
        abstract.append(sentence_clean[i])


    # use model to predict classification for query
    #classification_labels = model.predict([query])[0]
    #classification_results = dict(zip(df.columns[4:], classification_labels))
    abs_results=" ".join(abstract)

    # This will render the go.html Please see that file. 
    return render_template(
        'abs.html',
        query=query,
        result=abs_results
    )
