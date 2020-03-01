#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import jieba
import pickle
import gc 


def token(string):
    # we will learn the regular expression next course.
    return re.findall('\w+', string)

def cut(string): return list(jieba.cut(string))


# ### 汉语新闻

# In[2]:


#new text corpus
content=pd.read_csv("./data/sqlResult_1558435.csv",encoding='gb18030')


articles = content['content'].tolist()
titles = content['title'].tolist()


# In[6]:


articles_clean = [''.join(token(str(a)))for a in articles]
titles_1_clean=[''.join(token(str(a)))for a in titles]



TOKEN = []


# In[9]:


for i, line in enumerate(articles_clean):
    if i % 5000 == 0: print(i)
    TOKEN.append(cut(line))


# In[10]:


for i, line in enumerate(titles_1_clean):
    if i % 5000 == 0: print(i)
    TOKEN.append(cut(line))


# In[11]:


print("news data has done", len(TOKEN))
del articles_clean
del titles_1_clean
del content
gc.collect()

print("collect memory!")

# ### 维基百科

# In[12]:
print("reading wiki data")

w1=pd.read_json("./wikiextractor-master/extracted/merge_wiki.txt",lines=True)


w1_articles_clean = [''.join(token(str(a)))for a in w1["text"].tolist()]
W1_titles_1_clean=[''.join(token(str(a)))for a in w1["title"].tolist()]


# In[15]:


for i, line in enumerate(w1_articles_clean):
    if i % 5000 == 0: print(i)
    TOKEN.append(cut(line))


pickle.dump(TOKEN,open("token.pkl",'wb'))

