import pickle
from collections import Counter


TOKEN=pickle.load(open("token.pkl","rb"))

all_words=[]
for aline in TOKEN:
    all_words+=aline

words_count=Counter(all_words)

num=0
for key, value in words_count.items():
    num+=value

d={}
for key, value in words_count.items():
    d[key]=value*1.0/num

pickle.dump(d,open("weight_data.pkl",'wb'))
