import pickle

from gensim.models import Word2Vec

print("Reading pickle file")
TOKEN=pickle.load(open("token.pkl","rb"))


model = Word2Vec(TOKEN, sg=1, 
                 size=128,  
                 window=5,  
                 min_count=2,  
                 negative=1, 
                 sample=0.001, 
                 hs=1, 
                 workers=7)


model.save('model') 
