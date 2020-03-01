# NLP-project1-摘要提取

## 1.下载语料库
语料1：https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/zhwiki/20200201/zhwiki-20200201-pages-articles-multistream.xml.bz2
语料2：https://github.com/Computing-Intelligence/datasource

## 2. 文件说明：

data_prepare.py 对预料库内的所有文本进行清理，并用jieba对其分词；

weight_calculation.py 计算清洗之后的词语在整个文本库内的权重

word2vect.py 训练模型得到每个单词的词向量

abstract_creation.py 计算每个句子的向量，以及整个文本的向量，找到与文本向量相似度最高的几个句子作为摘要。

