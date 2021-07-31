# -*- coding: UTF-8 -*-
__author__ = 'ZhengXiang'
__time__ = '2019/10/15 16:56'

import jieba
# jieba.load_userdict("F:/Python/weibo/201906/userdict.txt")
from gensim.models import word2vec
import pandas as pd
import re

# 读取文件
# d = pd.read_csv('F:/Python/weibo/201906/08/weibo0812_clean.csv', usecols=['co_clean2'])
# re_clean = d['co_clean2'].values.tolist()
# print(type(re_clean))
# print(re_clean)
path="1.txt"


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# def movestopwords(sentence):
#     stopwords = stopwordslist('F:/Python/weibo/201906/08/stopwords.txt')  # 这里加载停用词的路径
#     santi_words =[x for x in sentence if len(x) >1 and x not in stopwords]
#     #santi_words= re.sub('[（）：:？“”《》，。！·、\d ]+', ' ', santi_words)
#     return santi_words

#------------------------------------------

# seg_lists=[]
# with open(path, encoding="utf-8") as f:
#     for i in f.readlines():
#         #i = jieba.cut(i)  # 默认是精确模式
#         seg_list = (jieba.lcut(i, cut_all=False))
#         # 去停用词
#         # tmp=movestopwords(seg_list)
#         # print(tmp)
#         tmp=' '.join(seg_list)
#         # print(tmp)
#         seg_lists.append(tmp)
#     print(seg_lists)
#
# # 分完词后保存到新的txt中
# with open('fenci_0225.txt','w',encoding='utf-8') as f:
#     for i in seg_lists:
#         if i =='':
#             pass
#         else:
#             f.write(i)
#             # f.write('\n')
# print("分词结果保存成功")

#------------------------------------------

# 用 word2vec 进行训练
sentences=word2vec.Text8Corpus('slurm-13014726.out')
#52776
# #用来处理按文本分词语料
print(sentences)
model = word2vec.Word2Vec(sentences, size=100,window=5,min_count=5,workers=5,sg=1,hs=1)  #训练模型就这一句话  去掉出现频率小于2的词
# model = word2vec.Word2Vec(sentences,sg=1,size=100,window=5,min_count=5,negative=3,sample=0.001,hs=1,workers=4)
# http://blog.csdn.net/szlcw1/article/details/52751314 训练skip-gram模型; 第一个参数是训练预料，min_count是小于该数的单词会被踢出，默认值为5，size是神经网络的隐藏层单元数，在保存的model.txt中会显示size维的向量值。默认是100。默认window=5
# # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
# model=word2vec.Word2Vec(sentences,min_count=3, size=50, window=5, workers=4)

# 保存模型，以便重用
model.save("test_18.model")   #保存模型
model.wv.save_word2vec_format('test_18.model.txt','test_18.vocab.txt',binary=False) # 将模型保存成文本，model.wv.save_word2vec_format()来进行模型的保存的话，会生成一个模型文件。里边存放着模型中所有词的词向量。这个文件中有多少行模型中就有多少个词向量。
model.wv.save_word2vec_format('test_18.model.bin',binary=True)

# y2=model.similarity(u"不错", u"好吃") #计算两个词之间的余弦距离
# print(y2)

# for i in model.most_similar(u"好吃"): #计算余弦距离最接近“滋润”的10个词
#     print(i[0],i[1])

# 训练词向量时传入的两个参数也对训练效果有很大影响，需要根据语料来决定参数的选择，好的词向量对NLP的分类、聚类、相似度判别等任务有重要意义