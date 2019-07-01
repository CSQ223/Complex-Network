import numpy as np
import pandas as pd
import os
import jieba
import matplotlib.pyplot as plt
import re

#导入信息
messages = pd.read_csv('message.csv', encoding='UTF-8', index_col=0, header=None)
messages.columns = ['type', 'content']

print('类别比例：\n',messages['type'].value_counts())

#样本分布
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
labels = '非垃圾短信', '垃圾短信'
plt.axes(aspect=1)  # 设置aspect=1,则所得图案为圆形，否则默认为椭圆
explode = [0, 0.1]  # 0.1为凸出这部分，
plt.pie(x=messages['type'].value_counts(),  # 数值
        labels=labels,  # 标签
        explode=explode,  # 设置凹出的部分
        autopct='%3.1f %%',  # 圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
        shadow=True,  # 饼是否有阴影
        labeldistance=1.1,  # 文本的位置离远点有多远，1.1指1.1倍半径的位置
        startangle=90,  # 起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
        pctdistance=0.6  # 百分比的text离圆心的距离
        )
plt.title('垃圾短信与非垃圾短信的一个分布情况')
plt.savefig('样本分布情况.png', format='png')
plt.show()

##########################################处理样本
#获取当前内容
contents = messages['content']
##################################  检查是否存在缺失
print('样本缺失数量：',contents.isnull().sum())
contents = contents.loc[[i for i in contents.index if contents[i] != '']] #找出非空的信息

##################################  去除重复值
#总字符数
num1 = contents.astype('str').apply( lambda x: len(x)).sum()
print('总字符数：', num1)

#去除重复的项
contents_drop = contents.drop_duplicates()
num2 = contents_drop.astype('str').apply( lambda x: len(x)).sum()
print('去除重复值之后字符数：{0}，减少了{1}'.format(num2, num1-num2))

#将奇怪的字符串去除 re.sub(patern, replace, str)
contents_clear = contents_drop.astype('str').apply(lambda x: re.sub('x', '', x))
num3 = contents_clear.astype('str').apply(lambda x: len(x)).sum()
print('去除奇怪字符之后字符数：{0}，减少了{1}'.format(num3, num2-num3))


########################################## 分词
#加载用户词典
jieba.load_userdict('userdict.txt')
#分词
contents_cut = contents_clear.astype('str').apply(lambda x: list(jieba.cut(x)))
#去除停用词
stopword_file = 'stopword.txt'
stopword = pd.read_csv(stopword_file, sep = 'bingrong', encoding = 'gbk', header = None)
stopword = [' ',',', '会', '的', '】', '【', '月', '日'] + list(stopword[0])  #添加停用词
num4 = contents_cut.astype('str').apply(lambda x: len(x)).sum()

contents_stop = contents_cut.apply(lambda x:[i for i in x if i not in stopword])
num5 = contents_stop.astype('str').apply(lambda x: len(x)).sum()
print('去除停用词之后字符数：{0}，减少了{1}'.format(num5, num4-num5))

#找到剩下的所有的标签
lab = [messages.loc[i, 'type'] for i in contents_stop.index]
labl = pd.Series(lab, index=contents_stop.index)

contents_garb = contents_stop.loc[labl==1]#标签为1为垃圾短信
contents_norm = contents_stop.loc[labl==0]#标签为0为正常短信

#统计词频
def statistic(contents, frequency=10):
    temp = [' '.join(x) for x in contents]
    temp1 = ' '.join(temp)
    temp2 = pd.Series(temp1.split()).value_counts()
    return temp2[temp2>frequency]

stat_garb = statistic(contents_garb, frequency=5)
stat_norm = statistic(contents_norm, frequency=50)

#绘制词云图
from scipy.misc import imread
from wordcloud import WordCloud
import matplotlib.pyplot as plt


back_image = imread("duihuakuan.jpg")  # 设置背景图片
wc = WordCloud(font_path='C:\\Windows\\Fonts\\simkai.TTF',  # 设置字体
               background_color="white",  # 背景颜色
               max_words=2000,  # 词云显示的最大词数
               mask=back_image,  # 设置背景图片
               max_font_size=200,  # 字体最大值
               random_state=42)
#垃圾短信的词云图
wordcloud = wc.fit_words(stat_garb)
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
wc.to_file("garbage.png")  # 保存图片

#正常短信的词云图
wordcloud = wc.fit_words(stat_norm)
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
wc.to_file("norm.png")  # 保存图片

############################################# 准备训练
############################################# 抽样
sample_num = 10000
data_garb = contents_garb.sample(sample_num, random_state=123)
data_norm = contents_norm.sample(sample_num, random_state=123)
#拼接起来形成总的样本
data = pd.concat([data_garb, data_norm])

#添加类别标签
lbl = pd.Series([0]*sample_num+[1]*sample_num, index=data.index)
data_sample = pd.concat([data, lbl], axis=1)
data_sample.columns = ['contents', 'label']

############################################# 划分数据集
############################################# 训练集：测试集 = 8：2
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_sample.contents,
                                                    data_sample.label,
                                                    test_size = 0.2,
                                                    random_state = 520)

############################################# 构建词条文本矩阵
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

#cv = CountVectorizer()
#train_cv = cv.fit_transform(x_train.astype('str'))
#train_cv.toarray()
cv = CountVectorizer()
train_transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
train_tfidf=train_transformer.fit_transform(cv.fit_transform(x_train.astype('str')))
train_tfidf.toarray()

############################################# 建模：朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
#nb.fit(train_cv, y_train)
nb.fit(train_tfidf, y_train)

cv1 = CountVectorizer(vocabulary=cv.vocabulary_)
#test_cv = cv1.fit_transform(x_test.astype('str'))
test_transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
test_tfidf=train_transformer.fit_transform(cv1.fit_transform(x_test.astype('str')))
pre = nb.predict(test_tfidf)

############################################# 评价模型
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, pre)
result = classification_report(y_test, pre)
print(result)


############################################ 建模——SVC
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(train_tfidf, y_train)

cv1 = CountVectorizer(vocabulary=cv.vocabulary_)
#test_cv = cv1.fit_transform(x_test.astype('str'))
test_transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
test_tfidf=train_transformer.fit_transform(cv1.fit_transform(x_test.astype('str')))
pre = svc.predict(test_tfidf)
############################################# 评价模型
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, pre)
result = classification_report(y_test, pre)
print(result)


def judge(msg=''):
    jieba.load_userdict('userdict.txt')
    #分词
    msg = msg.astype('str').apply(lambda x:re.sub('x','',x))
    msg_cut = msg.astype('str').apply(lambda x: list(jieba.cut(x)))
    #去除停用词
    stopword_file = 'stopword.txt'
    stopword = pd.read_csv(stopword_file, sep = 'bingrong', encoding = 'gbk', header = None,engine='python')
    stopword = [' ',',', '会', '的', '】', '【', '月', '日'] + list(stopword[0])  #添加停用词
    msg_stop = msg_cut.apply(lambda x:[i for i in x if i not in stopword])
    
    ############################################ 建模——SVC
    from sklearn.svm import LinearSVC
    svc = LinearSVC()
    svc.fit(train_tfidf, y_train)

    cv1 = CountVectorizer(vocabulary=cv.vocabulary_)
    #test_cv = cv1.fit_transform(x_test.astype('str'))
    test_transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    test_tfidf=train_transformer.fit_transform(cv1.fit_transform(msg_stop.astype('str')))
    pre = svc.predict(test_tfidf)
    return pre

txt = '乌兰察布丰镇市法院成立爱心救助基金'
msg = pd.Series(txt)
flag = judge(msg)
if flag==0:
    print('是垃圾短信')
else:
    print('不是垃圾短信')
    

