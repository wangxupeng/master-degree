import nltk
from nltk import FreqDist
import jieba
from nltk import TextCollection

text1 = '我很喜欢这部电影 '
text2 = '这部电影很棒 '
text3 = '这个很不错 '
text4 = '这个真的很烂 '
text5 = '这部电影不行'

tc = TextCollection([text1, text2, text3, text4, text5])
new_text = '这部电影实在太好看了!'
word = '电影'
tf_idf_val = tc.tf_idf(word, new_text)
print('{}的TF-IDF值为：{}'.format(word, tf_idf_val))
