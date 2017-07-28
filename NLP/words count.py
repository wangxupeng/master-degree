import nltk
from nltk import FreqDist
import jieba

text1 = '我很喜欢这部电影 '
text2 = '这部电影很棒 '
text3 = '这个很不错 '
text4 = '这个真的很烂 '
text5 = '这部电影不行'

text = text1 + text2 + text3 + text4 + text5
words = jieba.cut(text,cut_all=False)
freq_dist = FreqDist(words)

n = 5
most_common_words = freq_dist.most_common(n)
print(most_common_words)
