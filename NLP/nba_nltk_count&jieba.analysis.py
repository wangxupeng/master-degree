import re
import jieba
import nltk
from nltk import FreqDist
import jieba.analyse as analyse

with open('NBA.txt','r', encoding='UTF-8') as f:
    lines = f.read()
filter_pattern = re.compile("(?i)[^a-zA-Z0-9\u4E00-\u9FA5]")
article = filter_pattern.sub(" ",lines)
article_cut = jieba.cut(article,cut_all=False)
text =[]
for line in article_cut:
    text.append(line)


stopwords1 = [line.rstrip() for line in open('D:\python project\course\lect06_codes\lect06_proj\中文停用词库.txt', 'r', encoding='utf-8')]
stopwords = stopwords1
meaninful_words = []
for word in text:
    if word not in stopwords:
        meaninful_words.append(word)

freq_dist = FreqDist(meaninful_words)
n = 20
most_common_words = freq_dist.most_common(n)
print(most_common_words)

print("  ".join(analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=())))
