import nltk
from nltk import FreqDist
import jieba

text1 = '我很喜欢这部电影 '
text2 = '这部电影很棒 '
text3 = '这个很不错 '
text4 = '这个真的很烂 '
text5 = '这部电影不行'

text = text1 + text2 + text3 + text4 + text5
words = jieba.cut(text,cut_all=True)
freq_dist = FreqDist(words)
################################################
# 取出常用的n=5个单词
n = 5
# 构造“常用单词列表”
most_common_words = freq_dist.most_common(n)
print(most_common_words)
################################################
def lookup_pos(most_common_words):

    result = {}
    pos = 0
    for word in most_common_words:
        result[word[0]] = pos
        pos += 1
    return result
std_pos_dict = lookup_pos(most_common_words)
print(std_pos_dict)
###############################################
new_text = '那个电影很好看，但是这部电影更棒!'
freq_vec = [0] * n
new_words = jieba.cut_for_search(new_text)
for new_word in new_words:
    if new_word in list(std_pos_dict.keys()):
        freq_vec[std_pos_dict[new_word]] += 1
print(freq_vec)
