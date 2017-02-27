def minimum(x):
    least=x
    for each in x:
        if each < least:
            least =each
    return least
print(minimum("432543265476"))
#这个感觉有点跟排序算法相同，就是第一次的时候if each < least比较肯定是是知道each < least的。
#然后就把least=each。进过第一次的if判断，就会把least当成最小的。然后就后面的每一个each跟它比，比他小的就又把最小到给到least。
#其实跟在C语言中，你先假设第一位最小是一样的：min = a[0],
