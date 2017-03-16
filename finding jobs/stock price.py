"""题目描述
有股神吗？
有，小赛就是！
经过严密的计算，小赛买了一支股票，他知道从他买股票的那天开始，股票会有以下变化：第一天不变，以后涨一天，跌一天，涨两天，
跌一天，涨三天，跌一天...依此类推。
为方便计算，假设每次涨和跌皆为1，股票初始单价也为1，请计算买股票的第n天每股股票值多少钱？"""
minus=2
minus2=2
S=1
for i in range(0,10):
    if i>0:
        S+=1
    if i==minus:
        S=S-2
        minus2+=1
        minus+=minus2
    print(S)
# ==================================================WAY 2(from the internet)
def god(days):
    i = 0
    k = 2
    j = 2

    while k < days:
        i += 2
        j += 1
        k += j

    return days - i


if __name__ == '__main__':
    print(god(1))
    print(god(2))
    print(god(3))
    print(god(4))
    print(god(5))
    print(god(6))
    print(god(7))
    print(god(8))
    print(god(9))
    print(god(10))
