score=[["迷途",85],["黑夜",80],["小布丁",65],["福禄娃娃",95],["怡静",90]]
isfind= False
q="y"
while q == "y":
    name= input("请输入待查找的用户名:")
    for each in score:
        if name in each:
            print(name + "的得分是:",each[1])
            isfind=True
            break
        if isfind == False:
            print("查找的数据不存在.")
    q = input("你还想继续吗?请输入y:")
