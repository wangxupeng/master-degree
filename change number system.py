q = True
while q :
    num = input("请输入一个数字(q退出):")
    if num != "q" : # have to pay attention to the double quotes
        num = int(num)
        print("十进制->十六进制"+"%d->%#X"%((num),(num)))
        print("十进制->八进制"+"%d->%#o"%((num),(num)))
        print("十进制->二进制"+"%d"%(num)+"->"+bin(num))
    else:
        q = False

