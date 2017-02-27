f = open('123.txt', 'r')
result = list()
for line in f.readlines(): #按行读取(read f by row)
    line = line.strip()    #按行分开成list(strip it in line)
    if not len(line) or line.startswith('#'):  #如果是空的(if it is enmpty)
        continue
    result.append(line)
a= result #存储值(store every role in a)
n=len(a)


def support(*x):
    list1=[]
    count=0
    for i in a:
        list1.append(i.split(" "))  #将每一行按list分开(separate a by row)
    for row in list1:#读取每个row(read every row in the list1)
        c=set(row)
        b={*x}
        if b.issubset(c):#b是c的子集(b is the subset of c)
            count +=1
    print("%.2f%%"%(count/n*100))

print(support("heineken","cracker")) #想要输入的变量(the variables that you want to input)
