f = open('123.txt', 'r')
result = list()
for line in f.readlines(): #按行读取(read f by row)
    line = line.strip()    #按行分开成list(strip it in line)
    if not len(line) or line.startswith('#'):  #如果是空的(if it is enmpty)
        continue
    result.append(line)
a= result #存储值(store every role in a)
n=len(a)


list1=[]
list2=[]
count=0
counta=0
for i in a:
    list1.append(i.split(" "))  #将每一行按list分开(separate a by row)
for row in list1:#读取每个row(read every row in the list1)
    c=set(row)
    b={"bourbon"}  #输入基准的值(input the variable of support)
    if b.issubset(c):
        count+=1
        list2.append(row)
for ii in list2:
    e=set(ii)
    d={"olives"}  #输入基准值中包含的另一项(input confidentce)
    if d.issubset(e):
        counta+=1
print("%.2f%%"%(counta/count*100))





