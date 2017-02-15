#python problem 
#Q1
list1=[1,2,3,4,5,6,7,8]
import random
random.shuffle(list1)
print(list1)

#Q2
list1=["123","1","3"]
list1[0]=list1[0].replace("2","")
list1

#Q3
def MyFun((x, y), (a, b)):
    return x * y - a * b
#the function is wrong, because we cannot use tuple to create functions. tuple is immutable

#Q4
def count(*x):
    list1=[]
    list2=[]
    for i in x:
        i+=1
        list1.append(i)
    print(list1)
    for a in list1:
        a+=1
        list2.append(a)
    print(list2)

count(1,2,3)

list1=["1","2","3"]
list2=[]
for i in list1:
    a = float(i)
    a+= 1
    list2.append(a)
print(list2)
