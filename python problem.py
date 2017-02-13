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
