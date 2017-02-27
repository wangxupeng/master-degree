####course 18
def times(*params,base=3):
     result=0
     for i in params:
         result += i

     result *= base
     print("结果是:",result)
  ####################################
     
     
def counts(a,b):
    count=0
    length=len(a)
    if b not in a :
        print("你输入的东西不存在.")
    else:
        for i in range(length-1):
            if a[i] == b[0]:
                if a[i+1] == b[1]:
                    count += 1
    print("你的字符串出现:",count,"次")
a=input("请输入字符串:")
b=input("要查找的字符串:")
counts(a,b)

#another way
def findstr1():
    flag=0
    str=input("请输入字符串")
    se=input("请输入您想查的字符串")
    for i in range(len(str)):
        if str[i:(i+len(se))]==se[0:]:
            flag +=1
        else:
            pass
    print("这个出现了",flag,"次")
findstr1()
#####course19
def discounts(price,rates):
    result=price*rates
    print(result)

old_price=float(input("请输入价格:"))
old_rates=float(input("请输入折扣:"))
new_price=discounts(old_price,old_rates)

######################################
newx=input("请输入:")
def count(x):
    alpha=0
    digit=0
    space=0
    others=0
    for i in x:
        if i.isdigit():
            digit += 1
        elif i.isalpha():
            alpha +=1
        elif i.isspace():
            space += 1
        else:
            others += 1
    print("数字",digit,"字母",alpha,"空格",space,"其他",others)

print(count(newx))

###########################################################course 19
def check(*strs):
    for each in range(len(strs)):
        alpha=0
        digit=0
        space=0
        others=0
        for i in strs[each]:
            if i.isalpha():
                alpha +=1
            elif i.isdigit():
                digit +=1
            elif i.isspace():
                space +=1
            else:
                others+=1
        print("第%d个字符串,有%d个字母,有%d个数字,有%d个空格,和%d个其他字符"%(each,alpha,digit,space,others))
     #########################################
     list1=["1.耐克","2.李宁"]
list2=["2.一切皆有可能","1.just do it"]
list3=[name+":"+slogan[2:] for name in list1 for slogan in list2 if name[0]==slogan[0]]
for each in list3:
    print(each)
