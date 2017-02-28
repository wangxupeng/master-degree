dataset_outlook = [
   ("sunny", "yes", "yes", "no", "no","no")
   ,("overcast", "yes", "yes","yes", "yes",)
   ,("rainy", "yes",  "yes","yes", "no")
   ,("foggy","yes","yes","no","no","no")]
def count_y_n(x):
    for i in dataset_outlook:
        i={i[0]:i[1:]}
        for obj in i:
            if obj == x :
                yes_count = 0
                no_count = 0
                prob1=0
                prob2=0
                for count in i[x]:
                    if count == "yes":
                        yes_count += 1
                    else:
                        no_count += 1
                prob1= yes_count/(no_count+yes_count)
                prob2=no_count/(yes_count+no_count)
                return prob1,prob2
from math import *
def entropy(x):
    if x == 0:
        return 0
    else:
        result= -x*log(x,2)
        return  result
result_entropy=[]
infoname=[]
for i in dataset_outlook:
        i={i[0]:i[1:]}
        for obj in i:
            infoname.append(obj)
            a=count_y_n(obj)
            b=entropy(a[0])+entropy(a[1])
            result_entropy.append(b)

list_name_entropy=list(map(lambda x,y:[x,y],infoname,result_entropy))
print(list_name_entropy)
print("Here is the 'item' and its corresponding entropy")

length_byname=[]
for i in dataset_outlook:
    a=len(i)-1
    length_byname.append(a)

sum_lb= sum(length_byname)
infoaf_para=list(map(lambda x:x/sum_lb,length_byname))
info_after_split=list(map(lambda x,y:x*y,infoaf_para,result_entropy))
after_split=sum(info_after_split)
print(after_split)
print("Here is the information after split")
count_yesinall=0
count_noinall=0
for i in dataset_outlook:
    for yn in i:
        if yn == "yes":
            count_yesinall += 1
        if yn == "no":
            count_noinall += 1
allyes=count_yesinall/(count_noinall+count_yesinall)
allno=count_noinall/(count_noinall+count_yesinall)
before_split=-allyes*log(allyes,2)-allno*log(allno,2)
print(before_split)
print("Here is the information before split")
information_gain= before_split-after_split
print("So the information gain is %f"%(information_gain))
