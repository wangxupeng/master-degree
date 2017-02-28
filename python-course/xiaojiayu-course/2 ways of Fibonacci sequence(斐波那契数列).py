def fab(x):
    x1=1
    x2=1
    x3=1
    if x<1:
        print(-1)
        print("输入错误!")
    if x==1 or x==2:
        print(1)
    while x>=3:
        x1=x2
        x2=x3
        x3=x1+x2
        x -= 1
        print(x3)
fab(3)
##############################################
def fab(x):
    if x<1:
        print("输入错误!")
        return -1
    if x == 1 or x== 2:
        return 1
    else:
        return fab(x-1) + fab(x-2)
print(fab(0))
