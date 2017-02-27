def grad(x):
    if x=="A":
        gradeGPA=4.00
    elif x=="A-":
        gradeGPA=3.67
    elif x=="B+":
        gradeGPA=3.33
    elif x=="B":
        gradeGPA=3.00
    elif x=="B-":
        gradeGPA=2.67
    elif x=="C+":
        gradeGPA=2.33
    elif x=="C":
        gradeGPA=2.00
    elif x=="C-":
        gradeGPA=1.67
    else:
        gradeGPA=0
    return gradeGPA

def ovgrad(g1,g2,g3,g4,g5):
    list=[grad(g1),grad(g2),grad(g3),grad(g4),grad(g5)]
    return list
##########################
x=int(input("Please enter an integer: "))
if x<0:
    x=0
    print('negative changed to zero')

elif x==0:
        print ('Zero')

elif x==1:
        print ('Single')

else:
        print ('More')
        ###############################################################
        
