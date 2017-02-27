#This is the lecture of python 24 Feb. Dissertation of ML
#### The module - re ####
import re
print('''24 Feb 10.5-10.8
                         XIAO RU ''')
pro=input("Shall we start?")
print("\n","\n","*"*20,"10.5 String Pattern Matching","*"*20)
em=input()
print("These are examples from our textbook:")
em=input()
e=input(">>> import re")
e=input(">>> re.findall(r'\bf[a-z]*','which foot or hand fell fastest') ")
print(re.findall(r'\bf[a-z]*','which foot or hand fell fastest'))
em=input()
e=input(">>> re.sub(r'(\b[a-z]+)\1', r'\1', 'cat in the the hat')")
print(re.sub(r'(\b[a-z]+)\1', r'\1', 'cat in the the hat'))
em=input()
e=input(">>> 'tea for too'.replace('too', 'two')")
print('tea for too'.replace('too', 'two'))
print("-"*40)#### some thing behind ####
em=input()
e=input(r">>> s='\x5d' ")
s='\x5d'
print(s)
print("-"*40)#### re.match ####
em=input()
e=input(">>> text = 'JGood is a handsome boy, he is cool, clever, and so on...' ")
text = "JGood is a handsome boy, he is cool, clever, and so on..."
em=input()
e=input('''>>> m = re.match(r"(\w+)\s", text)
if m:
    print (m.group(0),'\\n',m.group(1))
else:
    print ('not match')''')
m = re.match(r"(\w+)\s", text)
if m:
    print (m.group(0),'\n',m.group(1))##########################################>>>>>
else:
    print ('not match')
print("-"*40)#### re.search ####
em=input()
e=input('''>>> m = re.search(r'\shan(ds)ome\s', text)
if m:
    print (m.group(0), m.group(1))
else:
    print ('not search')''')
m = re.search(r'\shan(ds)ome\s', text)
if m:
    print (m.group(0), m.group(1))##########################################>>>>>
else:
    print ('not search')
print("-"*40)#### re.sub ####
em=input()
e=input(">>> re.sub(r'\s+', '-', text)")
print (re.sub(r'\s+', '-', text))
print("-"*40)#### re.split ####
em=input()
e=input(">>> re.split(r'\s+', text)")
print(re.split(r'\s+', text))
print("-"*40)#### re.findall ####
em=input()
e=input(">>> re.findall(r'\w*oo\w*', text)")
print(re.findall(r'\w*oo\w*', text))
print("-"*40)#### re.compile ####
em=input()
e=input('''>>> regex = re.compile(r'\w*oo\w*')''')
e=input('''>>> regex.findall(text)''')
regex = re.compile(r'\w*oo\w*')
print (regex.findall(text)) #查找所有包含'oo'的单词
em=input()
e=input(">>> regex.sub(lambda m: '[' + m.group(0) + ']', text)")
print (regex.sub(lambda m: '[' + m.group(0) + ']', text)) #将字符串中含有'oo'的单词用[]括起来。
em=input()
e=input("A password check tool")
for i in range(5):
    co=input("Input your password:")
    m0=re.search('\w',co)
    m=re.search('\W',co)
    m1=re.search('\d',co)
    m2=re.search('[a-z]',co)
    if m0==None and m==None:
        print("Please print something")
        continue
    elif len(co) < 13:
        print("Weak password. Password should be longer")
        ask=input("Do you want try again? Y/N ----> ")
        if ask=='Y' or ask=='y':
            continue
        else:
            break
    elif m==None and m1 and m2:
        print('''Password is not complex enough but ok. You can try some special
              characters like $ % @ !''')
        ask=input("Do you want try again? Y/N ---->")
        if ask=='Y'or ask=='y':
            continue
        else:
            break
    elif m and m1 and m2:
        print("your password is good.")
        break
    else:
        print("loss of letters or numbers")
        ask=input("Do you want try again? Y/N ---->")
        if ask=='Y'or ask=='y':
            continue
        else:
            break
if i==4:
    print("-"*45)
    print('''   You have tried so many times
    If you want to try again
    please restart the program''')
else:
    pass
em=input("END ")
print("\n","\n","*"*20,"10.6 Mathematics ","*"*20)##10.6>>>>>>>>>>>>>>----------
em=input()
e=input(">>> import math")
import math
e=input(">>> math.e")
print(math.e)
em=input()
e=input(">>> math.pi")
print(math.pi)
em=input()
e=input(">>> math.cos(math.pi / 4)")
print(math.cos(math.pi / 4))
em=input()
e=input(">>> math.log(1024, 2)")
print(math.log(1024, 2))
print("-"*40)
em=input()
e=input(">>> import random")
import random
e=input(">>> random.choice(['apple', 'pear', 'banana'])")
print(random.choice(['apple', 'pear', 'banana']))
em=input()
e=input(">>> random.choice(['apple', 'pear', 'banana'])")
print(random.choice(['apple', 'pear', 'banana']))
em=input()
e=input(">>> random.choice(['apple', 'pear', 'banana'])")
print(random.choice(['apple', 'pear', 'banana']))
em=input()
e=input(">>> random.sample(range(100), 10)")
print(random.sample(range(100), 10))
em=input()
e=input(">>> random.random()")
print(random.random())
em=input()
e=input(">>> random.random()")
print(random.random())
em=input()
e=input(">>> random.random()")
print(random.random())
em=input()
e=input(">>> random.randrange(6)")
print(random.randrange(6))
em=input()
e=input(">>> random.randrange(10)")
print(random.randrange(10))
em=input()
for i in range(11):
    print(random.randrange(10))
print("-"*40)
em=input()
e=input(">>> import statistics")
import statistics
e=input(">>> data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]")
data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
em=input()
e=input(">>> statistics.mean(data)")
print(statistics.mean(data))
em=input()
e=input(">>> statistics.median(data)")
print(statistics.median(data))
em=input()
e=input(">>> statistics.variance(data)")
print(statistics.variance(data))
em=input()
print("\n","\n","*"*20,"10.8 Dates and Times ","*"*20)
em=input()
e=input("# dates are easily constructed and formatted")
e=input(">>> from datetime import date")
from datetime import date
e=input(">>> now = date.today()")
e=input(">>> now")
now=date.today()
print(now)
em=input()
e=input(">>> now.strftime('%m-%d-%y. %d %b %Y is a %A on the %d day of %B.')")
print(now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B."))
em=input()
e=input(">>> birthday = date(1964, 7, 31)")
birthday = date(1964, 7, 31)
age = now - birthday
e=input(">>> age = now - birthday")
e=input(">>> age.days")
print(age.days)
em=input()
print("\n","\n","*"*20,"10.7 Internet ","*"*20)
em=input()
e=input(">>> from urllib.request import urlopen")
from urllib.request import urlopen
e=input(">>> doc=urlopen('http://www.google.com')")
doc=urlopen("http://www.google.com")
e=input(">>> doc.info()")
print(doc.info())
print("-"*40)
em=input()
e=input('''No more code!
No more code!
No more code!
No more code!
No more code!''')
