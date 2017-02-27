#del function can delete one item or serveral items from the list. This differs
#from the pop() method which returns a value.
a = [-1, 1, 66.25, 333, 333, 1234.5] #define a list
del a[0]  #  square brackets.(delete the item 1)
print(a)
a.pop(0)  # to see the difference

del a[2:5]  # delete item 3 to item 5 ( not include item 6)

del a[:] # the list still exists.(colon & square brackets)

del a # it will give us an error, because everything was deleted.


word_initial = ["我也喜欢!","╭∩╮（︶︿︶）╭∩╮"]
q= "好"
while q == "好":
    word = input("你喜欢林志玲吗?(请输入(喜欢或不喜欢):")
    if word == "喜欢":
        y=word_initial[:]
        del y[1]
        print(y)
    if word == "不喜欢":
        x= word_initial[:]
        x[0]=x[0].replace('我也','你竟然不')
        print(x)
    q = input("再给你一次机会(请输入好或不好):")
