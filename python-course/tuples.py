#python has many types of sequence,for example,list, tuple, and strings
#how to create a tuple?
# tuple with mixed datatypes
t = 12345, 54321, 'hello!' #integer and string,separated by comma
t = (12345, 54321, 'hello!')
# empty tuple
t = ()
#how to create a tuple with one item
t1 =(1)
type(t1)
t1 = 1
type(t1)

t1 = (1,)
t1 = 1,
#so the comma is very important to the tuple
8*(8)
8*(8,)

#change list to tuple
t = [12345, 54321, 'hello!']
b=tuple(t)
type(b)
# Tuples may be nested
## it can contain mutable or immutable objects:
#immutable objects: int  float  tuple  string
#mutable objects:list set dictionary
t = 12345, 54321, 'hello!'
u= t,(1,2,3,4,5)
a=((1,2),(3,4),5)
b=[[1,2],[3,4],5]

dict1 = {1: 'apple', 2: 'ball'}
v =({1: 'apple', 2: 'ball'},) #pay attention to the comma
type(v)

# the tuple can be indexed and sliced
t = (12345, 54321, 'hello!')
t[0]
t[0:2]
t[:]

# tuple are immutable
t = (12345, 54321, 'hello!')
t[0]= 88888

#we will get error
#how to add something into the tuple
temp = ('1','2','3','4')
type(temp)
temp = temp[:2]+('5',)+temp[2:] # this is a new tuple,not the original one

# this is an example of tuple packing: the values 12345, 54321 and ’hello!’ are packed together in a tuple.
t = (12345, 54321, 'hello!')
#sequence unpacking,take all items out of the pack, one by one.
x, y, z = t
print(x)
print(y)
print(z)
#how to delete a tuple
del t
