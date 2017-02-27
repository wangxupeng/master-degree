#how to create a set?
#A set is created by placing all the items (elements) inside curly brackets {},
# separated by comma or by using the built-in function set().
# It can have different types (integer, float, tuple, string etc.)
# set of integers
my_set = {1, 2, 3}
print(my_set)

# set of mixed datatypes
my_set = {1.0, "Hello", (1, 2, 3)}
print(my_set)

# set does not have duplicates.Every item in the set is unique.
my_set = {1,2,3,4,3,2}
print(my_set)

# set cannot have mutable items
# here [3, 4] is a mutable list
my_set = {1, 2, [3, 4]}

# we can make set from a list
my_set = set([1,2,3,2])
print(my_set)

#Creating an empty set is a bit tricky.
#Empty curly brackets {} will make an empty dictionary in Python.
# To make a set without any elements we use the set() function
a = {}
# check data type of a
print(type(a))

a = set()
# check data type of a
print(type(a))

#Sets are mutable. But since they are unordered( follow a Hash table), indexing have no meaning.
#We cannot access or change an element of set using indexing or slicing. Set does not support it,but we can use
#add() method  and update() method
my_set = {1,3}
print(my_set)
my_set = {3,1}
print(my_set)
my_set[0] # we will get error
# add an element
my_set = {3,1}
my_set.add(2)
print(my_set)

#add multiple elements
my_set.update([2,3,4])
print(my_set)

# add list and set
my_set.update([4,5], {1,6,8})
print(my_set)


#How to remove elements from a set?
#we use discard() and remove()
# initialize my_set
my_set = {1, 3, 4, 5, 6}
print(my_set)

# discard an element
my_set = {1, 3, 4, 5, 6}
my_set.discard(4)
print(my_set)

# remove an element
my_set = {1, 3, 4, 5, 6}
my_set.remove(6)
print(my_set)

# discard an element
# not present in my_set
my_set = {1, 3, 4, 5, 6}
my_set.discard(2)
print(my_set)

# remove an element
# not present in my_set
# you will get an error.
my_set = {1, 3, 4, 5, 6}
my_set.remove(2)

#Similarly, we can remove and return an item using the pop() method.
#We can also remove all items from a set using clear().
# initialize my_set
my_set = set("HelloWorld")
print(my_set)
# pop an element
print(my_set.pop())
# pop another element
my_set.pop()
print(my_set)

# clear my_set
my_set.clear()
print(my_set)

#Sets can be used to carry out mathematical set operations like union, intersection, difference and symmetric difference.
#Union is performed using | (vertical bar) operator.Same can be accomplished using the method union().
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
print(A | B)
# use union function
A.union(B)
# use union function on B
B.union(A)

#Intersection of A and B is a set of elements that are common in both sets.
#Intersection is performed using &(ampersand) operator. Same can be accomplished using the method intersection().
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
# use & operator
print(A & B)

# use intersection function on A
A.intersection(B)
# use intersection function on B
B.intersection(A)

#Difference of A and B (A - B) is a set of elements that are only in A but not in B.
# Similarly, B - A is a set of element in B but not in A.
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# use - operator on A
print(A - B)

# use difference function on A
A.difference(B)
# use - operator on B
B - A
# use difference function on B
B.difference(A)

#Symmetric Difference of A and B is a set of elements in both A and B except those that are common in both.
#Symmetric difference is performed using ^ (caret)operator. Same can be accomplished using the method symmetric_difference().
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
# use ^ operator
print(A ^ B)
# use symmetric_difference function on A
A.symmetric_difference(B)
# use symmetric_difference function on B
B.symmetric_difference(A)

#We can test if an item exists in a set or not, using the keyword in.
my_set = set("apple")

# check if 'a' is present
print('a' in my_set)

# check if 'p' is present
print('p' not in my_set)

#similarly to list comprehension, set comprehensions are also supported
a={ x for x in '123456324r3457' if x not in '123'}
print(a)
