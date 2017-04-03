import numpy as np
import pandas as pd
# There are 5 general ways for creating arrays:
# 1. Conversion from other Python structures (e.g., lists, tuples)
# 2. Intrinsic numpy array array creation objects (e.g., arange, ones, zeros, etc.)
# 3. Reading arrays from disk, either from standard or custom formats
# 4. Creating arrays from raw bytes through the use of strings or buffers
# 5. Use of special library functions (e.g., random)
# =======================================================================
# Converting Python array_like Objects to Numpy Arrays
# a=[2,3,1,0]
# x = np.array(a)
# print(x,type(x))
# x = np.array([[1,2.0],[0,0],(1,3.)]) # note mix of tuple and lists,and types
# x = np.array([[1,2.0],[0,0],(1+1j,3.)])
# x = np.array([[1,2.0],[0+1j,0],(1,3)])
# print(x,type(x))
# x = np.array([[ 1.+0.j, 2.+0.j], [ 0.+0.j, 0.+0.j], [ 1.+1.j, 3.+0.j]])
# print(x,type(x))
# =======================================================================
# Intrinsic Numpy Array Creation
# zeros(shape) will create an array filled with 0 values with the specified shape. The default dtype is float64.
# a=np.zeros((2, 3))
# print(a,a.dtype)
# ones(shape) will create an array filled with 1 values. It is identical to zeros in all other respects
# a=np.ones((2, 3))
# print(a,a.dtype) #dtype is also float64.
# arange() will create arrays with regularly incrementing values.
# a=np.arange(10)
# print(a,type(a))
# b=np.arange(0, 10, dtype=np.int)
# print(b)
# c= np.arange(2, 4, 0.1)#For floating point arguments, the length of the result is ceil((stop - start)/step).
# print(c,len(c))
# linspace() will create arrays with a specified number of elements, and spaced equally between the specified beginning
# and end values
# a=np.linspace(1., 4., 6)# here we set the array has 6 items, it will return 6 number and numbers have same interval.
# print(a)
# b=np.arange(1., 4., 0.6)
# print(b)
# The advantage of this creation function is that one can guarantee the number of elements and the starting and end
# point, which arange() generally will not do for arbitrary start, stop, and step values.
# indices() will create a set of arrays (stacked as a one-higher dimensioned array), one per dimension with each
# representing variation in that dimension.
# a=np.indices((2,3))
# print(a)
# x = np.arange(20).reshape(5, 4)
# row, col = np.indices((2, 3))
# print(row,col)
# print(x,x[row, col])
# =======================================================================
# Reading Arrays From Disk
# CSV:In computing, a comma-separated values (CSV) file stores tabular data (numbers and text) in plain text.
# Each line of the file is a data record. Each record consists of one or more fields, separated by commas.
# arr = np.loadtxt('array_ex.txt', delimiter=',')# we use comma to separate the data
# arr = np.loadtxt('array_ex.txt', delimiter=',',usecols=(0,1,2))#usecols stands for the column which we choose.
# arr = np.loadtxt('array_ex.txt', delimiter=',',dtype=int)#the data type
# print(arr)
# print(arr.shape)
# ================================================================
# read_csv Load delimited data from a file, URL, or file-like object. Use comma as default delimiter
# df = pd.read_csv('ex1.csv') #we can use read_csv to read it into a DataFrame
# df=pd.read_table('ex1.csv', sep=',')
# print(df,type(df))
# ======================================================================
# A file will not always have a header row. Consider "ex2.csv"
# To read this in, you have a couple of options. You can allow pandas to assign default
# column names, or you can specify names yourself
# df=pd.read_csv('ex2.csv', header=None) #way 1
# df=pd.read_csv('ex2.csv', names=['a', 'b', 'c', 'd', 'message'])
# print(df)
# Suppose you wanted the message column to be the index of the returned DataFrame.
# You can either indicate you want the column at index 4 or named 'message' using the
# index_col argument:
# names = ['a', 'b', 'c', 'd', 'message']
# df1=pd.read_csv('ex2.csv', names=names, index_col='message')
# print(df1)
# ======================================================================
# you can skip the first, third, and fourth rows of a file with skiprows:
# df=pd.read_csv('ex4.csv', skiprows=[0, 2, 3])
# print(df)
# ======================================================================
# Handling missing values is an important process. Missing data is usually either not present (empty string)
# or marked by some sentinel value. By default, pandas uses a set of commonly occurring sentinels, such as
# NA and NULL:
# result = pd.read_csv('ex5.csv')
# print(result)
# ======================================================================
# When processing very large files or figuring out the right set of arguments to correctly
# process a large file, you may only want to read in a small piece of a file or iterate through
# smaller chunks of the file.If you want to only read out a small number of rows (avoiding reading the entire file),
# specify that with nrows:
# result = pd.read_csv('ex6.csv', nrows=10)#Number of rows to read from beginning of file
# print(result)
