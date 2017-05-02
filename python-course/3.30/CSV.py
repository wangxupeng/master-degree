import numpy as np
# arr = np.loadtxt('array_ex.txt', delimiter=',')# we use comma to separate the data
# arr = np.loadtxt('array_ex.txt', delimiter=',',usecols=(0,1,2,3))#usecols stands for the column which we choose.
# arr = np.loadtxt('array_ex.txt', delimiter=',',dtype=np.str)#the data type
# data = arr[:,:].astype(np.float)
# print(arr)
# print(arr.shape)
# print(data)
# ================================================================
# from io import StringIO   # StringIO behaves like a file object
# c = StringIO("0 1\n2 3")
# print(np.loadtxt(c))
# d = StringIO("M 21 72\nF 35 58")
# e=np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'), 'formats': ('U8', 'i4', 'f4')})
# #http://www.peergroup.com/Resources/SECSMessageLanguage.aspx
# print(e)
# ======================================================================
# read_csv Load delimited data from a file, URL, or file-like object. Use comma as default delimiter
import pandas as pd
# df = pd.read_csv('ex1.csv') #we can use read_csv to read it into a DataFrame
# df=pd.read_table('ex1.csv', sep=',')
# print(df)
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
# In the event that you want to form a hierarchical index from multiple columns, just
# pass a list of column numbers or names:
# parsed = pd.read_csv('csv_mindex.csv', index_col=['key1', 'key2'])
# print(parsed)
# ======================================================================
# the parser functions have many additional arguments to help you handle the wide
# variety of exception file formats that occur. For example, you can skip
# the first, third, and fourth rows of a file with skiprows:
# df=pd.read_csv('ex4.csv', skiprows=[0, 2, 3])
# print(df)
# ======================================================================
# Handling missing values is an important and frequently nuanced part of the file parsing
# process. Missing data is usually either not present (empty string) or marked by some
# sentinel value. By default, pandas uses a set of commonly occurring sentinels, such as
# NA, -1.#IND, and NULL:
# result = pd.read_csv('ex5.csv')
# print(result)
# ======================================================================
# When processing very large files or figuring out the right set of arguments to correctly
# process a large file, you may only want to read in a small piece of a file or iterate through
# smaller chunks of the file.If you want to only read out a small number of rows (avoiding reading the entire file),
# specify that with nrows:
# result = pd.read_csv('ex6.csv', nrows=1)#Number of rows to read from beginning of file
# print(result)
# ======================================================================
# To read out a file in pieces, specify a chunksize as a number of rows:
# chunker = pd.read_csv('ex6.csv', chunksize=1000)
# print(chunker)
# The TextParser object returned by read_csv allows you to iterate over the parts of the
# file according to the chunksize. For example, we can iterate over ex6.csv, aggregating
# the value counts in the 'key' column like so:
# tot = pd.Series([])
# for piece in chunker:
#     tot = tot.add(piece['key'].value_counts(), fill_value=0)
# tot=tot.sort_values()
# print(tot)
# ======================================================================
