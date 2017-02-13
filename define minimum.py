def minimum(x):
    least=x
    for each in x:
        if each < least:
            least =each
    return least
print(minimum("432543265476"))
