words = ['cat','window','defenestrate']
for w in range(0,len(words)):
    if len(words[w])>6:
        print(words[w],len(words[w]))
        

for w in words[:]:
    if len(w)>6:
        words.insert(0,w)
        print(words)
