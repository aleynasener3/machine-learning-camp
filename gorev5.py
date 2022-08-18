def func(liste):
    cift=[]
    tek=[]
    for i in liste:
        if i%2==0:
            cift.append(i)
        else:
            tek.append(i)

    return cift , tek
cift,tek = func([1,2,3,4,5,6,7])
print(cift)
print(tek)