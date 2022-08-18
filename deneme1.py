text = "my name is john and i am learning"
ntext=" "
i = len(text)

for x in range(0,i):
    if x%2==0:
        ntext += text[x].upper()
    else:
        ntext+=text[x].lower()


print(ntext)
