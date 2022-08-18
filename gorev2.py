

def gorev2 (text):

    yenitext = ""
    for x in range(0,len(text)):
        yenitext+=text[x].upper()
    yenitext=yenitext.replace("."," ")
    yenitext=yenitext.replace(","," ")
    print(yenitext.split())

gorev2("merhaba,ben aleyna.")
