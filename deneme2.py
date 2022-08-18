list1=[]
list2=[]

def divide_students (students):
    for index, student in enumerate(students):
        if index%2==0:
            list2.append(students[index])
        else:
            list1.append(students[index])
    return list1+list2

list3=["a","b","c","d"]
list4=divide_students(list3)

print(list4)