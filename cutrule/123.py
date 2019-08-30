# a='a,b,c'
# dic={'a':1 ,'b':2 ,'c':3}
# a=a.split(',')
# lists=[[]]*(len(a))
#
#
# print(lists)
# lists = [[]] * 5
# lists.append(1)
# print(lists)
# lists[2]+2
#
# print(lists[3])

# lists=[]
#
# for i in range(5):
#     list = []
#     if i==1:
#         list.append(2)
#     lists.append(list)
# # print(lists)
# f=open('ptack_auto_all','r+')
# for i in f:
#     sp=i.split('\t')
#     if sp[2]=='':
#         print(111)
# import  matplotlib.pyplot as plt
# all=0
# count=0
# f=open('ptack找錯了','r+')
# new=open('ptack錯的','w+')
#
# for i in f:
#     sp=i.split('\t')
#     if sp[0]!=sp[1]:
#         new.write(sp[1])
#         new.write('\t')
#         new.write(sp[2])
# f=open('ptack_auto_14000保留標點_checkpoint.txt','r+')
# new=open('ptack_auto_14000保留標點_checkpoint2.txt','w+')
# for i in f:
#     c=i.strip('\n')
#     if len(c)!=0:
#         new.write(c)
#         new.write('\n')
# import keras
#
# print(keras.__file__)

# count=0
# alist=np.array([])
# print(type(alist))
# for i in A:
#
#     count=count+1
#     if count<=3:
#         )
# print(alist)




# f=open('ptackcut完整.txt','r+')
# new=open('ptackcut完整_checkpoint.txt','w+')
# for i in f:
#     new.write(i.replace('\t','|||'))
#     print(i)
import re
import string
f=open('ptack_5ek.txt','r+')
new=open('ptack剩餘完整.txt','w+')
new2=open('ptack標點完整.txt','w+')
# new=open('ptack_auto_14000保留標點','w+')
count=0
for i in f:                     #抓dic
    line=i.split('\t')[1]
    label=i.split('\t')[0]
    line = line.strip().strip(string.punctuation)       #去首尾標點
    line = line.strip().strip(string.punctuation)
    line = line.strip().strip(string.punctuation)
    if '.' in line :            #小數點換int
        c = re.findall(r"[-+]?[0-9]*\.[0-9]+", line)
        if c != []:
            for h in c:
                line = line.replace(h, str(int(float(h))))
    if ',' in line or '.' in line or '!' in line or '?' in line or '#' in line or ':' in line:
        new2.write(label)
        new2.write('\t')
        new2.write(line)
        new2.write('\n')
    else:
        new.write(label)
        new.write('\t')
        new.write(line)
        new.write('\n')