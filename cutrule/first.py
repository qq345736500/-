import string
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
import numpy as np
import heapq
from textblob import TextBlob
dic={'n':['NN','NNS','NNP','NNPS'],'v':['VB','VBD','VBG','VBN','VBP','VBZ',],'a':['JJ','JJR','JJS '],'r':['RB','RBR','RBS']}
numbers=["01","02","03"]

f=open('ptack_5ek.txt','r+')
new=open('ptackcut完整.txt','w+')
# new=open('ptack_auto_14000保留標點','w+')
count=0
for i in f:                     #抓dic
    line=i.split('\t')[1]
    label=i.split('\t')[0]
    tokens = word_tokenize(line)
    tagged_sent = pos_tag(tokens)
    score_list = []
    word_list = []
    for i in tagged_sent:

        for k, v in dic.items():
            if i[1] in v:
                # print(k)
                score_save = []
                try:
                    for number in numbers:
                        try:
                            cc = i[0].rstrip('\n') + '.' + k + '.' + number
                            # print(cc)
                            single_score = swn.senti_synset(cc)
                            pos = re.search(r"=[-+]?[0-9]*\.?[0-9]*", str(single_score))
                            neg = re.search(r"[-+]?[0-9]*\.?[0-9]*>", str(single_score))
                            delete_score = float(pos.group().lstrip('=')) - float(neg.group().rstrip('>'))
                            score_save.append(delete_score)

                        except:
                            pass
                    if score_save:
                        minabs = abs(score_save[0])
                        minele = score_save[0]
                        for l in score_save:
                            if abs(l) > minabs:
                                minabs = abs(l)
                                minele = l
                        # print(minele)
                        if minele != 0:
                            # print(minele)
                            score_list.append(minele)
                            word_list.append((i[0]))
                            # print('score_list:'+score_list)
                except:
                    pass

    zidian = dict(zip(word_list, score_list))
    line = line.strip().strip(string.punctuation)       #去首尾標點
    line = line.strip().strip(string.punctuation)
    line = line.strip().strip(string.punctuation)
    if '.' in line :            #小數點換int
        c = re.findall(r"[-+]?[0-9]*\.[0-9]+", line)
        if c != []:
            for h in c:
                line = line.replace(h, str(int(float(h))))

    if ','in line or'.'in line or'!'in line or'?'in line or'#' in line or':' in line :  #最小部分加分數

        cutting = re.split(r'([,.:?!#])', line)
        cutting.append("")

        cutting = ["".join(i) for i in zip(cutting[0::2], cutting[1::2])]
        print(cutting)
        while ' ' in cutting:
            cutting.remove(' ')
        # print('看到符號切割:',cutting)
        alists=[]
        for subsentence in cutting:
            long=len(subsentence.strip().split(' '))*0.07
            sublist=[long]
            for scored in zidian:
                if scored in subsentence.rstrip('\n'):
                    sublist.append(zidian[scored])
            alists.append(sum(list(map(abs,sublist))))  #sum(list(map(abs,sublist
        print(alists)
        two_biggest=sorted(range(len(alists)), key=lambda i: alists[i])[-2:]
        # print(two_biggest)                      #最大兩個主體

        alist = two_biggest

        time = -1
        maxcutting = {}
        for c in range(1, len(cutting)):
            first = " ".join(cutting[:c])           #要加逗號！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1
            second = " ".join(cutting[c:])

            time = time + 1
            if time >= min(alist) and time < max(alist):
                # print(first + "\t\t\t\t" + second)
                cha = abs(TextBlob(first).sentiment.polarity - TextBlob(second).sentiment.polarity)
                maxcutting[first + "\t" + second] = cha
        # print('最大的：',max(maxcutting, key=maxcutting.get))
        new.write(label)
        new.write('\t')
        new.write(max(maxcutting, key=maxcutting.get))
        new.write('\n')







        # if '.'in line :
        #     count=count+1
        #     print(line)
        #     print(count)


    if ','not in line and'.'not in line and'!'not in line and'?'not in line and'#'not in line and':'  not in line :

        cut=line.split(' ')

        maxcutting = {}
        for c in range(1, len(cut)):
            first = " ".join(cut[:c])
            second = " ".join(cut[c:])



            cha = abs(TextBlob(first).sentiment.polarity - TextBlob(second).sentiment.polarity)
            maxcutting[first + "\t" + second] = cha
        print('最大的：', max(maxcutting, key=maxcutting.get))
        new.write(label)
        new.write('\t')
        new.write(max(maxcutting, key=maxcutting.get))
        new.write('\n')















