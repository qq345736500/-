import string
import re
from textblob import TextBlob
f=open('ptack_5ek.txt','r+')

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

    if ','not in line and'.'not in line and'!'not in line and'?'not in line and'#'not in line and':'  not in line :
        # cutting = re.split(r'([,.:?!#])', line)
        # cutting.append("")
        # cutting = ["".join(i) for i in zip(cutting[0::2], cutting[1::2])]

        while ' ' in cutting:
            cutting.remove(' ')

        maxcutting = {}
        for c in range(1, len(cutting)):
            first = " ".join(cutting[:c])  # 要加逗號！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1
            second = " ".join(cutting[c:])
            print(first+'\t\t'+second)

            cha = abs(TextBlob(first).sentiment.polarity - TextBlob(second).sentiment.polarity)
            maxcutting[first + "\t" + second] = cha
        print('最大的：', max(maxcutting, key=maxcutting.get))


        # cut=line.split(' ')
        #
        # maxcutting = {}
        # for c in range(1, len(cut)):
        #     first = " ".join(cut[:c])
        #     second = " ".join(cut[c:])
        #     print(first + "\t\t" + second)
        #
        #
        #     cha = abs(TextBlob(first).sentiment.polarity - TextBlob(second).sentiment.polarity)
        #     maxcutting[first + "\t" + second] = cha
    # print('最大的：', max(maxcutting, key=maxcutting.get))
    #     new.write(label)
    #     new.write('\t')
    #     new.write(max(maxcutting, key=maxcutting.get))
    #     # new.write('\t')
    #     # new.write(line2)
    #     # new.write('\t')
    #     # new.write(line)
    #     new.write('\n')

