import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# 获取指定路径下的所有 .txt 文件
def get_file_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]

# 解析出 .txt 图件文件的名称
def get_img_name_str(imgPath):
    return imgPath.split(os.path.sep)[-1]

flist = get_file_list('data')
dataNum = len(flist)
time = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(dataNum):
    imgNameStr = flist[i]
    imgName = get_img_name_str(imgNameStr)  # 得到 数字_实例编号.png
    # print("imgName: {}".format(imgName))
    classTag = imgName.split(".")[0].split("(")[0]  # 得到 类标签(数字)
    # print(classTag)
    if int(classTag)==20161017:#修改这里的值来得到每天的流量
        file = open("data/"+imgName)
        # f = open('making/' + imgName, 'w')
        timeTmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for line in file.readlines():
            intTime = 0
            str_after = re.sub(' +', ' ', line.strip())
            curLine = str_after.split(" ")
            floatLine = list(map(lambda x: x, curLine))
            # print(floatLine)
            intTime = (int((floatLine[1][0:floatLine[1].find(':')])))
            xplot=float(floatLine[3])
            yplot=float(floatLine[4])
            # f.write(floatLine[3]+','+floatLine[4]+','+floatLine[7]+','+floatLine[8]+'\n')
            # print(xplot)
            # print(yplot)
            ## 这里就是限定范围的地方！！！！！！！！！！！！ 合理的trick
            if((xplot>114.287 and xplot<114.3 and yplot<30.559 and yplot>30.552)):
                # intTime = map(int, floatLine[1][0:floatLine[1].find(':')])
                # print(intTime)
                if (float(floatLine[3]) - float(floatLine[5]) < 0.04):
                    timeTmp[intTime - 1] = 1
        for i in range(len(time)):
            time[i]+=timeTmp[i]
        # f.close()
print(time)




