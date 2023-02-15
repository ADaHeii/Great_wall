import os
import xlrd
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import LSTM
import json
import shutil
out_sta = []
test_path = "/share/home/jinshengxu/SD/GreateWall/20221030/MHSDLabel/"
# test_path = 'G:\\GWSDtest\\'
data_list = os.listdir(test_path)
print(data_list)
model = LSTM()
model.cuda()
checkpoint = torch.load("/share/home/jinshengxu/SD/GreateWall/20221030/LSTM+FC32.ptm")
# checkpoint = torch.load('D:\\GWptm\\LSTM+FC32.ptm')
model.load_state_dict(checkpoint)
model.eval()
loubao_file=[]
wubao_file=[]
def acc (out,label):
    correct_num = 0
    one_zero_num = 0
    zero_one_num = 0
    one_num = 0
    zero_num = 0
    TP = 0
    TN = 0
    num = len(label)
    for i in range(len(out)):
        if out[i] > 0:
            if label[i] == 80:
                correct_num += 1
                one_num += 1
                TP += 1
            else:
                zero_one_num += 1
                zero_num += 1
        else:
            if label[i] == 80:
                one_zero_num += 1
                one_num += 1
            else:
                correct_num += 1
                zero_num += 1
                TN += 1
    correct_rate = correct_num / num  #准确率
    if one_num == 0:
        one_zero_rate=0
    else:
        one_zero_rate = one_zero_num / one_num  #漏报
    zero_one_rate = zero_one_num / zero_num  #误报

    return correct_rate, one_zero_rate, zero_one_rate

init = {'Frac_stage': 0,
         'TP': 0,
         'TN': -1,
         'FP': -1,
         'FN': -1}
def Draw(P_Cycle,curvecolor):
    for i in range(len(P_Cycle) - 1):
        start = P_Cycle[i][0]
        end = P_Cycle[i][1]
        px = np.arange(start, end + 1)
        plt.plot(px, P[start:end + 1], linewidth=width, color=curvecolor)
    plt.savefig("/share/home/jinshengxu/SD/GreateWall/20221030/draw_fig/" + file.split('.')[0] + '.png', dpi=500)

Global_Pre=[]
Global_L=[]
Global_out=[]
for file in data_list:
    print(file)
    xl = xlrd.open_workbook(test_path + '/' + file)
    data = xl.sheet_by_index(0)
    flag = 0
    cnt = -1
    sand_start = 0
    P_Cycle_Green = list()
    P_Cycle_Orange = list()
    P_Cycle_Purple = list()
    # P_Cycle.append([0, -1])
    # P_Cycle.append([0, -1])
    L = list()
    P = list()  # 压力
    Q = list()  # 排量
    S = list()  # 砂浓度
    lstm_result = list()
    Inputdata = [[[0] * 3 for _ in range(180)]]
    # print((np.array(Inputdata)).shape)
    pre = list()  # 预警砂堵概率
    for row in data.get_rows():
        if row[4].value != 'S' and row[4].value != '%':
            L.append(row[1].value) #excel 里标签
            P.append(row[2].value)
            Q.append(row[3].value)
            S.append(row[4].value)
            cnt += 1
            if row[4].value > 0 and flag == 0:
                flag = 1
                sand_start = cnt
            if flag == 1 and len(P)%10==0 and len(P) >= 180+sand_start:
                for i in range(cnt - 180, cnt):
                    Inputdata[0][i - (cnt - 180)][0] = P[i]
                    Inputdata[0][i - (cnt - 180)][1] = Q[i]
                    Inputdata[0][i - (cnt - 180)][2] = S[i]
                input = torch.tensor(Inputdata, dtype=torch.float)
                input = input.cuda()
                out = model(input)
                Global_out.append(out.tolist())
                # out_sta.append()
                # print(out)  # 输出预警概率
                if out >= 0.4 and out< 0.6:
                    P_Cycle_Green.append([cnt, cnt+10])
                if out >= 0.6 and out< 0.76:
                    P_Cycle_Orange.append([cnt, cnt+10])
                if out >= 0.76 and out<= 1.0:
                    P_Cycle_Purple.append([cnt, cnt+10])
                    # [100 110] [111 121]  [100 121]
    PreLabel = np.zeros(len(P))
    for j in range(len(P_Cycle_Purple)):
        PreLabel[P_Cycle_Purple[j][0]:(P_Cycle_Purple[j][1]+1)] = 85
    # index = np.array(np.where(Label == 80))
    correct_rate, one_zero_rate, zero_one_rate = acc(PreLabel, L)

    Global_Pre.extend(PreLabel.tolist())
    Global_L.extend(L)
    # print(file, correct_rate, one_zero_rate, zero_one_rate)
    print('井段：{}, 准确率：{}, 漏报率：{}, 误报率：{}'.format(file, correct_rate, one_zero_rate, zero_one_rate))
    print(len(L), len(PreLabel))


    # Judge = []
    # for k in range(len(PreLabel)):
    #     # TP
    #     if PreLabel[k] == 85 and L[k] == 80:
    #         Judge.append(1)
    #     # FP
    #     if PreLabel[k] == 85 and L[k] == 0:
    #         Judge.append(2)
    #     # FN
    #     if PreLabel[k] == 0 and L[k] == 80:
    #         Judge.append(3)
    #     # TN
    #     if PreLabel[k] == 0 and L[k] == 0:
    #         Judge.append(0)

    x = np.arange(0, len(P))
    width = 2
    plt.figure(figsize=(16, 9))
    plt.plot(x, P, linewidth=width, color='red')
    plt.plot(x, Q, linewidth=width, color='blue')
    plt.plot(x, [i for i in S], linewidth=width, color='green')
    plt.plot(x, L, linewidth=0.8, color='purple', alpha=0.6)
    # plt.plot(x, PreLabel, linewidth=0.8, color='blue', alpha=0.6)
    plt.title(file.split('.')[0])
    Draw(P_Cycle_Orange, 'Orange')
    Draw(P_Cycle_Green, 'Green')
    Draw(P_Cycle_Purple, 'Purple')

    datapath = "/share/home/jinshengxu/SD/GreateWall/20221030/SDtest/"
    savepath1 = '/share/home/jinshengxu/SD/GreateWall/20221030/loubao/'
    savepath2 = '/share/home/jinshengxu/SD/GreateWall/20221030/wubao/'

    if one_zero_rate > 0.4:
        loubao_file.append(file.split('.')[0] + '.png')
        fig = file.split('.')[0] + '.png'
        source = datapath + fig
        target = savepath1 + fig
        shutil.copyfile(source, target)
    if zero_one_rate > 0.4:
        wubao_file.append(file.split('.')[0] + '.png')
        fig = file.split('.')[0] + '.png'
        source = datapath + fig
        target = savepath2 + fig
        shutil.copyfile(source, target)



print('全局预测长度：{}, 全局标签长度：{}'.format(len(Global_Pre), len(Global_L)))
correct_rate, one_zero_rate, zero_one_rate = acc(Global_Pre, Global_L)
print('全局准确率：{}, 全局漏报率：{}, 全局误报率：{}'.format(correct_rate, one_zero_rate, zero_one_rate))

dic = json.dumps(Global_out)
with open("/share/home/jinshengxu/SD/GreateWall/20221030/draw_fig/" + 'Global_out' + '.json', "w", encoding='utf-8') as f:
    f.write(dic)




    # for k in range(len(Judge)):
    #     if Judge[k] == 1:
    #         plt.plot(k, P[k], linewidth=width, color='black')
    #     if Judge[k] == 2:
    #         plt.plot(k, P[k], linewidth=width, color='orange')
    #     if Judge[k] == 2:
    #         plt.plot(k, P[k], linewidth=width, color='purple')
    #
    # plt.savefig('D:\\GreatWall\\20221021\\classify\\' + file.split('.')[0] + '.png', dpi=500)

    # for k in range(len(Judge)):
    #     if Judge[k] == 1:
    #         plt.plot(k, P[k], linewidth=width, color='red', label='预警')
    #     if Judge[k] == 2:
    #         plt.plot(k, P[k], linewidth=width, color='orange', label='虚警')
    #     if Judge[k] == 2:
    #         plt.plot(k, P[k], linewidth=width, color='purple', label='漏警')
    #     plt.legend()

    # for i in range(len(P_Cycle) - 1):
    #     start = P_Cycle[i][0]
    #     end = P_Cycle[i][1]
    #     px = np.arange(start, end + 1)
    #     plt.plot(px, P[start:end + 1], linewidth=width, color='royalblue')
    # plt.savefig('D:\\GreatWall\\20220929\\GWbestpara\\' + file.split('.')[0] + '.png', dpi=500)

'''
datapath = 'D:\\GreatWall\\20221021\\32\\'
savepath1 = 'D:\\GreatWall\\20221021\\wubao\\'
savepath2 = 'D:\\GreatWall\\20221021\\loubao\\'

for fig in wubao_file:
    source = datapath + fig
    target = savepath1 + fig
    shutil.copyfile(source, target)

for fig in loubao_file:
    source = datapath + fig
    target = savepath2 + fig
    shutil.copyfile(source, target)
'''





