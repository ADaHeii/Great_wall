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
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
plt.rcParams['figure.figsize'] = (14.0, 8.0)

warnings.filterwarnings('ignore')

out_sta = []
test_path = "/home/jin/桌面/data/data_2/test/"
# test_path = 'G:\\GWSDtest\\'
data_list = os.listdir(test_path)
model = LSTM()
model.cuda()
checkpoint = torch.load("/home/jin/桌面/data/data_2/para/GWbestpara/Best113.ptm")
# checkpoint = torch.load('D:\\GWptm\\LSTM+FC32.ptm')
model.load_state_dict(checkpoint)
model.eval()
loubao_file = []
wubao_file = []
#测试集全局次数
sd_num = 0 #砂堵次数
correct_num = 0 #正确预警次数
fail_num = 0 #漏警次数
pre_time = list() #相较于峰值点提前预警时间
data_num = 0 #总样本数
xvjing_num = 0 #虚警次数
# def acc (out,label):
#     TP = 0
#     abnormal = 0
#     num = len(label)
#     for i in range(num):
#         if int(label[i]) != 0:
#             abnormal += 1
#             if int(out[i]) == int(label[i]):
#                 TP = TP+1
#     if abnormal==0:
#         return abnormal
#     else:
#         acc = TP / abnormal  #准确率
#         return acc

def acc (pre, SDLabel, P, P_Cycle_SD):
    global sd_num, correct_num, fail_num, pre_time, xvjing_num
    xvjing_list = list()
    correct = 0
    fail = 0
    sd = 0
    Label = list()
    SDlow = []
    SDhigh = []
    for i in range(1, len(SDLabel)):   # 0 1 2     000000111100020000001112222000002221110
        if SDLabel[i] != SDLabel[i-1] and SDLabel[i] > SDLabel[i-1]:  # 转化点 风险提级  0 1 2
            if SDLabel[i] == 1:     # 0-1 start
                SDlow.append([i, -1])

            if SDLabel[i] == 2:     # 0-2/1-2 start
                SDhigh.append([i, -1])
                if SDLabel[i-1] == 1:   # 2-1 end
                    SDlow[len(SDlow)-1][1] = i-1

        if SDLabel[i] != SDLabel[i - 1] and SDLabel[i] < SDLabel[i - 1]:  #转化点 风险降级
            if SDLabel[i-1] == 1:   # 1-0 end
                SDlow[len(SDlow)-1][1] = i - 1

            if SDLabel[i-1] == 2:   # 2-0/2-1 end
                SDhigh[len(SDhigh)-1][1] = i - 1
                if SDLabel[i] == 1:  # 2-1 start
                    SDlow.append([i, -1])

    #统一认为是砂堵，合并进 Label
    for i in range(len(SDlow)):
        Label.append([SDlow[i][0], SDlow[i][1]])
    for i in range(len(SDhigh)):
        Label.append([SDhigh[i][0], SDhigh[i][1]])
#[ 1000,1500] 1300
    for i in range(len(Label)):
        sd += 1
        ind = P.index(max(P[Label[i][0]:P[Label[i][1]]]))
        if pre[ind] == 1 or pre[ind] == 2:
            correct += 1
            for j in range(ind, 0, -1):
                if pre[j] < 1:
                    pre_time.append(ind - j)
                    break
        else:
            fail += 1

    xvjing = 0
    for i in range(len(P_Cycle_SD)):
        if 1 or 2 not in L[P_Cycle_SD[i][0]:P_Cycle_SD[i][1]]:
            xvjing += 1
            xvjing_list.append(P_Cycle_SD[i])  # [100,200]
    #把当前段 正确、虚警、漏警添加到总数上
    xvjing_num += xvjing
    correct_num += correct
    fail_num += fail
    sd_num += sd
    # 如果当前段有砂堵标注 就计算准确率 没有就只计算虚警率
    if sd > 0:
        return correct / sd, fail / sd, xvjing, xvjing_list
    else:
        return -1, -1, xvjing, xvjing_list

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
    plt.savefig("/home/jin/桌面/data/data_2/draw/" + file.split('.')[0] + '.png', dpi=500)

Global_Pre=[]
Global_L=[]
Global_out=[]
for file in data_list:
    xl = xlrd.open_workbook(test_path + '/' + file)
    data = xl.sheet_by_index(0)
    flag = 0
    cnt = -1
    sand_start = 0
    P_Cycle_Green = list()
    P_Cycle_Orange = list()
    P_Cycle_Purple = list()
    P_Cycle_SD = list()
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
    cur_data = 0  #当前段样本数
    for row in data.get_rows():
        if row[2].value != '砂浓度' and row[2].value != '%':
            L.append(row[5].value) #excel 里标签
            P.append(row[0].value)
            Q.append(row[1].value)
            S.append(row[2].value)
            cnt += 1
            if row[2].value > 0 and flag == 0:
                flag = 1
                sand_start = cnt
            if flag == 1 and len(P)%10==0 and len(P) >= 180+sand_start:
                cur_data += 1
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
                mask = (out == out.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
                if mask[0].equal(torch.tensor([1, 0, 0],dtype=torch.int32).cuda()):
                    P_Cycle_Green.append([cnt,cnt+10])
                if mask[0].equal(torch.tensor([0, 1, 0],dtype=torch.int32).cuda()) and mask[0][1]>0.95:
                    P_Cycle_Orange.append([cnt,cnt+10])
                if mask[0].equal(torch.tensor([0, 0, 1],dtype=torch.int32).cuda()) and mask[0][2]>0.95:
                    P_Cycle_Purple.append([cnt,cnt+10])
    PreLabel = np.zeros(len(P))
    data_num += cur_data

    for j in range(len(P_Cycle_Purple)): # 预测出来的高风险起止点区间
        PreLabel[P_Cycle_Purple[j][0]:(P_Cycle_Purple[j][1]+1)] = 2
        P_Cycle_SD.append([P_Cycle_Purple[j][0],P_Cycle_Purple[j][1]])

    for j in range(len(P_Cycle_Orange)): # 预测出来的低风险起止点区间   P_Cycle_SD
        PreLabel[P_Cycle_Orange[j][0]:(P_Cycle_Orange[j][1]+1)] = 1
        P_Cycle_SD.append([P_Cycle_Orange[j][0], P_Cycle_Orange[j][1]])

    # index = np.array(np.where(Label == 80))
    # accuracy = acc(PreLabel, L)
    # 准确率 漏警率 虚警率
    correct_rate, fail_rate, xvjing, xvjing_list = acc(PreLabel, L, P, P_Cycle_SD)
    print('井段：{}, 准确率：{}, 漏报率：{}, 虚井率：{}'.format(file, correct_rate, fail_rate, xvjing / cur_data))

    # sns.set()
    C2 = confusion_matrix(L, PreLabel, labels=[0, 1, 2])
    print(C2)
    # fig = sns.heatmap(C2, annot=True)
    # heatfig = fig.get_figure()
    # heatfig.savefig('/home/jin/桌面/data/data_1/draw/heatmap_'+file.split('.')[0]+'.png',dpi=500)

    Global_Pre.extend(PreLabel.tolist())
    Global_L.extend(L)
    # print(file, correct_rate, one_zero_rate, zero_one_rate)
    # print('井段：{}, 准确率：{}'.format(file, accuracy))
    # print(len(L), len(PreLabel))


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


    L_mid = []
    L_high = []
    L_new = []
    for i in L:
        if i == 2:
            L_high.append(85)
            L_mid.append(0)
            L_new.append(0)
        elif i == 1:
            L_high.append(0)
            L_mid.append(85)
            L_new.append(0)
        else:
            L_high.append(0)
            L_mid.append(0)
            L_new.append(0)

    x = np.arange(0, len(P))
    width = 1.5
    # plt.figure(figsize=(16, 9))
    # plt.ylim((0, 100))
    # plt.grid()
    # plt.plot(x, P, linewidth=width, color='red')
    # plt.plot(x, [i * 4 for i in Q], linewidth=width, color='blue')
    # plt.plot(x, [i * 5 for i in S], linewidth=width, color='black')
    # plt.plot(x, L_high, linewidth=0.8, color='purple', alpha=0.6)
    # plt.plot(x, L_mid, linewidth=0.8, color='orange', alpha=0.6)
    # # plt.plot(x, L_new, linewidth=0.8, color='black', alpha=0.6)
    # # plt.plot(x, PreLabel, linewidth=0.8, color='blue', alpha=0.6)
    # plt.title(file.split('.')[0])
    # Draw(P_Cycle_Orange, 'Orange')
    # # Draw(P_Cycle_Green, 'Green')
    # Draw(P_Cycle_Purple, 'Purple')

    fig, ax = plt.subplots()
    ax2, ax3 = ax.twinx(), ax.twinx()
    rspine = ax3.spines['right']
    rspine.set_position(('axes', 1.125))
    ax3.patch.set_visible(False)
    fig.subplots_adjust(right=0.85)

    ax.plot(x, P, linewidth=width, color='red')
    ax.set_ylim([0,100])
    ax.set_ylabel('P(MPa)', c='red')
    ax.tick_params(axis='y', colors='red')
    ax.grid()
    ax2.plot(x, Q, linewidth=width, color='blue')
    ax2.set_ylim([0,25])
    ax2.set_ylabel('Q(m3/min)',c='blue')
    ax2.tick_params(axis='y', colors='blue')
    ax3.plot(x, S, linewidth=width, color='black')
    ax3.set_ylim([0,20])
    ax3.set_ylabel('S(%)',c='black')
    ax3.tick_params(axis='y', colors='black')
    ax.plot(x, L_high, linewidth=0.8, color='purple', alpha=0.6)
    ax.plot(x, L_mid, linewidth=0.8, color='orange', alpha=0.6)

    for i in range(len(P_Cycle_Purple) - 1):
        start = P_Cycle_Purple[i][0]
        end = P_Cycle_Purple[i][1]
        px = np.arange(start, end + 1)
        ax.plot(px, P[start:end + 1], linewidth=width, color='Purple')

    for i in range(len(P_Cycle_Orange) - 1):
        start = P_Cycle_Orange[i][0]
        end = P_Cycle_Orange[i][1]
        px = np.arange(start, end + 1)
        ax.plot(px, P[start:end + 1], linewidth=width, color='Orange')

    plt.title(file.split('.')[0])
    plt.savefig("/home/jin/桌面/data/data_2/draw/" + file.split('.')[0] + '.png', dpi=500)


    datapath = '/share/home/jinshengxu/SD_classification/draw_acc/'
    # savepath1 = 'D:\\GreatWall\\20221021\\loubao\\'
    # savepath2 = 'D:\\GreatWall\\20221021\\wubao\\'

    # if one_zero_rate > 0.4:
    #     loubao_file.append(file.split('.')[0] + '.png')
    #     fig = file.split('.')[0] + '.png'
    #     source = datapath + fig
    #     target = savepath1 + fig
    #     shutil.copyfile(source, target)
    # if zero_one_rate > 0.4:
    #     wubao_file.append(file.split('.')[0] + '.png')
    #     fig = file.split('.')[0] + '.png'
    #     source = datapath + fig
    #     target = savepath2 + fig
    #     shutil.copyfile(source, target)


print('全局准确率：{}, 全局漏报率：{}, 全局虚井率：{}'.format(correct_num / sd_num, fail_num / sd_num, xvjing_num / data_num))
# print('全局预测长度：{}, 全局标签长度：{}'.format(len(Global_Pre), len(Global_L)))
# correct_rate, one_zero_rate, zero_one_rate = acc(Global_Pre, Global_L)
# print('全局准确率：{}'.format(acc))

# dic = json.dumps(Global_out)
# with open('/share/home/jinshengxu/SD_classification/draw_acc/' + 'Global_out' + '.json', "w", encoding='utf-8') as f:
#     f.write(dic)




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





