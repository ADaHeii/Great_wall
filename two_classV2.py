'''
V1相较于V0改进了Acc计算公式
V2相较于V1 增加了注释 删除了冗余测试代码 增强可读性
'''
import os
import xlrd
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import LSTM
import json
import shutil

out_sta = []
test_path = "D:\\python\\SD_Prediction_Model\\SDtest"
fig_path = "D:\\python\\SD_Prediction_Model\\testfig\\"
data_list = os.listdir(test_path)
print(data_list)
'''
加载模型和权重
'''
model = LSTM()
model.cuda()
checkpoint = torch.load("D:\\python\\SD_Prediction_Model\\Best140.ptm")
model.load_state_dict(checkpoint)
model.eval()
'''
定义全局变量 计算全局准确度（文件夹内所有数据准确度）
'''
loubao_file = []
wubao_file = []
# 测试集全局次数
sd_num = 0  # 砂堵次数
correct_num = 0  # 正确预警次数
fail_num = 0  # 漏警次数
pre_time = list()  # 相较于峰值点提前预警时间
data_num = 0  # 总样本数
xvjing_num = 0  # 虚警次数
Global_Pre = []
Global_L = []
Global_out = []
'''
准确度计算函数acc
acc中传入四个参数    pre是预测列标签 SDLabel是标注标签 
P是压力 P_Cycle_SD 是预测的砂堵起止点列表 用于计算虚警率
'''
def acc(pre, SDLabel, P, P_Cycle_SD):
    global sd_num, correct_num, fail_num, pre_time, xvjing_num
    xvjing_list = list()
    correct = 0
    fail = 0
    sd = 0
    SDrisk = []

    for i in range(1, len(SDLabel)):  # 0 1 2     000000111100020000001112222000002221110
        if SDLabel[i] > SDLabel[i - 1] and SDLabel[i] == 80:  # 转化点 风险提级  0 1 2
            SDrisk.append([i, -1])
        if SDLabel[i] < SDLabel[i - 1] and SDLabel[i] == 0:
            SDrisk[len(SDrisk) - 1][1] = i - 1
    for i in range(len(SDrisk)):
        sd += 1
        P_section = P[SDrisk[i][0]:SDrisk[i][1]]
        ind = P_section.index(max(P_section)) + SDrisk[i][0]
        predict = 0
        for k in range(SDrisk[i][0], ind):
            if pre[k] != 0:
                predict = 1
                break
        print(i, SDrisk[i][0], SDrisk[i][1], predict)
        if predict == 1:
            correct += 1
            for j in range(ind, 0, -1):
                if pre[j] == 0:
                    pre_time.append(ind - j)
                    break
        else:
            fail += 1

    xvjing = 0
    for i in range(len(P_Cycle_SD)):
        if 80 not in SDLabel[P_Cycle_SD[i][0]:P_Cycle_SD[i][1]]:
            xvjing += 1
            xvjing_list.append(P_Cycle_SD[i])  # [100,200]
    # 把当前段 正确、虚警、漏警添加到总数上
    xvjing_num += xvjing
    correct_num += correct
    fail_num += fail
    sd_num += sd
    # 如果当前段有砂堵标注 就计算准确率 没有就只计算虚警率
    if sd > 0:
        return correct / sd, fail / sd, xvjing, xvjing_list
    else:
        return -1, -1, xvjing, xvjing_list

'''
曲线绘图函数 把砂堵风险概率画在压力曲线上
'''
def Draw(P_Cycle, curvecolor):
    for i in range(len(P_Cycle) - 1):
        start = P_Cycle[i][0]
        end = P_Cycle[i][1]
        px = np.arange(start, end + 1)
        ax.plot(px, P[start:end + 1], linewidth=width, color=curvecolor)
    plt.savefig("D:\\python\\SD_Prediction_Model\\testfig\\" + file.split('.')[0] + '.png', dpi=500)




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
    cur_data = 0
    for row in data.get_rows():
        if row[6].value != '砂比' and row[6].value != '%':
            L.append(row[14].value)  # excel 里标签
            P.append(row[3].value)
            Q.append(row[5].value)
            S.append(row[6].value)
            cnt += 1
            # print(row[6].value)
            if row[6].value > 0 and flag == 0:
                flag = 1
                sand_start = cnt
            if flag == 1 and len(P) % 10 == 0 and len(P) >= 180 + sand_start:
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
                if out >= 0.4 and out < 0.6:
                    P_Cycle_Green.append([cnt, cnt + 10])
                if out >= 0.6 and out < 0.76:
                    P_Cycle_Orange.append([cnt, cnt + 10])
                if out >= 0.76 and out <= 1.0:
                    P_Cycle_Purple.append([cnt, cnt + 10])
                    # [100 110] [111 121]  [100 121]
    PreLabel = np.zeros(len(P))     #生成标签列数据写入excel指定列 （砂堵标签）
    data_num += cur_data
    '''
    P_Cycle_Purple 为所有预警值大于0.76的区间，对此区间内 砂堵标签赋值85
    P_Cycle_SD 合并后的砂堵区间起止点 长度为该段数据砂堵次数
    '''
    for j in range(len(P_Cycle_Purple)):
        PreLabel[P_Cycle_Purple[j][0]:(P_Cycle_Purple[j][1] + 1)] = 85
    for i in (1, len(PreLabel) - 1):
        if PreLabel[i] > PreLabel[i - 1]:  # 转化点 风险提级  0 1 2
            P_Cycle_SD.append([i, -1])
            print(len(P_Cycle_SD), P_Cycle_SD)
        if PreLabel[i] < PreLabel[i - 1]:
            P_Cycle_SD[len(P_Cycle_SD) - 1][1] = i - 1
            print(len(P_Cycle_SD), P_Cycle_SD)
    # print(len(P_Cycle_SD),P_Cycle_SD)
    # index = np.array(np.where(Label == 80))
    '''
    acc中pre是预测列标签 SDLabel是标注标签 
    P是压力 P_Cycle_SD 是预测的砂堵起止点列表 用于计算虚警率
    '''
    correct_rate, fail_rate, xvjing, xvjing_list = acc(PreLabel, L, P, P_Cycle_SD)

    Global_Pre.extend(PreLabel.tolist())
    Global_L.extend(L)

    if correct_rate != -1:
        print('井段：{}, 准确率：{}, 漏报率：{}, 虚警次数：{}'.format(file, correct_rate, fail_rate, xvjing))

    '''
    绘制施工曲线图代码
    下一步建议修改参数 合并进入Draw函数 
    规范入参 形成函数方便调用
    '''
    x = np.arange(0, len(P))
    width = 2
    # plt.figure(figsize=(16, 9), dpi=500)
    # plt.ylim((0, 100))
    # plt.rcParams['axes.grid'] = True
    fig, ax = plt.subplots(figsize=(16, 9), dpi=500)  # 设置图片比例与dpi
    fig.subplots_adjust(right=0.85)  # 设置图片左右位置
    ax.grid(True, color='grey')  # 设置网格线
    # 建立副坐标轴
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above. 调整twin2坐标轴位置
    twin2.spines.right.set_position(("axes", 1.08))
    # 分坐标轴画图
    p1, = ax.plot(x, P, linewidth=width, color="red", label="压力")
    p1, = ax.plot(x, L, linewidth=width, color='purple', alpha=0.6, label="标签")
    p2, = twin1.plot(x, Q, linewidth=width, color="blue", label="排量")
    p3, = twin2.plot(x, S, color="black", label="砂浓度")
    # 设置坐标轴刻度
    # ax.set_xlim(0, 2)
    ax.set_ylim(0, 100)
    twin1.set_ylim(0, 25)
    twin2.set_ylim(1, 20)
    # 设置坐标轴轴名与字体大小
    font = {'size': 15}
    ax.set_xlabel("时间 (s)", font)
    ax.set_ylabel("压力 (MPa)", font)
    twin1.set_ylabel("排量 ($m^3$/min)", font)
    twin2.set_ylabel("砂浓度 (%)", font)
    # 设置坐标轴颜色
    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())
    # 设置坐标轴字体颜色与刻度大小
    tkw = dict(size=6, width=2)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)
    # 设置图例名称 （位置？）
    # ax.legend(handles=[p1, p2, p3])
    # 保存图像 绘制图名 绘制风险段
    # plt.title('GreatWAll-Pressure Curve' + '-' + file.split('.')[0])
    plt.title('长城压裂施工曲线' + ' - ' + file.split('.')[0], font)
    Draw(P_Cycle_Orange, 'Orange')
    Draw(P_Cycle_Green, 'Green')
    Draw(P_Cycle_Purple, 'Purple')
    plt.close()

    datapath = "D:\\python\\SD_Prediction_Model\\testfig\\"
    savepath1 = "D:\\python\\SD_Prediction_Model\\loubao\\"
    savepath2 = "D:\\python\\SD_Prediction_Model\\wubao\\"

    if fail_rate > 0:
        loubao_file.append(file.split('.')[0] + '.png')
        fig = file.split('.')[0] + '.png'
        source = datapath + fig
        target = savepath1 + fig
        shutil.copyfile(source, target)
    if xvjing > 0.4:
        wubao_file.append(file.split('.')[0] + '.png')
        fig = file.split('.')[0] + '.png'
        source = datapath + fig
        target = savepath2 + fig
        shutil.copyfile(source, target)

ahead_time = np.mean(pre_time)      #计算平均提前预警时间
print('全局砂堵次数：{}, 正确预警次数：{}, 漏警次数：{}, 虚警次数：{}'.format(sd_num, correct_num, fail_num, xvjing_num))
print('分类总样本数：{}, 平均提前预警时间：{}'.format(data_num, ahead_time))
print('全局准确率：{}, 全局漏报率：{}, 全局虚井率：{}'.format(correct_num / sd_num, fail_num / sd_num,
                                                           xvjing_num / data_num))

dic = json.dumps(pre_time)
with open('D:\\python\\SD_Prediction_Model\\pretime.json', 'w', encoding='utf-8') as f:
    f.write(dic)

