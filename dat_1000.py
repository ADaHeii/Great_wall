import csv

f = open(r'F:\zx\1.dat')
lines = f.readlines()
f.close()
# a = [i for i in range(4)]
a = [[0] * 4 for i in range(675000)]
i = 0
j = 0
t = 0
# for line in lines:
#     if len(line) == 96:
#         i += 1
# print(i)

for line in lines:
    t1 = 1
    t2 = 1
    j = 0
    if len(line) == 96:
        for x in line:
            if t1 == 0 and t2 == 1:  # 01 str-空
                j += 1
            t1 = t2
            if x != '\n' and x != ' ':  # 如果x为str，t2等于0
                t2 = 0
            else:
                t2 = 1  # 如果x为空，t2=1

            if t1 == 1 and t2 == 0:  # 10   空-str
                a[i][j] = x
            else:
                if t1 == 0 and t2 == 0:  # 00  str-str
                    a[i][j] += x
        i += 1
# for i in range(len(a)):
#     print(a[i])

for w in range(len(a)):
    for e in range(len(a[0])):
        a[w][e] = float(a[w][e]) / 1000
for i in range(len(a)):
    print(a[i])

f = open(r'F:\zx\1.csv', 'w', encoding='UTF8', newline='')
writer = csv.writer(f)
for i in range(len(a)):
    writer.writerow(a[i])
f.close()
