import pandas as pd
import ast

df = pd.read_csv("/home/s1/sungmokjung/KoNUBench/neg_stat_final.csv")

SHORT_AHN = 0
LONG_AHN = 0
SHORT_MOT = 0
LONG_MOT =0 
MALDA = 0

all = ["안 단형", "안 장형", "못 단형", "못 장형", "말다 부정"]
all2 = ["관형절", "부사절", "명사절", "인용절", "종속절", "주절", "확인형"]
for cell in df["안 단형"]:
    lst = ast.literal_eval(cell)
    SHORT_AHN += len(lst)

for cell in df["안 장형"]:
    lst = ast.literal_eval(cell)
    LONG_AHN += len(lst)

for cell in df["못 단형"]:
    lst = ast.literal_eval(cell)
    SHORT_MOT += len(lst)

for cell in df["못 장형"]:
    lst = ast.literal_eval(cell)
    LONG_MOT += len(lst)

for cell in df["말다 부정"]:
    lst = ast.literal_eval(cell)
    MALDA += len(lst)
print(f'안 단형: {SHORT_AHN}')
print(f'안 장형: {LONG_AHN}')
print(f'못 단형: {SHORT_MOT}')
print(f'못 장형: {LONG_MOT}')
print(f'말다: {MALDA}')

for idx, row in df.iterrows():
    total = 0
    for each in all:
        total += len(ast.literal_eval(row[each]))
    total2 =0
    for each2 in all2:
        val = row[each2]
        if pd.isna(val):
            continue
        total2 += int(row[each2])

    
    if not (total == total2):
        print(idx)
        print(row["sentence"])