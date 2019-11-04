from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
import torch
from setting import getresult, LSTM

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.device = 'cuda'


args.x_frame = 168
args.model = 'model.h5' #model name to save and load
args.problem_data = '2nd_data.csv' #path to problem ace data


# 2차대회 문제 불러오기 및 가공
ace_data_to_pred = pd.read_csv(args.problem_data)
ace_pred_data = pd.DataFrame()

for j in range(1, 731):
    ace_group_doy = ace_data_to_pred.loc[(ace_data_to_pred['doy'] == j)]

    for k in range(0, 8):
        ace_group_hr = ace_group_doy.loc[(ace_group_doy['hr'] >= 3 * k) & (ace_group_doy['hr'] < (3 * k + 3))]

        if len(ace_group_hr) >= 168:
            ace_group_hr = ace_group_hr.iloc[:168, :]
            ace_pred_data = ace_pred_data.append(ace_group_hr)

        else:
            ace_pred_data = ace_pred_data.append(ace_group_hr)

ace_pred_data = ace_pred_data[['Vp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt']]
print(len(ace_pred_data))

f_r = []
resultdata = getresult(ace_pred_data, f_r, args.x_frame)
partitions = {'result': resultdata }

# 모델 불러오기(h5 파일 불러오기)
device = torch.device('cuda')
model = LSTM(5, 70, 1, 4, 1, False, True)
resultloader = DataLoader(partitions['result'], batch_size=1, shuffle=False, drop_last=False)
model.load_state_dict(torch.load(args.model))
model.cuda()
model.eval()

# 2년치 예측값(kp) 산출

result = []
with torch.no_grad():
    for i, (X,y) in enumerate(resultloader):

        X = X.transpose(0, 1).float().to(device)

        model.hidden = [hidden.to(device) for hidden in model.init_hidden()]

        y_pred = model(X).to('cpu')

        y_pred = float(y_pred)
        if y_pred - int(y_pred) < 0.5:
            y_pred = int(y_pred)
        else:
            y_pred = int(y_pred) + 1
        result.append(y_pred)

result = [np.nan] + result[:-1]

r = np.array(result).reshape(-1,8)
r = pd.DataFrame(r)
r.to_csv('result.csv', mode = 'w', index= False, header=False)