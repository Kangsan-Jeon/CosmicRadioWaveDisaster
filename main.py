import pandas as pd
import numpy as np
import os
from tqdm import trange
import argparse
from copy import deepcopy
import torch
from setting import Ace_Dataset, experiment
from sklearn.impute import IterativeImputer


def main():
    seed = 100
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = 'cuda'

    args.batch_size = 30
    args.x_frame = 168
    args.kp_data_path = 'kp_data_training.csv'
    args.dataset = './dataset'  # path to training dataset folder
    args.model = 'model.h5'  # model name to save and load
    args.problem_data = '2nd_data.csv'  # path to problem ace data
    args.input_dim = 5
    args.output_dim = 1
    args.hid_dim = 70
    args.n_layers = 4

    args.dropout = 0.0
    args.use_bn = True

    args.epoch = 82

    # 데이터 불러오기
    ace_data = pd.DataFrame()
    ace_list = os.listdir(args.dataset)

    # 학습 데이터 생성(데이터 가공 등)
    for i in ace_list:
        ace_data = ace_data.append(pd.read_csv('./dataset/' + i))

    ace_data = ace_data.drop('Np', axis=1)
    ace_data[ace_data < -9000] = np.nan

    imp = IterativeImputer(sample_posterior=True, initial_strategy='median')
    imp.fit(ace_data)
    ace_data = imp.transform(ace_data)
    ace_data = pd.DataFrame(ace_data)
    ace_data.set_axis(['year', 'doy', 'hr', 'min', 'Vp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt'], axis=1, inplace=True)

    ace_final_data = pd.DataFrame()
    for i in trange(1999, 2014):
        ace_group_year = ace_data.loc[(ace_data['year'] == i)]

        for j in range(1, 367):
            ace_group_doy = ace_group_year.loc[(ace_group_year['doy'] == j)]

            for k in range(0, 8):
                ace_group_hr = ace_group_doy.loc[(ace_group_doy['hr'] >= 3 * k) & (ace_group_doy['hr'] < (3 * k + 3))]

                if len(ace_group_hr) >= 168:
                    ace_group_hr = ace_group_hr.iloc[:168, :]
                    ace_final_data = ace_final_data.append(ace_group_hr)


                elif len(ace_group_hr) == 168:
                    ace_final_data = ace_final_data.append(ace_group_hr)

    # Train 및 validation 셋 분류
    trainset = Ace_Dataset(ace_final_data, args.kp_data_path, args.x_frame, 1999, 2010)
    valset = Ace_Dataset(ace_final_data, args.kp_data_path, args.x_frame, 2011, 2012)
    testset = Ace_Dataset(ace_final_data, args.kp_data_path, args.x_frame, 2013, 2013)
    partition = {'train': trainset, 'val': valset, 'test': testset}

    # Train and test
    experiment(partition, deepcopy(args))

if __name__ == "__main__":
    main()