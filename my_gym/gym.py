import sqlite3
import numpy as np
import pandas as pd
from sklearn import preprocessing


# import하면 곧바로 initialize
cnx = sqlite3.connect('kospi-201704.sqlite')

# ratio, amount, ends, foreigner, insti, person, program, credit
columns = ['ratio', 'amount', 'ends', 'foreigner', 'insti', 'person', 'program', 'credit']
input_size = len(columns) + 1

# 0: buy
# 1: sell
# 2: do nothing
action_space = [0, 1, 2]

# training, test set 비율
TRAIN_DATA_RATIO = 0.8

index = 0
earn = 0 #번돈
deposit = 1 #살수 있는 수량
buy = 0 #산 수량
buy_cost = 0 #산 가격
isTestMode = False

def load(code):
    global train, test

    # 한 종목씩 select
    df = pd.read_sql_query('SELECT * from stocks where code="'+code+'"', cnx)

    # column 정리
    df['ratio'] = df['ratio'].astype(float)
    df['diff'] = df['ratio'].astype(float)
    df['amount'] = df['ratio'].astype(float)
    df['start'] = df['start'].apply(lambda x: x.replace('+', '').replace('-', '')).astype(float)
    df['ends'] = df['ends'].apply(lambda x: x.replace('+', '').replace('-', '')).astype(float)
    df['high'] = df['high'].apply(lambda x: x.replace('+', '').replace('-', '')).astype(float)
    df['low'] = df['low'].apply(lambda x: x.replace('+', '').replace('-', '')).astype(float)
    df['foreigner'] = df['foreigner'].apply(lambda x: x.replace('++', '+').replace('--', '-')).astype(float)
    df['insti'] = df['insti'].apply(lambda x: x.replace('++', '+').replace('--', '-')).astype(float)
    df['person'] = df['person'].apply(lambda x: x.replace('++', '+').replace('--', '-')).astype(float)
    df['program'] = df['program'].apply(lambda x: x.replace('++', '+').replace('--', '-')).astype(float)
    df['credit'] = df['credit'].astype(float)

    # normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    df[['ends', 'amount', 'foreigner', 'insti', 'person', 'program', 'credit']] = min_max_scaler.fit_transform(
        df[['ends', 'amount', 'foreigner', 'insti', 'person', 'program', 'credit']])

    div_point = int(len(df)*TRAIN_DATA_RATIO)
    print('Training set:', div_point, 'of', len(df))
    # training set - 초반의 ratio 만큼
    train = df[:div_point]
    # test set - 후반 나머지
    test = df[div_point+1:]

" isTest: training모드일때는 false, test모드일때는 true"
def reset(isTest=False):
    global index, earn, deposit, buy, isTestMode
    index = 0
    earn = 0
    deposit = 1
    buy = 0
    isTestMode = isTest

    if isTestMode:
        cur = test.iloc[index]
    else:
        cur = train.iloc[index]
    # ratio, amount, ends, foreigner, insti, person, program, credit
    state = []
    for col in columns:
        state.append( cur[col] )
    state.append( deposit )

    return state

# state, reward, done, _
def step(action):
    global index, earn, deposit, buy, buy_cost
    reward = 0
    done = False

    if isTestMode:
        data = test
    else:
        data = train

    try:
        index += 1
        cur = data.iloc[index]
    except IndexError:
        done = True
        cur = data.iloc[index-1]
    else:
        # 현재는 1주씩 살 수만 있도록 한다
        if action==0: #buy
            if deposit>0:
                deposit -= 1
                buy += 1
                buy_cost = cur['ends']
            else:
                # 살 수 없는 경우에 대한 penalty
                #reward = -.1
                pass
        elif action==1: #sell
            if buy>0:
                deposit += 1
                buy -= 1
                reward = cur['ends'] - buy_cost
            else:
                # 팔수 없는 경우에 대한 penalty
                #reward = -.1
                pass

    # ratio, amount, ends, foreigner, insti, person, program, credit
    state = []
    for col in columns:
        state.append( cur[col] )
    state.append(deposit)

    # if isTestMode:
    #     print( index, action, reward )

    # earn : total reward
    #earn += reward

    # 너무 많이 잃으면 그만 두게 할까?
    # if earn<-1:
    #     reward = -100
    #     done = True

    return [state, reward, done, None]

"""get all stock codes"""
def getAllCodes():
    res = []
    df = pd.read_sql_query('SELECT * from stocks', cnx)
    codes = df.groupby('code').size()
    for i in range(len(codes)):
        size = codes.iloc[i]
        code = codes.index[i]

        # size가 어느정도 되는 것들만...
        if size>500:
            res.append(code)
    return res

