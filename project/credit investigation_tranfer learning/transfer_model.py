import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc,roc_auc_score
import xgboost as xgb
from sklearn.cross_validation import train_test_split

def get_discrete_variables(df):
    columns = []
    features = df.columns
    for feature in features:
        bins = len(df.groupby(feature).count())
        if bins < 20: #10,15都一样
            not_null = np.sum([df[feature] != -999] )
            if not_null < len(df) * 0.1:
                continue
            columns.append(feature)
    return  columns


def create_dummy_variables(df):
    columns = get_discrete_variables(df)
    for col in columns:
        df = pd.concat((df ,pd.get_dummies(df[col],prefix = col)), axis=1)
        del df[col]
    new_columns = df.columns
    for column in new_columns:
        if '999' in column:
            del df[column]
    return df

def choose_clolumns(df):
    columns = [x for x in b_train_x.columns if x in b_test.columns and x in a_train.columns]
    cols = df.columns
    for column in cols:
        if column not in columns:
            del df[column]
    return df



if __name__ == '__main__':
    a_train = pd.read_csv('A_train.csv')
    b_train = pd.read_csv('B_train.csv')
    b_test = pd.read_csv('B_test.csv')

    a_train = a_train.fillna(-999)
    b_train = b_train.fillna(-999)
    b_test = b_test.fillna(-999)

    a_train_y = a_train['flag']
    b_train_y = b_train['flag']

    a_train_x = a_train.drop(['no','flag'],axis=1)
    b_train_x = b_train.drop(['no','flag'],axis=1)

    submit = pd.DataFrame(b_test['no'])
    b_test.drop('no',axis=1,inplace=True)



    print('creating dummy variables\n ')
    a_train_x = choose_clolumns(a_train_x)
    b_train_x = choose_clolumns(b_train_x)
    b_test = choose_clolumns(b_test)
    a_train_x = create_dummy_variables(a_train_x)
    b_train_x = create_dummy_variables(b_train_x)
    b_test = create_dummy_variables(b_test)

    watchlist = [(xgb.DMatrix(a_train_x, label=a_train_y), 'train'),
                 (xgb.DMatrix(b_train_x, label=b_train_y), 'eval')]

    # trainning model for transfer learning
    Trate = 0.15
    params = {'booster': 'gbtree',
              'eta': 0.5,
              'max_depth': 5,
              'max_delta_step': 0,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'base_score': Trate,
              'objective': 'binary:logistic',
              'lambda': 3,
              'alpha': 8
              }
    params['eval_metric'] = 'auc'

    # model_1 = xgb.train(params,xgb.DMatrix(b_train_dummy[col],b_train['flag']),num_boost_round=150,maximize=True,verbose_eval=True)
    model_phase_1 = xgb.train(params, xgb.DMatrix(a_train_x, label=a_train_y), num_boost_round=1000,
                              evals=watchlist, early_stopping_rounds=100, maximize=True, verbose_eval=True)

    # train_X,test_X,train_Y,test_Y = train_test_split(b_train_x,b_train_y,test_size=0.2,random_state  = 2)
    # watchlist=[(xgb.DMatrix(train_X,label=train_Y),'train'),(xgb.DMatrix(test_X,label=test_Y),'eval')]

    Trate = 0.2
    params = {'booster': 'gbtree',
              'eta': 0.05,
              'max_depth': 4,
              'max_delta_step': 0,
              'subsample': 1,
              'colsample_bytree': 0.9,
              'base_score': Trate,
              'objective': 'binary:logistic',
              'lambda': 3,
              'alpha': 5
              }
    params['eval_metric'] = 'auc'
    model_phase_1_cla_1 = xgb.train(params, xgb.DMatrix(b_train_x, b_train_y), num_boost_round=25,
                                    xgb_model=model_phase_1, maximize=True, verbose_eval=True)

    Trate = 0.2
    params = {'booster': 'gbtree',
              'eta': 0.05,
              'max_depth': 5,
              'max_delta_step': 0,
              'subsample': 0.85,
              'colsample_bytree': 0.9,
              'base_score': Trate,
              'objective': 'binary:logistic',
              'lambda': 3,
              'alpha': 5
              }
    params['eval_metric'] = 'auc'
    model_phase_1_cla_2 = xgb.train(params, xgb.DMatrix(b_train_x, b_train_y), num_boost_round=40,
                                    xgb_model=model_phase_1, maximize=True, verbose_eval=True)

    Trate = 0.2
    params = {'booster': 'gbtree',
              'eta': 0.05,
              'max_depth': 4,
              'max_delta_step': 0,
              'subsample': 1,
              'colsample_bytree': 0.9,
              'base_score': Trate,
              'objective': 'binary:logistic',
              'lambda': 3,
              'alpha': 5
              }
    params['eval_metric'] = 'auc'
    model_phase_1_cla_3 = xgb.train(params, xgb.DMatrix(b_train_x, b_train_y), num_boost_round=28,
                                    xgb_model=model_phase_1, maximize=True, verbose_eval=True)

    Trate = 0.25
    params = {'booster': 'gbtree',
              'eta': 0.05,
              'max_depth': 5,
              'max_delta_step': 0,
              'subsample': 1,
              'colsample_bytree': 0.9,
              'base_score': Trate,
              'objective': 'binary:logistic',
              'lambda': 3,
              'alpha': 6
              }
    params['eval_metric'] = 'auc'
    model_phase_1_cla_4 = xgb.train(params, xgb.DMatrix(b_train_x, b_train_y), num_boost_round=30,
                                    xgb_model=model_phase_1, maximize=True, verbose_eval=True)


    pred = model_phase_1_cla_1.predict(xgb.DMatrix(b_test))
    pred1 = model_phase_1_cla_2.predict(xgb.DMatrix(b_test))
    pred2 = model_phase_1_cla_3.predict(xgb.DMatrix(b_test))
    pred3 = model_phase_1_cla_4.predict(xgb.DMatrix(b_test))
    submit['pred'] = (pred + pred1 + pred2 + pred3) / 4

    submit.to_csv('transfer_submit.csv', index=False)