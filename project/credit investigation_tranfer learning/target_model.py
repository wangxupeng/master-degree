import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import auc,roc_auc_score
import xgboost as xgb


def col_filter(df):
    columns = [x for x in b_train_x.columns if x in b_test.columns]
    cols = df.columns
    for column in cols:
        if column not in columns:
            del df[column]
    return df





if __name__ == '__main__':
    np.random.seed(0)

    b_train = pd.read_csv('B_train_dummy.csv')
    b_test = pd.read_csv('B_test_dummy.csv')

    b_train_x = b_train.drop(['no','flag'], axis=1)
    b_train_y = b_train['flag']
    b_test_x = b_test.drop(['no'], axis=1)

    b_train_x =col_filter(b_train_x)
    b_test_x = col_filter(b_test_x)





    # train_X, test_X, train_Y, test_Y = train_test_split(b_train_x, b_train_y, test_size=0.2, random_state=0)
    # watchlist = [(xgb.DMatrix(train_X, label=train_Y), 'train'), (xgb.DMatrix(test_X, label=test_Y), 'eval')]

    dtrain_B = xgb.DMatrix(b_train_x, b_train_y)
    Trate = 0.25
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
    model_phase_1_cla_2 = xgb.train(params, dtrain_B, num_boost_round=130, maximize=True, verbose_eval=True)

    # 0.599155
    # Trate=0.15
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 3,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    # #               'objective': 'binary:logitraw',
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=130,maximize=True,verbose_eval=True)
    # model_phase_1_cla = xgb.train(params,xgb.DMatrix(train_X,label=train_Y),num_boost_round=1000,evals=watchlist,early_stopping_rounds=50,maximize=True,verbose_eval=True)


    # 0.594276
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 5,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':2,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=150,maximize=True,verbose_eval=True)
    # model_phase_1_cla = xgb.train(params,xgb.DMatrix(train_X,label=train_Y),num_boost_round=1000,evals=watchlist,early_stopping_rounds=50,maximize=True,verbose_eval=True)

    # 0.595855
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 5,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=150,maximize=True,verbose_eval=True)

    # 0.594632
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 5,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=200,maximize=True,verbose_eval=True)

    # 0.596701
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 5,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=138,maximize=True,verbose_eval=True)

    # 0.596326
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 5,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=120,maximize=True,verbose_eval=True)

    # 0.598221
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 4,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=150,maximize=True,verbose_eval=True)

    # 0.599235
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 4,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=138,maximize=True,verbose_eval=True)


    # 0.598832
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 4,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=120,maximize=True,verbose_eval=True)

    # 0.600018
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 4,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=130,maximize=True,verbose_eval=True)

    # 0.595537
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 3,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=150,maximize=True,verbose_eval=True)

    # 0.593465
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 3,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=130,maximize=True,verbose_eval=True)


    # 0.594750
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 3,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=200,maximize=True,verbose_eval=True)

    # #0.599226
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 4,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=135,maximize=True,verbose_eval=True)


    # 0.598256
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 4,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=125,maximize=True,verbose_eval=True)

    # 0.599600
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 4,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=132,maximize=True,verbose_eval=True)


    # 0.599844
    # Trate=0.25
    # params = {'booster':'gbtree',
    #               'eta': 0.05,
    #               'max_depth': 4,
    #               'max_delta_step': 0,
    #               'subsample':1,
    #               'colsample_bytree': 0.9,
    #               'base_score': Trate,
    #               'objective': 'binary:logistic',
    #               'lambda':3,
    #               'alpha':5
    #               }
    # params['eval_metric'] = 'auc'
    # model_phase_1_cla_2 = xgb.train(params,dtrain_B,num_boost_round=132,maximize=True,verbose_eval=True)

    predict = model_phase_1_cla_2.predict(xgb.DMatrix(b_test_x))
    result1 = pd.DataFrame()
    result1['no'] = b_test['no']
    result1['pred'] = predict[:]
    result1.to_csv('subimit_target.csv', index=False)


