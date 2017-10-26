import pandas as pd
import numpy as np


def get_train_label(df):
    train_data = df.drop(['flag'], axis=1)
    label = df['flag']
    return train_data, label


def column_filter(df):
    features = df.columns
    for feature in features:
        total_size= len(df[feature])
        not_null = len(df[df[feature].isnull()])
        result = not_null/total_size
        if result >0.9: #0.9 the number of reserved columns is 348. 0.8 the number of reserved columns is 334
            df.drop(feature,axis=1 ,inplace = True)
    return df

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





if __name__ == '__main__':
    pd.set_option('display.width', 100)
    a_train = pd.read_csv('A_train.csv')
    b_train = pd.read_csv('B_train.csv')
    b_test = pd.read_csv('B_test.csv')

    a_train_data, a_train_label = get_train_label(a_train)
    b_train_data, b_train_label = get_train_label(b_train)



    print('column processing\n')
    b_train_data = column_filter(b_train_data)
    b_test = column_filter(b_test)



    print('handling missing values\n')
    b_train_data = b_train_data.fillna(-999)
    b_test = b_test.fillna(-999)



    print('creating dummy variables\n ')
    # b_train_columns = get_discrete_variables(b_train)
    # b_test_columns = get_discrete_variables(b_test)
    b_train_data = create_dummy_variables(b_train_data)
    b_test = create_dummy_variables(b_test)

    print('save data')
    b_train = pd.concat((b_train_data,b_train_label),axis=1)
    b_train.to_csv('B_train_dummy.csv', index=False)
    b_test.to_csv('B_test_dummy.csv', index=False)


