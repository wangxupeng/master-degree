import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    model1 = pd.read_csv('subimit_target.csv')
    model2 = pd.read_csv('transfer_submit.csv')
    model1['pred'] = model1['pred'] * 0.85 + model2['pred'] * 0.15
    model1.to_csv('submit_online.csv', index=False)