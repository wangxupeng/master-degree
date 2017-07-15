# -*- coding: utf-8 -*-
"""
@author: Wang xupeng
"""

import tensorflow as tf 
import numpy as np  
from scipy.stats import norm
import matplotlib.pyplot as plt
import time 

def lookback(r,sigma,T,x):
    result=[]
    for s0 in range(x):
        a1=((r+sigma**2/2)*T)/(sigma*np.sqrt(T))
        a2=a1-sigma*np.sqrt(T)
        a3=((-r+sigma**2/2)*T)/(sigma*np.sqrt(T))
        y1=0
        call_option=s0*norm.cdf(a1)-s0*sigma**2/(2*r)*norm.cdf(-a1)-s0*np.exp(-r*T)*(norm.cdf(a2)
        -sigma**2/(2*r)*np.exp(y1)*norm.cdf(-a3))
        result.append(call_option)
    return result

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

def run_main():
    start = time.time()
    option_price1 = lookback(0.5,0.2,1,500)
    option_price2 = lookback(1,2,2,500)
    option_price3 = lookback(1.25,2.5,3,500)
    option_price4 =  lookback(0.75,1,1.5,500)
    option_price_train = np.array(option_price1+option_price2+option_price3)
    option_price_test = np.array(option_price4)
    
    r_train = np.array(500*[0.1]+500*[1]+500*[1.5])
    sigma_train= np.array(500*[0.2]+500*[2]+500*[2.5])
    Time_train = np.array(500*[1]+500*[2]+500*[3])
    
    r_test = np.array(500*[0.75])
    sigma_test = np.array(500*[1])
    Time_test = np.array(500*[2])
    
    sess = tf.Session()
    x_vals_train = np.array([option_price_train,r_train,sigma_train,Time_train]).T
    x_vals_test = np.array([option_price_test,r_test,sigma_test,Time_test]).T
    y_vals_train = np.array(option_price_train)
    y_vals_test = np.array(option_price_test)
    
    x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
    x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))
    
    y_vals_train = np.nan_to_num(normalize_cols(y_vals_train))
    y_vals_test = np.nan_to_num(normalize_cols(y_vals_test))

    batch_size = 50
    x_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    
    hidden_layer_nodes = 100
    
    A1 = tf.Variable(tf.random_normal(shape=[4,hidden_layer_nodes]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
    A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1]))
    b2 = tf.Variable(tf.random_normal(shape=[1]))
    hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
    final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2),b2))
    loss = tf.reduce_mean(tf.square(y_target - final_output))    
    my_opt = tf.train.GradientDescentOptimizer(0.001)
    train_step = my_opt.minimize(loss)
    init = tf.initialize_all_variables()
    sess.run(init)
    
    loss_vec = []
    test_loss = []
    
    for i in range(20000):
        # First we select a random set of indices for the batch.
        rand_index=np.random.choice(len(x_vals_train),size=batch_size)
        # We then select the training values
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        # Now we run the training step
        sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})
        # We save the training loss
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(np.sqrt(temp_loss))
        # Finally, we run the test-set loss and save it.
        test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_loss.append(np.sqrt(test_temp_loss))
        if (i+1)%50==0:
                print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))
    end = time.time()
    print("time consumption:",end-start)           
    plt.plot(loss_vec, 'k-', label='Train Loss')
    plt.plot(test_loss, 'r--', label='Test Loss')
    plt.title('Loss (MSE) per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    
if __name__ == '__main__':
    run_main()  

