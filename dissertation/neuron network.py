import tensorflow as tf 
import numpy as np  
import pandas as pd  
from scipy.stats import norm
import matplotlib.pyplot as plt

#s0=initial stock price
#sigma= volatility 
#smin= the minimum asset price achieved to date
r = 0.1
sigma = 0.3
T = 1
result=[]
for s0 in range(500):
    a1=((r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    a2=a1-sigma*np.sqrt(T)
    a3=((-r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    y1=-2*(r-sigma**2/2)/(sigma**2)
    call_option=s0*norm.cdf(a1)-s0*sigma**2/(2*r)*norm.cdf(-a1)-s0*np.exp(-r*T)*(norm.cdf(a2)
    -sigma**2/(2*r)*np.exp(y1)*norm.cdf(-a3))
    result.append(call_option)
option_price=result

r=np.array(500*[0.1])
sigma=np.array(500*[0.3])
Time=np.array(500*[1])
s0=[]
for i in range(500):
    s0.append(i)
s0=np.array(s0)    
num=len(s0)
x_vals=np.array([s0,r,sigma,Time]).T
y_vals=np.array(option_price)  

sess = tf.Session()
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

batch_size = 50
x_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

hidden_layer_nodes = 10
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

accuracy=0.
loss_vec = []
test_loss = []
for i in range(10000):
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
    if temp_loss <0.05:
        accuracy += 1./len(y_vals_test)
print("accuracy is ",'%.2f%%' %accuracy)
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
