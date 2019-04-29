import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

w_target = np.array([0.5, 3, 2.4])  # 定义参数
b_target = np.array([0.9])  # 定义参数

f_des = 'y = {:.2f} + {:.2f} * x + ' \
        '{:.2f} * x^2 + {:.2f} * x^3'.\
        format(b_target[0], w_target[0],
               w_target[1], w_target[2])  # 打印出函数的式子
print(f_des)

x_sample = np.arange(-3, 3.1, 0.2)
y_sample = b_target[0] +w_target[0] * x_sample+w_target[1] \
           *x_sample**2+w_target[2] * x_sample**3
plt.plot(x_sample, y_sample, label='real curve')
plt.legend()
plt.show()

x_train = np.stack([x_sample**i for i in range(1, 4)], axis=1)
# print(x_sample)
# print(x_train)

x_train = tf.constant(x_train, dtype=tf.float32, name='x_train')
y_train = tf.constant(y_sample, dtype=tf.float32, name='y_train')
w = tf.Variable(initial_value=tf.random_normal(shape=(3, 1)),
                dtype=tf.float32,name='weights')
b = tf.Variable(initial_value=0, dtype=tf.float32, name='bias')

def multi_linear(x):\
    return tf.squeeze(tf.matmul(x, w) +b)

y_ = multi_linear(x_train)

sess = tf.InteractiveSession()
# %matplotlibinline
sess.run(tf.global_variables_initializer())
x_train_value = x_train.eval(session=sess)
y_train_value = y_train.eval(session=sess)
y_pred_value = y_.eval(session=sess)

plt.plot(x_train_value[:,0], y_pred_value,
         label='fitting curve', color='r')
plt.plot(x_train_value[:,0], y_train_value,
         label='real curve', color='b')
plt.legend()

plt.show()

loss = tf.reduce_mean(tf.square(y_train-y_))
loss_numpy = sess.run(loss)
print("0", loss_numpy)

# 利用`tf.gradients()`自动求解导数
w_grad, b_grad = tf.gradients(loss, [w, b])
print("w_grad:",  w_grad.eval(session=sess))
print("b_grad:", b_grad.eval(session=sess))
# 利用梯度下降更新参数
lr = 1e-3

w_update = w.assign_sub(lr*w_grad)
b_update = b.assign_sub(lr*b_grad)

sess.run([w_update, b_update])
x_train_value = x_train.eval(session=sess)
y_train_value = y_train.eval(session=sess)
y_pred_value = y_.eval(session=sess)
loss_numpy = loss.eval(session=sess)

plt.plot(x_train_value[:,0], y_pred_value,
         label='fitting curve', color='r')
plt.plot(x_train_value[:,0], y_train_value,
         label='real curve', color='b')
plt.legend()
plt.title('loss: %.4f'%loss_numpy)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.show()
fig.canvas.draw()
sess.run(tf.global_variables_initializer())

for e in range(2500):
    sess.run([w_update, b_update])

x_train_value = x_train.eval(session=sess)
y_train_value = y_train.eval(session=sess)
y_pred_value = y_.eval(session=sess)
loss_numpy = loss.eval(session=sess)

ax.clear()
ax.plot(x_train_value[:,0], y_pred_value,
        label='fitting curve', color='r')
ax.plot(x_train_value[:,0], y_train_value,
        label='real curve', color='b')
ax.legend()

fig.canvas.draw()
plt.show()
print()
print("1", loss_numpy)
print("w_grad:",  w_grad.eval(session=sess))
print("b_grad:", b_grad.eval(session=sess))
plt.pause(0.1)

sess.close()
