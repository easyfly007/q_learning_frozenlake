
# q net
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline

tf.reset_default_graph()
inputs1 = tf.placeholder(shape = [1, 16], dtype = tf.float32)
w = tf.Variable(tf.rnadom_uniform([16, 4], 0, 0.01))
qout = tf.matmul(inputs1, w)
predict = tf.argmax(qout, 1)

nextq = tf.placeholder(shape = [1,4], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextq - qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()
y = 0.99
e = 0.1
num_episodes = 2000
jList = []
rList = []
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_episodes):
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		while j < 99:
			j += 1
			a, allq = sess.run([predict, qout], feed_dict = {inputs1: np.identity(16)[s:s+1]})
			if np.random.rand(1) < e:
				a[0] = env.action_space.sample()
			s1, r, d, _ = env.step(a[0])
			q1 = sess.run(qout, feed_dict = {inputs1: np.identity(16)[s1:s1+1]})
			maxq1 = np.max(q1)
			targetq = allq
			targetq[0, a[0]] = r + y*maxq1
			_, w1 = sess.run([updateModel, w], feed_dict = {inputs1:np.identity(16)[s:s+1], nextq:targetq})
			rAll += r
			s = s1
			if d == True:
				e = 1.0/((i/50) + 10)
				break
		jList.append(j)
		rList.append(rAll)
	print('precent of successful episodes: ' + str(sum(rList)/num_episodes) + "%")