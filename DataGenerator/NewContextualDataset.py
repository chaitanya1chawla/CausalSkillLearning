<<<<<<< HEAD
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from IPython import embed
import matplotlib.pyplot as plt

np.random.seed(seed=0)
number_datapoints = 20000
number_timesteps = 20

x_array_dataset = np.zeros((number_datapoints, number_timesteps, 2))
a_array_dataset = np.zeros((number_datapoints, number_timesteps-1, 2))
y_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
b_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
goal_array_dataset = np.zeros((number_datapoints, 1),dtype=int)

action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]])
start_limit = 10
start_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*start_limit
# valid_options = np.array([[2,3],[3,0],[1,2],[0,1]])

# option_probabilities = np.array([[0.2,0.7,0.,0.1],
# 								 [0.6,0.3,0.1,0.],
# 								 [0.,0.2,0.2,0.6],
# 								 [0.,0.,0.8,0.2]])
# option_probabilities = np.array([[0.5,0.5,0.,0.],
# 								 [0.,0.5,0.5,0.],
# 								 [0.25,0.0,0.25,0.5],
# 								 [0.,0.,0.,1.]])
# option_probabilities = np.array([[0.,0.95,0.,0.05],
# 								 [0.,0.,0.95,0.05],
# 								 [0.0,0.7,0.2,0.1],
# 								 [0.,0.,0.95,0.05]])
option_probabilities = np.array([[0.,0.95,0.,0.05],
								 [0.,0.,0.95,0.05],
								 [0.3,0.7,0.,0.],
								 [0.,0.,0.95,0.05]])

annotate_dps = 1000
start_noise_limit = 15
lim = 30
action_noise = 0.1

for i in range(number_datapoints):

	if i%annotate_dps==0:
		print("Processing Datapoint: ",i)		
	b_array_dataset[i,0] = 1.

	# Select one of four starting points. (-2,-2), (-2,2), (2,-2), (2,2)
	goal_array_dataset[i] = np.random.random_integers(0,high=3)
	# Adding random noise to start state.
	# x_array_dataset[i,0] = start_states[goal_array_dataset[i]] + 0.2*(np.random.random(2)-0.5)	
	x_array_dataset[i,0] = start_states[goal_array_dataset[i]] + start_noise_limit*(np.random.random(2)-0.5)
	if i==0:
		print(x_array_dataset[i,0])
	goal = -start_states[goal_array_dataset[i]]
	
	y_array_dataset[i,0] = np.random.randint(4)

	reset_counter = 0
	for t in range(number_timesteps-1):

		# GET B
		if t>0:
			# b_array[t] = np.random.binomial(1,prob_b_given_x)
			# b_array_dataset[i,t] = np.random.binomial(1,pb_x[0,x_array_dataset[i,t]])

			# If 3,4,5 timesteps have passed, terminate. 
			# if reset_counter>=3 and reset_counter<5:
			# 	b_array_dataset[i,t] = np.random.binomial(1,0.33)
			# elif reset_counter==5:
			# 	b_array_dataset[i,t] = 1

			if reset_counter==4:
				b_array_dataset[i,t] = 1

		# GET Y
		if b_array_dataset[i,t]:			

			# axes = -goal/abs(goal)
			# step1 = lim*np.ones((2))-axes*np.abs(x_array_dataset[i,t]-x_array_dataset[i,0])
			# # baseline = t*20*np.sqrt(2)/20
			# baseline = t
			# step2 = step1-baseline
			# step3 = step2/step2.sum()
			# y_array_dataset[i,t] = np.random.choice(valid_options[goal_array_dataset[i][0]])

			# Select y based on probability and previous y... 
			if t>0:
				y_array_dataset[i,t] = np.random.choice(range(4),p=option_probabilities[y_array_dataset[i,t-1]])

			reset_counter = 0
		else:
			reset_counter+=1
			y_array_dataset[i,t] = y_array_dataset[i,t-1]

		# GET A
		a_array_dataset[i,t] = action_map[y_array_dataset[i,t]]-action_noise/2+action_noise*np.random.random((2))  		

		# GET X
		x_array_dataset[i,t+1] = x_array_dataset[i,t]+a_array_dataset[i,t]

	if i%annotate_dps==0:
		plt.scatter(start_states[:,0],start_states[:,1],s=50)
		# plt.scatter()

		a = 10
		b = 50
		s = (b-a)/number_timesteps
		sizes = np.arange(a, b, s)[::-1]
		plt.scatter(x_array_dataset[i,:,0],x_array_dataset[i,:,1],cmap='jet',c=range(number_timesteps),s=sizes)
		plt.xlim(-lim,lim)
		plt.ylim(-lim,lim)
		plt.savefig("Traj_{0}.png".format(i))
		plt.close()

np.save("X_context.npy",x_array_dataset)
np.save("Y_context.npy",y_array_dataset)
np.save("B_context.npy",b_array_dataset)
np.save("A_context.npy",a_array_dataset)
np.save("G_context.npy",goal_array_dataset)
=======
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from IPython import embed
import matplotlib.pyplot as plt

np.random.seed(seed=0)
number_datapoints = 20000
number_timesteps = 20

x_array_dataset = np.zeros((number_datapoints, number_timesteps, 2))
a_array_dataset = np.zeros((number_datapoints, number_timesteps-1, 2))
y_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
b_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
goal_array_dataset = np.zeros((number_datapoints, 1),dtype=int)

action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]])
start_limit = 10
start_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*start_limit
# valid_options = np.array([[2,3],[3,0],[1,2],[0,1]])

# option_probabilities = np.array([[0.2,0.7,0.,0.1],
# 								 [0.6,0.3,0.1,0.],
# 								 [0.,0.2,0.2,0.6],
# 								 [0.,0.,0.8,0.2]])
option_probabilities = np.array([[0.5,0.5,0.,0.],
								 [0.,0.5,0.5,0.],
								 [0.25,0.0,0.25,0.5],
								 [0.,0.,0.,1.]])



start_noise_limit = 15
lim = 30
action_noise = 0.1

for i in range(number_datapoints):

	if i%1000==0:
		print("Processing Datapoint: ",i)
	b_array_dataset[i,0] = 1.

	# Select one of four starting points. (-2,-2), (-2,2), (2,-2), (2,2)
	goal_array_dataset[i] = np.random.random_integers(0,high=3)
	# Adding random noise to start state.
	# x_array_dataset[i,0] = start_states[goal_array_dataset[i]] + 0.2*(np.random.random(2)-0.5)	
	x_array_dataset[i,0] = start_states[goal_array_dataset[i]] + start_noise_limit*(np.random.random(2)-0.5)
	goal = -start_states[goal_array_dataset[i]]
	
	y_array_dataset[i,0] = np.random.randint(4)

	reset_counter = 0
	for t in range(number_timesteps-1):

		# GET B
		if t>0:
			# b_array[t] = np.random.binomial(1,prob_b_given_x)
			# b_array_dataset[i,t] = np.random.binomial(1,pb_x[0,x_array_dataset[i,t]])

			# If 3,4,5 timesteps have passed, terminate. 
			# if reset_counter>=3 and reset_counter<5:
			# 	b_array_dataset[i,t] = np.random.binomial(1,0.33)
			# elif reset_counter==5:
			# 	b_array_dataset[i,t] = 1

			if reset_counter==4:
				b_array_dataset[i,t] = 1

		# GET Y
		if b_array_dataset[i,t]:			

			# axes = -goal/abs(goal)
			# step1 = lim*np.ones((2))-axes*np.abs(x_array_dataset[i,t]-x_array_dataset[i,0])
			# # baseline = t*20*np.sqrt(2)/20
			# baseline = t
			# step2 = step1-baseline
			# step3 = step2/step2.sum()
			# y_array_dataset[i,t] = np.random.choice(valid_options[goal_array_dataset[i][0]])

			# Select y based on probability and previous y... 
			if t>0:
				y_array_dataset[i,t] = np.random.choice(range(4),p=option_probabilities[y_array_dataset[i,t-1]])

			reset_counter = 0
		else:
			reset_counter+=1
			y_array_dataset[i,t] = y_array_dataset[i,t-1]

		# GET A
		a_array_dataset[i,t] = action_map[y_array_dataset[i,t]]-action_noise/2+action_noise*np.random.random((2))  		

		# GET X
		x_array_dataset[i,t+1] = x_array_dataset[i,t]+a_array_dataset[i,t]


	# if i%1000==0:
	# 	plt.scatter(start_states[:,0],start_states[:,1],s=50)
	# 	# plt.scatter()
	# 	plt.scatter(x_array_dataset[i,:,0],x_array_dataset[i,:,1],cmap='jet',c=range(number_timesteps))
	# 	plt.xlim(-lim,lim)
	# 	plt.ylim(-lim,lim)
	# 	plt.savefig("Traj_{0}.png".format(i))
	# 	plt.close()

np.save("X_context.npy",x_array_dataset)
np.save("Y_context.npy",y_array_dataset)
np.save("B_context.npy",b_array_dataset)
np.save("A_context.npy",a_array_dataset)
np.save("G_context.npy",goal_array_dataset)

>>>>>>> 8a00d770d9f712d084b48df58682534b799db07c
