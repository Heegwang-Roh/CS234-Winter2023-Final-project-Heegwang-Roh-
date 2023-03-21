import csv
import numpy as np
import random
from linucb import linUCB_arm, linUCB_policy, linUCB_run
from baseline import baseline_linear, baseline_fixed
from utils import process_data, plot_multiple_data
import matplotlib.pyplot as plt
from supervised_learning import run_linear_regression, run_softmax, run_neuralnet

filename = 'data/warfarin.csv'
calc_reward = [[0, -1, -1], [-1, 0, -1], [-1, -1, 0]]
calc_reward_biased = [[0, -0.5, -1], [-1, 0, -0.5], [-2, -1, 0]]
total_rewards_fixed = []
total_rewards_linear = []
total_rewards_ucb = []
linear_regression_result = {}
softmax_result = {}
NN_result = {}

###### CHANGE THIS TO MODIFY REWARD MATRIX ######
reward_matrix = calc_reward

###### CHANGE THIS TO MODIFY NUMBER OF PERMUTATIONS ######
num_permutations = 20
		
###### CHANGE THIS TO MODIFY DATA PROCESSING TYPE ######
include_genotype = 1
include_comorbities = 0

###### Reading csv file ######
a = []
with open(filename,'rt')as f:
	data = csv.reader(f)
	for row in data:
		if row[34].replace('.', '1').isdigit(): a.append(row)

###### Processing data #######		
feature_and_reward, doses = process_data(a, include_genotype, include_comorbities)

###### Splitting the data (Features + True Warfarin dose) #######
length = feature_and_reward[0].size
features = feature_and_reward[:, :length-1]
rewards = feature_and_reward[:, length-1:]
rewards = rewards.flatten().astype(int)
rewards_linear = np.empty((num_permutations, rewards.size))
rewards_fixed = np.empty((num_permutations, rewards.size))
rewards_ucb = np.empty((num_permutations, rewards.size))
fracs_incorrect_linear = np.empty((num_permutations, rewards.size))
fracs_incorrect_fixed = np.empty((num_permutations, rewards.size))
fracs_incorrect_ucb = np.empty((num_permutations, rewards.size))
regrets_ucb = np.empty((num_permutations, rewards.size))
frac_arms_correct_ucb = np.empty((num_permutations, 3))



###### Random shuffling ######
for i in range(num_permutations):
	feature_reward_shuffle = np.random.permutation(feature_and_reward)
	features = feature_reward_shuffle[:, :length-1]
	rewards = feature_reward_shuffle[:, length-1:]
	rewards = rewards.flatten().astype(int)
	
	linear_regression_result[i] = run_linear_regression(features, rewards, doses)
	softmax_result[i] = run_softmax(features, rewards)
	NN_result[i] = run_neuralnet(features, rewards)
	
	total_reward_linear, time_linear, reward_linear, frac_incorrect_linear = baseline_linear(features, rewards, reward_matrix)
	total_reward_fixed, time_fixed, reward_fixed, frac_incorrect_fixed = baseline_fixed(features, rewards, reward_matrix)
	linUCB = linUCB_policy(3, len(features[0]), alpha = 0.6)
	total_reward_ucb, time_ucb, reward_ucb, frac_incorrect_ucb, regret_ucb, arm_total_ucb, arm_correct_ucb = linUCB_run(linUCB, features, rewards, reward_matrix)
	
	total_rewards_linear.append(total_reward_linear)
	total_rewards_fixed.append(total_reward_fixed)
	total_rewards_ucb.append(total_reward_ucb)
	rewards_linear[i] = reward_linear
	rewards_fixed[i] = reward_fixed
	rewards_ucb[i] = reward_ucb
	regrets_ucb[i] = regret_ucb
	fracs_incorrect_linear[i] = frac_incorrect_linear
	fracs_incorrect_fixed[i] = frac_incorrect_fixed
	fracs_incorrect_ucb[i] = frac_incorrect_ucb
	frac_arms_correct_ucb[i] = arm_correct_ucb / arm_total_ucb
	print(i+1, 'th permutation done...')

print('<Result for', num_permutations, 'permutations>')
print('Baseline-fixed - total reward:', np.mean(total_rewards_fixed), '±', 2.093 / np.sqrt(20) * np.std(total_rewards_fixed))
print('Baseline-linear - total reward:', np.mean(total_rewards_linear), '±', 2.093 / np.sqrt(20) * np.std(total_rewards_linear))
print('UCB - total reward:', np.mean(total_rewards_ucb), '±', 2.093 / np.sqrt(20) * np.std(total_rewards_ucb))

print('Baseline-fixed - fraction incorrect:', np.mean(fracs_incorrect_fixed[:,fracs_incorrect_fixed.shape[1]-1]), '±', 2.093 / np.sqrt(20) * np.std(fracs_incorrect_fixed[:,fracs_incorrect_fixed.shape[1]-1]))
print('Baseline-linear - fraction incorrect:', np.mean(fracs_incorrect_linear[:,fracs_incorrect_linear.shape[1]-1]), '±', 2.093 / np.sqrt(20) * np.std(fracs_incorrect_linear[:,fracs_incorrect_linear.shape[1]-1]))
print('UCB - fraction incorrect:', np.mean(fracs_incorrect_ucb[:,fracs_incorrect_ucb.shape[1]-1]), '±', 2.093 / np.sqrt(20) * np.std(fracs_incorrect_ucb[:,fracs_incorrect_ucb.shape[1]-1]))

print('UCB - fraction correct for arm 1:', np.mean(frac_arms_correct_ucb[:,0]), '±', 2.093 / np.sqrt(20) * np.std(frac_arms_correct_ucb[:,0]))
print('UCB - fraction correct for arm 2:', np.mean(frac_arms_correct_ucb[:,1]), '±', 2.093 / np.sqrt(20) * np.std(frac_arms_correct_ucb[:,1]))
print('UCB - fraction correct for arm 3:', np.mean(frac_arms_correct_ucb[:,2]), '±', 2.093 / np.sqrt(20) * np.std(frac_arms_correct_ucb[:,2]))

###### Plotting ######	
plot_multiple_data(time_linear, [rewards_linear, rewards_fixed, rewards_ucb], ['linear', 'fixed', 'ucb'], 'reward.png', 'Patients seen', 'Cumulative reward', 0, 1000, -500, 0)
plot_multiple_data(time_linear, [regrets_ucb], ['ucb'], 'regret.png', 'Patients seen', 'Cumulative regret', 0, 1000, -500, 0)
plot_multiple_data(time_linear, [fracs_incorrect_linear, fracs_incorrect_fixed, fracs_incorrect_ucb], ['linear', 'fixed', 'ucb'], 'frac_incorrect.png', 'Patients seen', 'Fraction incorrect', 0, 1000, 0.3, 0.6)

print("===========")
print("Linear regression chose arms as:")
for i in range(num_permutations):
	print("Iteration ", i, " : ", linear_regression_result[i][0])

print("===========")
print("Softmax classifier chose arms as:")
for i in range(num_permutations):
	print("Iteration ", i, " : ", softmax_result[i][0])

print("===========")
print("Neural network classifier chose arms as:")
for i in range(num_permutations):
	print("Iteration ", i, " : ", NN_result[i][0])	


#total_reward_NN, time_NN, reward_NN, frac_incorrect_NN = NN_warfarin_run(features, rewards, reward_matrix)
#print(total_reward)