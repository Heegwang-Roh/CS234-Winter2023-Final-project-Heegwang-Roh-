import numpy as np
import random

class linUCB_arm():

	def __init__(self, num_features, alpha):
		self.alpha = alpha
		self.A = np.identity(num_features)
		self.b = np.zeros((num_features, 1))
		
	def calc_UCB(self, features):
		A_inv = np.linalg.inv(self.A)
		self.theta = np.dot(A_inv, self.b)
		features_col = features.reshape(-1, 1)
		p = np.dot(self.theta.T, features_col) + self.alpha * np.sqrt(np.dot(features_col.T, np.dot(A_inv, features_col)))
		
		return p[0][0]
	
	def calc_UCB_no_confidence(self, features):
		A_inv = np.linalg.inv(self.A)
		self.theta = np.dot(A_inv, self.b)
		features_col = features.reshape(-1, 1)
		p = np.dot(self.theta.T, features_col)
		
		return p[0][0]
		
	def reward_update(self, reward, features):
		features_col = features.reshape(-1, 1)
		self.A += np.dot(features_col, features_col.T)
		self.b += reward * features_col

class linUCB_policy():
  
	def __init__(self, num_arms, num_features, alpha):
		self.num_arms = num_arms
		self.linUCB_arms = []
		for i in range(num_arms):
			self.linUCB_arms.append(linUCB_arm(num_features, alpha))
		
	def choose_arm(self, features):
		max_value = -10
		index = 0
		for arm_index in range(self.num_arms):
			arm_value = self.linUCB_arms[arm_index].calc_UCB(features)
			if arm_value > max_value:
				max_value = arm_value
				index = arm_index
		return index
		
	def choose_arm_no_confidence (self, features):
		max_value = -10
		index = 0
		for arm_index in range(self.num_arms):
			arm_value = self.linUCB_arms[arm_index].calc_UCB_no_confidence(features)
			if arm_value > max_value:
				max_value = arm_value
				index = arm_index
		return index
	
def linUCB_run(linUCB, features, rewards, reward_matrix):

	###########################
	#
	# This function calculates the reward for the LinUCB method
	#
	# INPUT
	# - features : 2-D numpy array, where each row is an 1-D numpy array of patient features [weight, height, ....])x values (ex. # of patients trained, time, etc.)
	# - correct_ranges : 1-D numpy array of correct warfarin dose range (1, 2, or 3) of each patients
	# - reward_matrix : 3 x 3 matrix where A[i][j] is the reward of prescribing dose range "i" while the correct dose range is "j"
	#
	# OUTPUT
	# - total_reward : cumulative sum of reward for each patient
	# - time_log : an 1-D array of patient numbers [0, 1, 2, ..... , num_patient - 1]
	# - reward_log : an 1-D array of reward of decisions for each patient
	# - frac_incorrect : an 1-D array of fraction of incorrect decisions made, up to each timepoint
	# - regret_log : an 1-D array of cumulative regret
	# - arm_total : total number of patients in each arm
	# - arm_correct : total number of correct prescription made, for each arm
	#
	###########################
	
	time_log = []
	reward_log = []
	total_reward = 0
	frac_incorrect = []
	count_total = 0
	count_incorrect = 0
	regret_log = []
	total_regret = 0
	fraction_correct_arm = 0
	arm_total = np.zeros(3)
	arm_correct = np.zeros(3)
	
	for i in range(len(features)):
		feature = features[i]
		chosen_arm = linUCB.choose_arm(feature)
		chosen_arm_no_confidence = linUCB.choose_arm_no_confidence(feature)
		reward = reward_matrix[chosen_arm][rewards[i]]
		regret = linUCB.linUCB_arms[chosen_arm_no_confidence].calc_UCB_no_confidence(feature) - linUCB.linUCB_arms[chosen_arm].calc_UCB_no_confidence(feature)
		total_reward += reward
		linUCB.linUCB_arms[chosen_arm].reward_update(reward, feature)
		time_log.append(i)
		reward_log.append(total_reward)
		count_total += 1
		arm_total[rewards[i]] += 1
		arm_correct[chosen_arm] += 1
		if chosen_arm != rewards[i]:
			count_incorrect += 1
			arm_correct[chosen_arm] -= 1
		frac_incorrect.append(count_incorrect/count_total)
		total_regret += regret
		regret_log.append(total_regret)
		#print('correct arm', rewards[i], 'chosen arm', chosen_arm)
		
	return total_reward, time_log, reward_log, frac_incorrect, regret_log, arm_total, arm_correct