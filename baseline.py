def baseline_fixed(features, correct_ranges, reward_matrix):

	###########################
	#
	# This function calculates the reward for the baseline method (fixed dose)
	#
	# INPUT
	# - features : 2-D numpy array, where each row is an 1-D numpy array of patient features [weight, height, ....])x values (ex. # of patients trained, time, etc.)
	# - correct_ranges : 1-D numpy array of correct warfarin dose range (0, 1, or 2) of each patients
	# - reward_matrix : 3 x 3 matrix where A[i][j] is the reward of prescribing dose range "i" while the correct dose range is "j"
	#
	# OUTPUT
	# - reward : cumulative sum of reward for each patient
	# - time_log : an 1-D array of patient numbers [0, 1, 2, ..... , num_patient - 1]
	# - reward_log : an 1-D array of reward of decisions for each patient
	# - frac_incorrect : an 1-D array of fraction of incorrect decisions made, up to each timepoint
	#
	###########################
	
	time_log = []
	reward_log = []
	reward = 0
	frac_incorrect = []
	count_incorrect = 0
	count_total = 0
	for i in range(len(features)):
		correct_range = correct_ranges[i]
		reward += reward_matrix[1][correct_range]
		time_log.append(i)
		reward_log.append(reward)
		count_total += 1
		if correct_range != 1: # fixed dose (35mg/week) is always dose range 2
			count_incorrect += 1
		frac_incorrect.append(count_incorrect/count_total)
	return reward, time_log, reward_log, frac_incorrect

def baseline_linear(features, correct_ranges, reward_matrix):

	###########################
	#
	# This function calculates the reward for the baseline method (linear clinical dosig algorithm)
	#
	# INPUT
	# - features : 2-D numpy array, where each row is an 1-D numpy array of patient features [weight, height, ....])x values (ex. # of patients trained, time, etc.)
	# - correct_ranges : 1-D numpy array of correct warfarin dose range (0, 1, or 2) of each patients
	# - reward_matrix : 3 x 3 matrix where A[i][j] is the reward of prescribing dose range "i" while the correct dose range is "j"
	#
	# OUTPUT
	# - reward : cumulative sum of reward for each patient
	# - time_log : an 1-D array of patient numbers [0, 1, 2, ..... , num_patient - 1]
	# - reward_log : an 1-D array of reward of decisions for each patient
	# - frac_incorrect : an 1-D array of fraction of incorrect decisions made, up to each timepoint
	#
	###########################
	
	reward = 0
	time_log = []
	reward_log = []
	frac_incorrect = []
	count_incorrect = 0
	count_total = 0
	for i in range(len(features)):
		dose = 4.0376
		age = features[i][0]
		height = features[i][1]
		weight = features[i][2]
		is_asian = features[i][3]
		is_black = features[i][4]
		is_unknown = features[i][5]
		medication = features[i][6]
		amiodarone = features[i][7]
		correct_range = correct_ranges[i]
		dose = dose - 0.2546 * age + 0.0118 * height + 0.0134 * weight - 0.6752 * is_asian + 0.4060 * is_black + 0.0443 * is_unknown + 1.2799 * medication - 0.5695 * amiodarone
		dose = dose * dose
		
		## Comparing dose range with ground truth
		dose_range = 0
		if dose >= 21: dose_range = 1
		if dose > 49: dose_range = 2
		reward += reward_matrix[dose_range][correct_range]
		time_log.append(i)
		reward_log.append(reward)
		count_total += 1
		if correct_range != dose_range:
			count_incorrect += 1
		frac_incorrect.append(count_incorrect/count_total)

	return reward, time_log, reward_log, frac_incorrect