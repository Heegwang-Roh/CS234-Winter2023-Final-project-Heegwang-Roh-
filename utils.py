import numpy as np
import matplotlib.pyplot as plt

def process_data(data, flag_genotypes, flag_comorbities):

	###########################
	#
	# INPUT
	# - data : file directory to the warfarin.csv
	# - flag_comorbities : an integer specifying whether to read comorbities (1) or not (0)
	#
	# OUTPUT
	# - numpy array of features, as documented in appx.pdf
	# - for each row, the last data is the ground true value of Warfarin dose, either 1 (low), 2 (med), or 3 (high)
	#
	###########################
	
	features = []
	doses = []
	for i in range(len(data)):
		feature = []
		age = 5.936 #Average of ages in the dataset
		if data[i][4].split(' ')[0].isnumeric(): age = int(data[i][4].split(' ')[0])/10
		## Height and Weight
		height = 166.7155 #Average of heights in the dataset
		weight = 77.85306 #Average of weights in the dataset
		if data[i][5].replace('.', '1').isdigit():
			height = float(data[i][5])
			if not data[i][6].replace('.', '1').isdigit():
				weight = height * 1.1507 - 115.1
		if data[i][6].replace('.', '1').isdigit():
			weight = float(data[i][6])
			if not data[i][5].replace('.', '1').isdigit():
				height = (weight + 115.1) / 1.1507
		## Ethnicity
		is_asian = 0
		is_black = 0
		is_unknown = 0
		race = data[i][2]
		if race == 'Asian': is_asian = 1
		elif race == 'Black or African American': is_black = 1
		elif race == 'Unknown': is_unknown = 1
		## Medication
		medication = 0
		if data[i][24] == '1': medication = 1
		if data[i][25] == '1': medication = 1
		if data[i][26] == '1': medication = 1
		## Medication - Amiodarone
		amiodarone = 0
		if data[i][23] == '1': amiodarone = 1		
		
		# Add basic features, that is used in IWPC linear algorithm
		feature.extend((age, height, weight, is_asian, is_black, is_unknown, medication, amiodarone))
		
		if flag_genotypes == 1:
			vkorc_a_g = 0
			vkorc_a_a = 0
			cyp2c9_1_2 = 0
			cyp2c9_1_3 = 0
			cyp2c9_2_2 = 0
			cyp2c9_2_3 = 0
			cyp2c9_3_3 = 0
			## Genotype - VKORC1 rs9923231
			if data[i][41] == 'A/G': vkorc_a_g = 1
			if data[i][41] == 'A/A': vkorc_a_a = 1
			## Genotype - CYP2C9
			if data[i][37] == '*1/*2': cyp2c9_1_2 = 1
			if data[i][37] == '*1/*3': cyp2c9_1_3 = 1
			if data[i][37] == '*2/*2': cyp2c9_2_2 = 1
			if data[i][37] == '*2/*3': cyp2c9_2_3 = 1
			if data[i][37] == '*3/*3': cyp2c9_3_3 = 1
			
			# Add features related to genotypes
			feature.extend((vkorc_a_g, vkorc_a_a, cyp2c9_1_2, cyp2c9_1_3, cyp2c9_2_2, cyp2c9_2_3, cyp2c9_3_3))
			
		
		if flag_comorbities == 1:
			obesity = 0
			artery_disase = 0
			cholesterol = 0
			perip_vascular = 0
			cardiomyopathy = 0
			flutter = 0
			infarction = 0
			abnormal_heart = 0
			stroke = 0
			ischemic = 0
			dyslipidemia = 0
			for string in data[i][8].split("; "):
				comorbity = string.lower()
				if 'obesity' in comorbity: obesity = 1
				if 'coronary artery disease': artery_disease = 1
				if 'high cholesterol' in comorbity: cholesterol = 1
				if 'peripheral vascular disease' in comorbity: perip_vascular = 1
				if 'dilated cardiomyopathy' in comorbity: cardiomyopathy = 1
				if 'atrial flutter' in comorbity: flutter = 1
				if 'myocardial infarction' in comorbity: infarction = 1
				if 'abnormal heart rhythm' in comorbity: abnormal_heart = 1
				if 'stroke' in comorbity: stroke = 1
				if 'ischemic stroke' in comorbity: ischemic = 1
				if 'dyslipidemia' in comorbity: dyslipidemia = 1
			# Add features related to comorbities
			feature.extend((obesity, artery_disease, cholesterol, perip_vascular, cardiomyopathy, flutter, infarction, abnormal_heart, stroke, ischemic, dyslipidemia))
	
		correct_dose_range = 0
		correct_dose = float(data[i][34])
		if correct_dose >= 21: correct_dose_range = 1
		if correct_dose > 49: correct_dose_range = 2
		
		feature.append(correct_dose_range)
		features.append(feature)
		doses.append(correct_dose)
	return np.array(features), np.array(doses)
	
def plot_multiple_data(x, datasets, labels, plot_name, x_label, y_label, xlim_1, xlim_2, ylim_1, ylim_2):

	###########################
	#
	# INPUT
	# - x : 1-D numpy array of x values (ex. # of patients trained, time, etc.)
	# - datasets : a list of 2-D numpy arrays. Each 2-D numpy array is an array of individual data in 1-D numpy array format
	# - labels : a list of strings, where each string specifies the label for each dataset
	# - plot_name : string which specifies the name of the to be saved plot
	#
	# OUTPUT
	# - save the plot of multiple datasets, including the confidence interval for individual dataset.
	#
	###########################
	
	fig, ax = plt.subplots()
	for i in range(len(datasets)):
		mean = np.mean(datasets[i], axis=0)
		std = np.std(datasets[i], axis=0)
		ax.plot(x, mean, label = labels[i])
		ax.fill_between(x, mean - 2.093 / np.sqrt(20) * std, mean + 2.093 / np.sqrt(20) * std, alpha = 0.15)
	ax.legend()
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(plot_name)
	
	plt.xlim(xlim_1, xlim_2)
	plt.ylim(ylim_1, ylim_2)
	plot_name_restricted = plot_name[:-4] + 'restricted.png'
	plt.savefig(plot_name_restricted)
	