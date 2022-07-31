import csv
import numpy as np

NOT_SPAM = 0
SPAM = 1
TOTAL = 2
NUM_CLASSES = 2 # NOT_SPAM/SPAM
TRAINING = 0
TEST = 1
CLASSIFIER = 57

datafile = {0:'./data/training.csv', 1:'./data/test.csv'}
count = [0] * (NUM_CLASSES + 1) # NOT_SPAM/SPAM/TOTAL
training_data = []
test_data = []

# Load training data - 0 = training data, 1 = test data
for file in range(2):
    with open(datafile[file], newline='') as csvfile:
        spam_data = []
        not_spam_data = []
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data_row = []
            # Cast values to float and append to data_row
            for feature in range(len(row)):
                data_row.append(float(row[feature]))
            if data_row[CLASSIFIER] == NOT_SPAM:
                # If training_data, add one to 'not spam' count
                if file == TRAINING:
                    count[NOT_SPAM] += 1
                # Pop last element (classifier)
                data_row.pop()
                not_spam_data.append(data_row)
            else:
                # If training_data, add one to 'spam' count
                if file == TRAINING:
                    count[SPAM] += 1
                # Pop last element (classifier)
                data_row.pop()
                spam_data.append(data_row)
        # If training file is being loaded, append the two arrays to training_data    
        if file == TRAINING:
            training_data.append(not_spam_data)
            training_data.append(spam_data)
        # If test file is being loaded, append the two arrays to test_data
        else:
            test_data.append(not_spam_data)
            test_data.append(spam_data)

# Number of features in the data
FEATURES = len(training_data[0][0])-1

# Calculate starting prior probability for SPAM and NOT_SPAM
count[TOTAL] = sum(count)
p_prior = [0.0] * NUM_CLASSES
p_prior[SPAM] = np.log(count[SPAM] / count[TOTAL]) 
p_prior[NOT_SPAM] = np.log(count[NOT_SPAM] / count[TOTAL])

mean = []
stddev = []
# Transpose the training data array for each data class so features are 
# on the same row. Calculate the mean and standard deviation for each feature.
# data_class = NOT_SPAM/SPAM
for data_class in range(NUM_CLASSES):
    tmp = np.transpose(training_data[data_class])
    tmp_mean = []
    tmp_stddev = []
    # Iterate over each feature row in the transposed temp array
    for row in tmp:
        # Calculate mean and std deviation for each row
        tmp_mean.append(np.mean(row))
        std_deviation = np.std(row)
        # If the std deviation is 0, set it to 0.0001
        if std_deviation == 0:
            tmp_stddev.append(0.0001)
        else:
            tmp_stddev.append(std_deviation)
    mean.append(tmp_mean)
    stddev.append(tmp_stddev)

# -- CHECKING TEST DATA -- 
# Array for storing correct and incorrect identification of classes
CORRECT = 0
INCORRECT = 1
score = [[0,0],[0,0]]

# Array for storing P(x_i|class) calculations
p_likelyhood = [[0.0 for i in range(FEATURES)] for j in range(NUM_CLASSES)]

# Array for storing P(SPAM) and P(NOT_SPAM) values
p_class = [0.0] * NUM_CLASSES

for data_class in range(NUM_CLASSES):
    # Iterate over each row of data in the current data_class (NOT_SPAM/SPAM)
    for data_row in test_data[data_class]:
        # Calculate P(class) for both 'calc_class' classes (NOT_SPAM/SPAM)
        for class_being_calculated in range(NUM_CLASSES):
            # Calculate the likelyhood - P(x|class) - for each feature
            for feature in range(FEATURES):
                std = stddev[class_being_calculated][feature]
                mu = mean[class_being_calculated][feature]
                x = data_row[feature]
                exp = -(np.power(x - mu, 2) / (2 * np.power(std, 2)))
                gnb = (np.e**(exp)) / (np.sqrt(2*np.pi)*std)
                if (gnb) != 0:
                    p_likelyhood[class_being_calculated][feature] = np.log(gnb)
                else:
                    p_likelyhood[class_being_calculated][feature] = -np.inf
#                print (f"Gaussian: {gnb}, STD: {std}, MU: {mu}, X: {x}, NUMERATOR: {num}, DENOMINATOR: {denom}, EXP: {exp}")
        # Store the P(class) 
        p_class[SPAM] = p_prior[SPAM] + sum(p_likelyhood[SPAM])
        p_class[NOT_SPAM] = p_prior[NOT_SPAM] + sum(p_likelyhood[NOT_SPAM])

        # Score the result
        if(p_class[SPAM] > p_class[NOT_SPAM]):
            if (data_class == SPAM):
                score[SPAM][CORRECT] += 1
            else:
                score[SPAM][INCORRECT] += 1
        else:
            if (data_class == NOT_SPAM):
                score[NOT_SPAM][CORRECT] += 1
            else:
                score[NOT_SPAM][INCORRECT] += 1

print(f"Correctly identified SPAM:\t{score[SPAM][CORRECT]}")
print(f"Identified SPAM as NOT SPAM: \t{score[NOT_SPAM][INCORRECT]}")
print(f"Correctly identified NOT SPAM:\t{score[NOT_SPAM][CORRECT]}")
print(f"Identified NOT SPAM as SPAM:\t{score[SPAM][INCORRECT]}")

        w