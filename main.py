import csv
import numpy as np

NOT_SPAM = 0
SPAM = 1
TOTAL = 2
NUM_CLASSES = 2 # NOT_SPAM/SPAM

# def calc_mean(data):
#     means = []
#     total = len(data)
#     for feature in range(FEATURES):
#         sum = 0.0
#         for row in data:
#             sum += row[feature]
#         means.append(sum/total)
#     return means

# def calc_std(data, mean):
#     std = []
#     total = len(data)    
#     for feature in range(FEATURES):
#         sum = 0.0
#         for row in data:
#             sum += np.power(row[feature]-mean[feature], 2)
#         std_deviation = np.sqrt(sum/total)
#         if std_deviation == 0:
#             std.append(0.0001)
#         else:
#             std.append(std_deviation)
#     return std
    
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
            for i in range(len(row)):
                data_row.append(float(row[i]))
            # Last value (classifier) cast to int and append to data_row
            data_row.append(int(row[len(row)-1]))
            # Count spam/not-spam, pop the classifier value off the data_row, 
            # append data_row to spam or not_spam array
            if data_row[len(row)-1] == NOT_SPAM:
                # If training_data, add one to 'not spam' count
                if file == 0:
                    count[NOT_SPAM] += 1
                # Pop last element (data_class)
                data_row.pop()
                not_spam_data.append(data_row)
            else:
                # If training_data, add one to 'spam' count
                if file == 0:
                    count[SPAM] += 1
                # Pop last element (data_class)
                data_row.pop()
                spam_data.append(data_row)
        # If training file is being loaded, append the two arrays to training_data
        if file == 0:
            training_data.append(not_spam_data)
            training_data.append(spam_data)
        # If test file is being loaded, append the two arrays to test_data
        else:
            test_data.append(not_spam_data)
            test_data.append(spam_data)

count[TOTAL] = sum(count)

# Number of features in the data
FEATURES = len(training_data[0][0])-1

# Calculate starting prior probability for SPAM and NOT_SPAM
p_prior = [0.0] * NUM_CLASSES
p_prior[SPAM] = np.log(count[SPAM] / count[TOTAL]) 
p_prior[NOT_SPAM] = np.log(count[NOT_SPAM] / count[TOTAL])

mean = []
stddev = []
# mean.append(calc_mean(training_data[NOT_SPAM]))
# mean.append(calc_mean(training_data[SPAM]))
# stddev.append(calc_std(training_data[NOT_SPAM], mean[NOT_SPAM]))
# stddev.append(calc_std(training_data[SPAM], mean[SPAM]))

#Transpose the training data array for each data class so features are 
#on the same row. Calculate the mean and standard deviation for each feature.
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

set = {0:"Training", 1:"Test"}
# all_data[Training/Test][NOT_SPAM/SPAM][FEATURES]
all_data = []
all_data.append(training_data)
all_data.append(test_data)
CORRECT = 0
INCORRECT = 1

# -- CHECKING TEST DATA -- 
# Iterate over each data_class lists (data_type = Training/Test)
for data_type in range(2):
    # Array for storing correct and incorrect identification of classes
    score = [[0,0],[0,0]]

    # Array for storing P(x_i|class) calculations
    p_likelyhood = [[0.0 for i in range(FEATURES)] for j in range(NUM_CLASSES)]
    # Array for storing P(SPAM) and P(NOT_SPAM) values
    p_class = [0.0] * NUM_CLASSES

    for data_class in range(NUM_CLASSES):
        # Iterate over each row of data in the current data_class (NOT_SPAM/SPAM)
        for data in all_data[data_type][data_class]:
            # Calculate P(class) for both 'calc_class' classes (NOT_SPAM/SPAM)
            for calc_class in range(NUM_CLASSES):
                # Calculate the likelyhood - P(x|class) - for each feature
                for feature in range(FEATURES):
                    std = stddev[calc_class][feature]
                    mu = mean[calc_class][feature]
                    x = data[feature]
                    exp = -(np.power(x - mu, 2) / (2 * np.power(std, 2)))
                    gnb = (np.e**(exp)) / (np.sqrt(2*np.pi)*std)
                    if (gnb) != 0:
                        p_likelyhood[calc_class][feature] = np.log(gnb)
                    else:
                        p_likelyhood[calc_class][feature] = -np.inf
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

    print(f"RESULTS - {set[data_type]}:")
    print(f"Correctly identified SPAM:\t{score[SPAM][CORRECT]}")
    print(f"Identified SPAM as NOT SPAM: \t{score[NOT_SPAM][INCORRECT]}")
    print(f"Correctly identified NOT SPAM:\t{score[NOT_SPAM][CORRECT]}")
    print(f"Identified NOT SPAM as SPAM:\t{score[SPAM][INCORRECT]}")
    print()

        