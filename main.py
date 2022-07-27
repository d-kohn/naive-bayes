import csv
import numpy as np

SPAM = 1
NOT_SPAM = 0

count = [0,0]
training_data = []
training_targets = []
test_data = []
test_targets = []

# Load training data
with open('./data/training.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data_row = []
        # Cast values to float and append to data_row
        for i in range(len(row) - 3):
            data_row.append(float(row[i]))
        # Cast last 3 values to ints and append to data_row
        data_row.append(int(row[len(row)-3]))
        data_row.append(int(row[len(row)-2]))
        data_row.append(int(row[len(row)-1]))
        # Count spam/not-spam
        if data_row[len(row)-1] == 0:
            count[NOT_SPAM] += 1
        else:
            count[SPAM] += 1
        training_data.append(data_row)

# Load test data
with open('./data/test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data_row = []
        # Cast values to float and append to data_row
        for i in range(len(row) - 3):
            data_row.append(float(row[i]))
        # Cast last 3 values to ints and append to data_row
        data_row.append(int(row[len(row)-3]))
        data_row.append(int(row[len(row)-2]))
        data_row.append(int(row[len(row)-1]))
        test_data.append(data_row)

# Array location of the output target
TARGET = len(training_data[0])-1
# Calculate starting prior probability for SPAM and NOT_SPAM
prior = [0.0,0.0]
prior[SPAM] = count[SPAM] / sum(count)
prior[NOT_SPAM] = count[NOT_SPAM] / sum(count)

mean = []
stddev = []
# Transpose the training data array so features are in the same row.
# Calculate the mean and standard deviation for each feature.
tmp = np.transpose(training_data)
for row in tmp:
    mean.append(np.mean(row))
    stddev.append(np.std(row))
    # If the std deviation is 0, set it to 0.0001
    if stddev[len(stddev)-1] == 0:
        stddev[len(stddev)-1] = 0.0001



