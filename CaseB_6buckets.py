import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
np.set_printoptions(legacy='1.25')

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#Exempt Data which isn't self-measurable
newdf = df[['gender', 'age', 'ever_married', 'work_type', 'bmi', 'smoking_status', 'stroke']]

noNandf = newdf.dropna()
noUnknown = noNandf[noNandf['smoking_status'] != 'Unknown']

noUnknown = pd.get_dummies(noUnknown)

#Labels are the values we want to predict
labels = np.array(noUnknown['stroke'])

#Remove the labels from the features
#Axis 1 refers to the columns
noUnknown = noUnknown.drop('stroke', axis=1)

#Saving feature names for later use
feature_list = list(noUnknown.columns)

noUnknown = np.array(noUnknown)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(noUnknown, labels, test_size = 0.10, random_state = 32)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 47)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

for i in range(0, len(predictions)):
    print(predictions[i], ' : ', test_labels[i])

bin6 = []
bin12 = []
bin25 = []
bin50 = []
bin75 = []
bin100 = []

#Calculate proportion of each bin which experienced stroke
bin6res = 0
bin12res = 0
bin25res = 0
bin50res = 0
bin75res = 0
bin100res = 0
bin6mean = 0
bin12mean = 0
bin25mean = 0
bin50mean = 0
bin75mean = 0
bin100mean = 0
bin6prop = 0
bin12prop = 0
bin25prop = 0
bin50prop = 0
bin75prop = 0
bin100prop = 0

for i in range(0, len(predictions)):
    if 0 <= predictions[i] <= 0.06:
        bin6.append(predictions[i])
        #print('Appended to bin6')
        bin6mean += predictions[i]
        if test_labels[i] == 1:
            bin6res += 1
    elif 0.06 < predictions[i] <= 0.125:
        bin12.append(predictions[i])
        #print('Appended to bin12')
        bin12mean += predictions[i]
        if test_labels[i] == 1:
            bin12res += 1
    elif 0.125 < predictions[i] <= 0.25:
        bin25.append(predictions[i])
        #print('Appended to bin25')
        bin25mean += predictions[i]
        if test_labels[i] == 1:
            bin25res += 1
    elif 0.25 < predictions[i] <= 0.50:
        bin50.append(predictions[i])
        #print('Appended to bin50')
        bin50mean += predictions[i]
        if test_labels[i] == 1:
            bin50res += 1
    elif 0.5 < predictions[i] <= 0.75:
        bin75.append(predictions[i])
        #print('Appended to bin75')
        bin75mean += predictions[i]
        if test_labels[i] == 1:
            bin75res += 1
    elif 0.75 < predictions[i] <= 1:
        bin100.append(predictions[i])
        #print('Appended to bin100')
        bin100mean += predictions[i]
        if test_labels[i] == 1:
            bin100res += 1

if not bin6:
    print('bin6 is empty')
    bin6len = 0
else:
    bin6prop = bin6res/len(bin6)
    bin6mean = bin6mean/len(bin6)
    bin6len = len(bin6)

if not bin12:
    print('bin12 is empty')
    bin12len = 0
else:
    bin12prop = bin12res/len(bin12)
    bin12mean = bin12mean/len(bin12)
    bin12len = len(bin12)

if not bin25:
    print('bin25 is empty')
    bin25len = 0
else:
    bin25prop = bin25res/len(bin25)
    bin25mean = bin25mean/len(bin25)
    bin25len = len(bin25)

if not bin50:
    print('bin50 is empty')
    bin50len = 0
else:
    bin50prop = bin50res/len(bin50)
    bin50mean = bin50mean/len(bin50)
    bin50len = len(bin50)

if not bin75:
    print('bin75 is empty')
    bin75len = 0
else:
    bin75prop = bin75res/len(bin75)
    bin75mean = bin75mean/len(bin75)
    bin75len = len(bin75)

if not bin100:
    print('bin100 is empty')
    bin100len = 0
else:
    bin100prop = bin100res/len(bin100)
    bin100mean = bin100mean/len(bin100)
    bin100len = len(bin100)

print('Proportion of strokes in 0 - 6 bin: ', bin6prop, ' || Mean of Bin6: ', bin6mean)
print('Proportion of strokes in 6 - 12 bin: ', bin12prop, ' || Mean of Bin12: ', bin12mean)
print('Proportion of strokes in 12 - 25 bin: ', bin25prop, ' || Mean of Bin25: ', bin25mean)
print('Proportion of strokes in 25 - 50 bin: ', bin50prop, ' || Mean of Bin50: ', bin50mean)
print('Proportion of strokes in 50 - 75 bin: ', bin75prop, ' || Mean of Bin75: ', bin75mean)
print('Proportion of strokes in 75 - 100 bin: ', bin100prop, ' || Mean of Bin100: ', bin100mean)

#Calculating Expected Calibration Error
ECE = (bin6len/len(predictions))*(abs(bin6prop - bin6mean)) + (bin12len/len(predictions))*(abs(bin12prop - bin12mean)) + (bin25len/len(predictions))*(abs(bin25prop - bin25mean)) + (bin50len/len(predictions))*abs(bin50prop - bin50mean) + (bin75len/len(predictions))*abs(bin75prop - bin75mean) + (bin100len/len(predictions))*abs(bin100prop - bin100mean)

print('bin6len/len(predictions): ', bin6len/len(predictions), 'bin12len/len(predictions): ', bin12len/len(predictions), 'bin25len/len(predictions): ', bin25len/len(predictions), 'bin50len/len(predictions): ', bin50len/len(predictions), 'bin75len/len(predictions): ', bin75len/len(predictions), 'bin100len/len(predictions): ', bin100len/len(predictions))
print('Total coverage: ', bin6len/len(predictions) + bin12len/len(predictions) + bin25len/len(predictions) + bin50len/len(predictions) + bin75len/len(predictions) + bin100len/len(predictions))
print('Expected Calibration Error: ', ECE)