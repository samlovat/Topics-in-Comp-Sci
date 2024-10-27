import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
np.set_printoptions(legacy='1.25')

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#Exempt Data which isn't self-measurable
newdf = df[['gender', 'age', 'ever_married', 'work_type', 'bmi', 'smoking_status', 'stroke']]

noNandf = newdf.dropna()
noUnknown = noNandf[noNandf['smoking_status'] != 'Unknown']

'''
# Define the bins and labels (ranges)
bins = [0, 20, 40, 60, 80, 120]
ranges = ["0-20", "21-40", "41-60", "61-80", "81-120"]

# Replace values in the "age" column with the specified age ranges
noUnknown['age'] = pd.cut(noUnknown['age'], bins=bins, labels=ranges, right=True, include_lowest=True)


# Define the bins and labels (ranges)
bins = [0, 18.5, 25, 30, 35, 40, 50, 65, 80, 100]
ranges = ["0-18.5", "18.5-25", "25-30", "30-35", "35-40", "40-50", "50-65", "65-80", "80-100"]

# Replace values in the "age" column with the specified age ranges
noUnknown['bmi'] = pd.cut(noUnknown['bmi'], bins=bins, labels=ranges, right=True, include_lowest=True)
'''
print(noUnknown)

#Convert Categorical Values into a format for machine learning models
noUnknown = pd.get_dummies(noUnknown)

#Labels are the values we want to predict
labels = np.array(noUnknown['stroke'])

#Save for confusion matrix
actual = noUnknown[['stroke']]

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
    if predictions[i] <= 0.5:
        predictions[i] = 0
    else:
        predictions[i] = 1
print(actual)
actual2 = np.array(actual)
actual2 = actual2.flatten()
predictions = np.array(predictions).astype(int)
print(len(test_labels))
print(len(predictions))
actually = np.random.binomial(1, 0.9, size = 1000)
pred = np.random.binomial(1, 0.9, size = 1000)
print(actually)
print(pred)
confusion_matrix = metrics.confusion_matrix(test_labels, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.show()