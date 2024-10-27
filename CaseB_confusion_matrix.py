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