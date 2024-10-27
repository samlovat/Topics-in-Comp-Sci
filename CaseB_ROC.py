import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

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

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 47)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Compute the false positive rate (FPR)  
# and true positive rate (TPR) for different classification thresholds 
fpr, tpr, thresholds = roc_curve(test_labels, predictions, pos_label=1)

roc_auc = auc(fpr, tpr)

plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='Luck Based')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Stroke Prediction (Case A and B)')
plt.legend()
plt.show()