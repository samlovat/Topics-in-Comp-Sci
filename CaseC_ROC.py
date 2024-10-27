import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
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


#Remove the labels from the features
#Axis 1 refers to the columns
actual = noUnknown['stroke']
noUnknown = noUnknown.drop(['stroke'], axis=1)
print(actual.value_counts())

#ros = RandomOverSampler(sampling_strategy=1) # Float


print(actual.value_counts())

#Labels are the values we want to predict
labels = np.array(actual)


#Saving feature names for later use
feature_list = list(noUnknown.columns)

noUnknown = np.array(noUnknown)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(noUnknown, labels, test_size = 0.10, random_state = 32)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

train_feature_resampled, train_labels_resampled = ADASYN().fit_resample(train_features, train_labels)
print(sorted(Counter(train_labels_resampled).items()))



# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 47)

# Train the model on training data
rf.fit(train_feature_resampled, train_labels_resampled)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

fpr, tpr, thresholds = roc_curve(test_labels, predictions, pos_label=1)

roc_auc = auc(fpr, tpr)

plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='Luck Based')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Stroke Prediction (Case C)')
plt.legend()
plt.show()