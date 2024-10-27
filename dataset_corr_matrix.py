import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#Exempt Data which isn't self-measurable
newdf = df[['gender', 'age', 'ever_married', 'work_type', 'bmi', 'smoking_status', 'stroke']]

#Convert Categorical Values to Numbers via Mapping
gender_mapping = {'Female': 0, 'Male': 1}
newdf['gender'] = newdf['gender'].map(gender_mapping)
married_mapping = {'Yes': 1, 'No': 0}
newdf['ever_married'] = newdf['ever_married'].map(married_mapping)
worktype_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2}
newdf['work_type'] = newdf['work_type'].map(worktype_mapping)
smoking_mapping = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}
newdf['smoking_status'] = newdf['smoking_status'].map(smoking_mapping)
residence_mapping = {'Urban': 0, 'Rural': 1}
newdf['Residence_type'] = newdf['Residence_type'].map(residence_mapping)

#Exempt rows with Nan values 
noNandf = newdf.dropna()
noUnknown = noNandf[noNandf['smoking_status'] != 'Unknown']


myMatrix = noUnknown.corr(method="spearman").round(3)
sns.heatmap(data = myMatrix, annot = True)
print(myMatrix)