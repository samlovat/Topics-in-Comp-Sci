import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import researchpy

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#Exempt Data which aren't nominally categorical
newdf = df[['gender', 'ever_married', 'work_type', 'smoking_status', 'stroke']]

#Convert Categorical Values to Numbers via Mapping
gender_mapping = {'Female': 0, 'Male': 1}
newdf['gender'] = newdf['gender'].map(gender_mapping)
married_mapping = {'Yes': 1, 'No': 0}
newdf['ever_married'] = newdf['ever_married'].map(married_mapping)
worktype_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2}
newdf['work_type'] = newdf['work_type'].map(worktype_mapping)
smoking_mapping = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}
newdf['smoking_status'] = newdf['smoking_status'].map(smoking_mapping)

#Exempt rows with Nan values 
noNandf = newdf.dropna()
noUnknown = noNandf[noNandf['smoking_status'] != 'Unknown']

#Isolate Categorical Fields
stroke = noUnknown['stroke']
gender = noUnknown['gender']
married = noUnknown['ever_married']
work = noUnknown['work_type']
smoke = noUnknown['smoking_status']

#Gather crosstabs of each field with stroke
genderTab = pd.crosstab(gender, stroke)
marriedTab = pd.crosstab(married, stroke)
workTab = pd.crosstab(work, stroke)
smokeTab = pd.crosstab(smoke, stroke)

print(genderTab)
print(marriedTab)
print(workTab)
print(smokeTab)

crosstab, res = researchpy.crosstab(gender, stroke, test='chi-square')
print('Gender Cramers V: ', res)
#genderdf = min(genderTab.shape[0], genderTab.shape[1]) - 1
#print(genderdf)

crosstab, res = researchpy.crosstab(married, stroke, test='chi-square')
print('Marital status Cramers V: ', res)
#marrieddf = min(marriedTab.shape[0], marriedTab.shape[1]) - 1
#print(marrieddf)

crosstab, res = researchpy.crosstab(work, stroke, test='chi-square')
print('Work type Cramers V: ', res)
#workdf = min(workTab.shape[0], workTab.shape[1]) - 1
#print(workdf)

crosstab, res = researchpy.crosstab(smoke, stroke, test='chi-square')
print('Smoking status Cramers V: ', res)
#smokedf = min(smokeTab.shape[0], smokeTab.shape[1]) - 1
#print(smokedf)