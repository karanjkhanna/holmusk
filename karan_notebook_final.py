
# coding: utf-8

# In[1]:


# Import pandas
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Read 'clinical_data.csv' into a DataFrame: clinical_data
clinical_data = pd.read_csv('clinical_data.csv')

# Read 'demographics.csv' into a DataFrame: demographics
demographics = pd.read_csv('demographics.csv')

# Read 'bill_id.csv' into a DataFrame: bill_id
bill_id = pd.read_csv('bill_id.csv')

# Read 'bill_amount.csv' into a DataFrame: bill_amount
bill_amount = pd.read_csv('bill_amount.csv')

# Print the first five rows of clinical_data
print(clinical_data.head())


# In[2]:


# Merge bill_id and bill_amount, sharing the common bill_id column, using Inner Join
# Why Inner Join? Because every bill must have a corresponding amount, and vice-versa
bill = pd.merge(left=bill_id, right=bill_amount, on="bill_id", how="inner")
bill.info()


# In[3]:


# Convert the date_of_admission column into pandas date-time
bill['date_of_admission'] = pd.to_datetime(bill['date_of_admission'])


# In[4]:


print(bill['date_of_admission'].head())


# In[5]:


# Verifying changes made
bill.info()


# In[6]:


# Rename id in clinical_data to patient_id, because they are the same
clinical_data.rename(columns={'id':'patient_id'}, inplace=True)


# In[7]:


# Merge demographics and clinical_data, on the common patient_id column, using Right Join
# Why Right Join? Because it is possible for a patient to be admitted more than once
patient = pd.merge(left=demographics, right=clinical_data, on="patient_id", how="right")
patient.info()


# In[8]:


# Convert date_of_admission, date_of_discharge, and date_of_birth into pandas date-time
patient['date_of_admission'] = pd.to_datetime(patient['date_of_admission'])
patient['date_of_discharge'] = pd.to_datetime(patient['date_of_discharge'])
patient['date_of_birth'] = pd.to_datetime(patient['date_of_birth'])


# In[9]:


# Calculate age of the patient at the time of admission
# We do this by calculating the time that has passed since the birth of the patient, until admission
patient['age'] = patient['date_of_admission'] - patient['date_of_birth']

# The result of the above is calculated in days, hence we need to convert it into whole years
patient['age'] = patient['age'].astype('timedelta64[Y]')
patient['age'] = patient['age'].astype(int)

# Verifying that the above steps went through smoothly
print(patient['age'].head())


# In[10]:


# Create age categories
patient['age_cat'] = np.nan
lst = [patient]

for col in lst:
    col.loc[(col['age'] >= 18) & (col['age'] <= 35), 'age_cat'] = 'Young Adult'
    col.loc[(col['age'] > 35) & (col['age'] <= 55), 'age_cat'] = 'Senior Adult'
    col.loc[col['age'] > 55, 'age_cat'] = 'Elder'
    
patient['age_cat'] = patient['age_cat'].astype('category')


# In[11]:


patient['height_m'] = patient['height'] / 100
patient['height_m'].head()


# In[12]:


patient['height_m_squared'] = patient['height_m'] ** 2
patient['height_m_squared'].head()


# In[13]:


# Calculating BMI (Body Mass Index)
patient['bmi'] = patient['weight'] / patient['height_m_squared']
patient['bmi'].head()


# In[14]:


patient['bmi_cat'] = np.nan
lst = [patient]

for col in lst:
    col.loc[col['bmi'] < 19, 'bmi_cat'] = 'Underweight'
    col.loc[(col['bmi'] >= 19) & (col['bmi'] <= 29), 'bmi_cat'] = 'Healthy'
    col.loc[col['bmi'] > 29, 'bmi_cat'] = 'Overweight'
    
patient['bmi_cat'] = patient['bmi_cat'].astype('category')


# In[15]:


# Move age, age_cat, bmi, bmi_cat from the right-side end to the appropriate demographics section within the dataframe
patient_cols = list(patient.columns.values)
patient = patient[['patient_id', 'age', 'age_cat', 'bmi', 'bmi_cat', 'gender', 'race', 'resident_status', 'date_of_birth', 'date_of_admission', 'date_of_discharge', 'medical_history_1', 'medical_history_2', 'medical_history_3', 'medical_history_4', 'medical_history_5', 'medical_history_6', 'medical_history_7', 'preop_medication_1', 'preop_medication_2', 'preop_medication_3', 'preop_medication_4', 'preop_medication_5', 'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2', 'lab_result_3', 'weight', 'height', 'height_m', 'height_m_squared']]


# In[16]:


# Convert gender into categorical
patient['gender'] = patient['gender'].astype('category')
patient['gender'].head()

# We assume that f stand for Female, and m for Male, hence making the below replacements
patient['gender'] = patient['gender'].replace('f', 'Female')
patient['gender'] = patient['gender'].replace('m', 'Male')

# Remove redundant categories, f and m
patient['gender'] = patient['gender'].cat.remove_unused_categories()
patient['gender'].head()


# In[17]:


# Convert race into categorical
patient['race'] = patient['race'].astype('category')
patient['race'].head()

# chinese with a lowercase 'c', and Chinese with an uppercase 'C', are different categories which doesn't make sense
# India is not a race, it is assumed it stands for Indian
patient['race'] = patient['race'].replace('chinese', 'Chinese')
patient['race'] = patient['race'].replace('India', 'Indian')

# Remove redundant categories, 'chinese' and 'India'
patient['race'] = patient['race'].cat.remove_unused_categories()
patient['race'].head()


# In[18]:


# Convert resident_status into categorical
patient['resident_status'] = patient['resident_status'].astype('category')
patient['resident_status'].head()

# It is assumed that Singaporean citizen and Singaporean are one and the same
patient['resident_status'] = patient['resident_status'].replace('Singapore citizen', 'Singaporean')

# Remove redundant category, 'Singapore citizen'
patient['resident_status'] = patient['resident_status'].cat.remove_unused_categories()
patient['resident_status'].head()


# In[19]:


# Convert medical_history_1 into categorical
patient['medical_history_1'] = patient['medical_history_1'].astype('category')
patient['medical_history_1'].head()


# In[20]:


# Convert medical history_2 into categorical
patient['medical_history_2'] = patient['medical_history_2'].fillna('999')
patient['medical_history_2'] = patient['medical_history_2'].astype(int)
patient['medical_history_2'] = patient['medical_history_2'].astype('category')
patient['medical_history_2'].head()


# In[21]:


# It is assumed that 'No' corresponds to 0, and 'Yes' corresponds to 1.
patient['medical_history_3'] = patient['medical_history_3'].replace('No', '0')
patient['medical_history_3'] = patient['medical_history_3'].replace('Yes', '1')
patient['medical_history_3'] = patient['medical_history_3'].astype(int)

# Convert medical_history_3 into categorical
patient['medical_history_3'] = patient['medical_history_3'].astype('category')
patient['medical_history_3'].head()

# Remove unwanted categories, 'No' and 'Yes'
patient['medical_history_3'] = patient['medical_history_3'].cat.remove_unused_categories()
patient['medical_history_3'].head()


# In[22]:


# Convert medical_history_4 into categorical
patient['medical_history_4'] = patient['medical_history_4'].astype('category')
patient['medical_history_4'].head()


# In[23]:


# Convert medical_history_5 into categorical
patient['medical_history_5'] = patient['medical_history_5'].fillna('999')
patient['medical_history_5'] = patient['medical_history_5'].astype(int)
patient['medical_history_5'] = patient['medical_history_5'].astype('category')
patient['medical_history_5'].head()


# In[24]:


# Convert medical_history_6 into categorical
patient['medical_history_6'] = patient['medical_history_6'].astype('category')
patient['medical_history_6'].head()


# In[25]:


# Convert medical_history_7 into categorical
patient['medical_history_7'] = patient['medical_history_7'].astype('category')
patient['medical_history_7'].head()


# In[26]:


# Convert preop_medication_1 into categorical
patient['preop_medication_1'] = patient['preop_medication_1'].astype('category')
patient['preop_medication_1'].head()


# In[27]:


# Convert preop_medication_2 into categorical
patient['preop_medication_2'] = patient['preop_medication_2'].astype('category')
patient['preop_medication_2'].head()


# In[28]:


# Convert preop_medication_3 into categorical
patient['preop_medication_3'] = patient['preop_medication_3'].astype('category')
patient['preop_medication_3'].head()


# In[29]:


# Convert preop_medication_4 into categorical
patient['preop_medication_4'] = patient['preop_medication_4'].astype('category')
patient['preop_medication_4'].head()


# In[30]:


# Convert preop_medication_5 into categorical
patient['preop_medication_5'] = patient['preop_medication_5'].astype('category')
patient['preop_medication_5'].head()


# In[31]:


# Convert preop_medication_6 into categorical
patient['preop_medication_6'] = patient['preop_medication_6'].astype('category')
patient['preop_medication_6'].head()


# In[32]:


# Convert symptom_1 into categorical
patient['symptom_1'] = patient['symptom_1'].astype('category')
patient['symptom_1'].head()


# In[33]:


# Convert symptom_2 into categorical
patient['symptom_2'] = patient['symptom_2'].astype('category')
patient['symptom_2'].head()


# In[34]:


# Convert symptom_3 into categorical
patient['symptom_3'] = patient['symptom_3'].astype('category')
patient['symptom_3'].head()


# In[35]:


# Convert symptom_4 into categorical
patient['symptom_4'] = patient['symptom_4'].astype('category')
patient['symptom_4'].head()


# In[36]:


# Convert symptom_5 into categorical
patient['symptom_5'] = patient['symptom_5'].astype('category')
patient['symptom_5'].head()


# In[37]:


# Verifying changes made
patient.info()
# For now, we will ignore the missing values in 'medical_history_2' (233) and 'medical_history_5' (304)


# In[38]:


# Merge the two dataframes, patient and bill, using a Right Join, on the common patient_id and date_of_admission columns
# Why Right Join? Because a patient can have multiple bills
# It is important to use the date of admission along with patient id for this join
merged = pd.merge(patient, bill, on=['patient_id', 'date_of_admission'], how="right")


# In[39]:


# Make sure patient and bill were merged smoothly
print(merged.head())


# In[40]:


# Export the three dataframes to CSV
merged.to_csv('merged.csv')
patient.to_csv('patient.csv')
bill.to_csv('bill.csv')


# In[41]:


# Sanity check
print(patient.info())


# In[42]:


# Encoding all categorical variables
from sklearn.preprocessing import LabelEncoder
# Gender
le = LabelEncoder()
le.fit(merged['gender'].drop_duplicates()) 
merged['gender'] = le.transform(merged['gender'])
# Age Category
le.fit(merged['age_cat'].drop_duplicates()) 
merged['age_cat'] = le.transform(merged['age_cat'])
# BMI Category
le.fit(merged['bmi_cat'].drop_duplicates()) 
merged['bmi_cat'] = le.transform(merged['bmi_cat'])
# Race
le.fit(merged['race'].drop_duplicates()) 
merged['race'] = le.transform(merged['race'])
# Resident Status
le.fit(merged['resident_status'].drop_duplicates()) 
merged['resident_status'] = le.transform(merged['resident_status'])


# In[43]:


# For loop to encode medical_history_x
for hist in range(1, 8):
    le.fit(merged['medical_history_{0}'.format(hist)].drop_duplicates()) 
    merged['medical_history_{0}'.format(hist)] = le.transform(merged['medical_history_{0}'.format(hist)])


# In[44]:


# For loop to encode preop_medication_x
for preop in range(1, 7):
    le.fit(merged['preop_medication_{0}'.format(preop)].drop_duplicates()) 
    merged['preop_medication_{0}'.format(preop)] = le.transform(merged['preop_medication_{0}'.format(preop)])


# In[45]:


# For loop to encode symptom_x
for symp in range(1, 6):
    le.fit(merged['symptom_{0}'.format(symp)].drop_duplicates()) 
    merged['symptom_{0}'.format(symp)] = le.transform(merged['symptom_{0}'.format(symp)])


# In[46]:


# Calculate correlation for amount
merged.corr()['amount'].sort_values(ascending=False)


# In[47]:


# Generate heatmap of correlation of amount, using seaborn
f, ax = plt.subplots(figsize=(10, 8))
corr = merged.corr()
sns.heatmap(corr, vmin=-0.15, vmax=0.20, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(10,240,as_cmap=True),
            square=True, ax=ax, xticklabels=True, yticklabels=True)


# In[48]:


# Generating distribution table using Bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook()
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
p = figure(title="Distribution of bill amount",tools="save",
            background_fill_color="#E8DDCB")
hist, edges = np.histogram(merged['amount'])
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
p.xaxis.axis_label = 'x'
p.yaxis.axis_label = 'Pr(x)'
show(gridplot(p,ncols = 2, plot_width=400, plot_height=400, toolbar_location=None))


# In[49]:


# Distribution Plot using Seaborn
plt.figure(figsize=(12,5))
plt.title("Amount Distribution Plot")
ax = sns.distplot(merged["amount"])


# In[50]:


# Generate distribution table, by resident status
f = plt.figure(figsize=(18,5))

ax=f.add_subplot(131)
sns.distplot(merged[(merged.resident_status == 2)]['amount'],color='c',ax=ax)
ax.set_title('Distribution of amount for Singaporeans')

ax=f.add_subplot(132)
sns.distplot(merged[(merged.resident_status == 1)]['amount'],color='b',ax=ax)
ax.set_title('Distribution of amount for Permanent Residents')

ax=f.add_subplot(133)
sns.distplot(merged[(merged.resident_status == 0)]['amount'],color='r',ax=ax)
ax.set_title('Distribution of amount for Foreigners')


# In[51]:


# Box plot for amount paid, by Resident Status
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by Resident Status")
sns.boxplot(x="amount", y="resident_status", data=merged, orient="h", palette = 'rainbow')


# In[52]:


# Amount Distribution, Race
f = plt.figure(figsize=(18,5))

ax=f.add_subplot(141)
sns.distplot(merged[(merged.race == 3)]['amount'],color='c',ax=ax)
ax.set_title('Amount Distribution, Race: Others')

ax=f.add_subplot(142)
sns.distplot(merged[(merged.race == 2)]['amount'],color='b',ax=ax)
ax.set_title('Amount Distribution, Race: Malay')

ax=f.add_subplot(143)
sns.distplot(merged[(merged.race == 1)]['amount'],color='r',ax=ax)
ax.set_title('Amount Distribution, Race: Indian')

ax=f.add_subplot(144)
sns.distplot(merged[(merged.race == 0)]['amount'],color='g',ax=ax)
ax.set_title('Amount Distribution, Race: Chinese')


# In[53]:


# Amount Box plot, Race
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by Race")
sns.boxplot(x="amount", y="race", data=merged, orient="h", palette = 'rainbow')


# In[54]:


# Age Distribution Plot
plt.figure(figsize=(12,5))
plt.title("Age Distribution")
ax = sns.distplot(merged["age"], color = 'm')


# In[55]:


# Amount Distribution, Age Category
f = plt.figure(figsize=(18,5))

ax=f.add_subplot(131)
sns.distplot(merged[(merged.age_cat == 2)]['amount'],color='c',ax=ax)
ax.set_title('Amount Distribution, Young Adult')

ax=f.add_subplot(132)
sns.distplot(merged[(merged.age_cat == 1)]['amount'],color='b',ax=ax)
ax.set_title('Amount Distribution, Senior Adult')

ax=f.add_subplot(133)
sns.distplot(merged[(merged.age_cat == 0)]['amount'],color='r',ax=ax)
ax.set_title('Amount Distribution, Elder')


# In[56]:


# Box Plot, Amount by Age Category
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by Age Category")
sns.boxplot(x="amount", y="age_cat", data=merged, orient="h", palette = 'rainbow')


# In[57]:


# BMI Distribution Plot
plt.figure(figsize=(12,5))
plt.title("BMI Distribution")
ax = sns.distplot(merged["bmi"], color = 'y')


# In[58]:


# Overweight BMI Distribution
plt.figure(figsize=(12,5))
plt.title("Overweight BMI Distribution")
ax = sns.distplot(merged[merged.bmi_cat == 1]['bmi'], color = 'r')


# In[59]:


# Underweight & Healthy BMI Distribution Plot
plt.figure(figsize=(12,5))
plt.title("Underweight & Healthy BMI Distribution")
ax = sns.distplot(merged[merged.bmi_cat != 1]['bmi'], color = 'b')


# In[60]:


# Amount Distribution, Amount by BMI
f = plt.figure(figsize=(18,5))

ax=f.add_subplot(131)
sns.distplot(merged[(merged.bmi_cat == 2)]['amount'],color='c',ax=ax)
ax.set_title('Amount Distribution by BMI, Underweight')

ax=f.add_subplot(132)
sns.distplot(merged[(merged.bmi_cat == 1)]['amount'],color='b',ax=ax)
ax.set_title('Amount Distribution by BMI, Overweight')

ax=f.add_subplot(133)
sns.distplot(merged[(merged.bmi_cat == 0)]['amount'],color='r',ax=ax)
ax.set_title('Amount Distribution by BMI, Healthy')


# In[61]:


# Box Plot for BMI
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by BMI (Body Mass Index)")
sns.boxplot(x="amount", y="bmi_cat", data=merged, orient="h", palette = 'rainbow')


# In[62]:


# Count Plot for BMI
plt.figure(figsize=(12,5))
plt.title("Count plot for BMI (Body Mass Index)")
sns.countplot(y="bmi_cat", data=merged, color="c");


# In[63]:


# Calculating BMI's correlation with other variables
bmi_cat_corr = merged.corr()['bmi_cat'].sort_values(ascending=False)
print(bmi_cat_corr)


# In[64]:


# Amount Distribution, Gender
f = plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(merged[(merged.gender == 1)]['amount'],color='c',ax=ax)
ax.set_title('Amount Distribution, Gender = Male')

ax=f.add_subplot(122)
sns.distplot(merged[(merged.gender == 0)]['amount'],color='r',ax=ax)
ax.set_title('Amount Distribution, Gender = Female')


# In[65]:


# Box Plot, Gender
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by gender")
sns.boxplot(x="amount", y="gender", data=merged, orient="h", palette = 'rainbow')


# In[66]:


# Link between BMI & Medical History
merged.corr()['bmi'].sort_values(ascending=False)


# In[67]:


# Amount Distribution by Med Hist 1
f = plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(merged[(merged.medical_history_1 == 1)]['amount'],color='c',ax=ax)
ax.set_title('Amount Distribution, Medical History 1 = Yes')

ax=f.add_subplot(122)
sns.distplot(merged[(merged.medical_history_1 == 0)]['amount'],color='r',ax=ax)
ax.set_title('Amount Distribution, Medical History 1 = No')


# In[68]:


# Box Plot, Med Hist 1
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by medical_history_1")
sns.boxplot(x="amount", y="medical_history_1", data=merged, orient="h", palette = 'rainbow')


# In[69]:


# Amount Distribution, Medical History 6
f = plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(merged[(merged.medical_history_6 == 1)]['amount'],color='c',ax=ax)
ax.set_title('Amount Distribution, Medical History 6 = Yes')

ax=f.add_subplot(122)
sns.distplot(merged[(merged.medical_history_6 == 0)]['amount'],color='r',ax=ax)
ax.set_title('Amount Distribution, Medical History 6 = No')


# In[70]:


# Box Plot, Amount by Med Hist_6
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by medical_history_6")
sns.boxplot(x="amount", y="medical_history_6", data=merged, orient="h", palette = 'rainbow')


# In[71]:


# Symptom 1 Distribution
f = plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(merged[(merged.symptom_1 == 1)]['amount'],color='c',ax=ax)
ax.set_title('Distribution of amount paid by patients WITH symptom_1')

ax=f.add_subplot(122)
sns.distplot(merged[(merged.symptom_1 == 0)]['amount'],color='r',ax=ax)
ax.set_title('Distribution of amount paid by patients WITHOUT symptom_1')


# In[72]:


# Symptom 1 Box plot
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by symptom_1")
sns.boxplot(x="amount", y="symptom_1", data=merged, orient="h", palette = 'rainbow')


# In[73]:


# Symptom 2 Distribution
f = plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(merged[(merged.symptom_2 == 1)]['amount'],color='c',ax=ax)
ax.set_title('Distribution of amount paid by patients WITH symptom_2')

ax=f.add_subplot(122)
sns.distplot(merged[(merged.symptom_2 == 0)]['amount'],color='r',ax=ax)
ax.set_title('Distribution of amount paid by patients WITHOUT symptom_2')


# In[74]:


# Box Plot, Symptom 2
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by symptom_2")
sns.boxplot(x="amount", y="symptom_2", data=merged, orient="h", palette = 'rainbow')


# In[75]:


# Distribution, Symptom 3
f = plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(merged[(merged.symptom_3 == 1)]['amount'],color='c',ax=ax)
ax.set_title('Distribution of amount paid by patients WITH symptom_3')

ax=f.add_subplot(122)
sns.distplot(merged[(merged.symptom_3 == 0)]['amount'],color='r',ax=ax)
ax.set_title('Distribution of amount paid by patients WITHOUT symptom_3')


# In[76]:


# Box Plot, Symptom 3
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by symptom_3")
sns.boxplot(x="amount", y="symptom_3", data=merged, orient="h", palette = 'rainbow')


# In[77]:


# Distribution, Symptom 4
f = plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(merged[(merged.symptom_4 == 1)]['amount'],color='c',ax=ax)
ax.set_title('Distribution of amount paid by patients WITH symptom_4')

ax=f.add_subplot(122)
sns.distplot(merged[(merged.symptom_4 == 0)]['amount'],color='r',ax=ax)
ax.set_title('Distribution of amount paid by patients WITHOUT symptom_4')


# In[78]:


# Box Plot, Symptom 4
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by symptom_4")
sns.boxplot(x="amount", y="symptom_4", data=merged, orient="h", palette = 'rainbow')


# In[79]:


# Symptom 5 Distribution
f = plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(merged[(merged.symptom_5 == 1)]['amount'],color='c',ax=ax)
ax.set_title('Distribution of amount paid by patients WITH symptom_5')

ax=f.add_subplot(122)
sns.distplot(merged[(merged.symptom_5 == 0)]['amount'],color='r',ax=ax)
ax.set_title('Distribution of amount paid by patients WITHOUT symptom_5')


# In[80]:


# Box Plot, Symptom 5
plt.figure(figsize=(12,5))
plt.title("Box plot for amount paid, by symptom_5")
sns.boxplot(x="amount", y="symptom_5", data=merged, orient="h", palette = 'rainbow')


# In[81]:


# Count, by race
sns.catplot(x="resident_status", kind="count", palette="ch:.25", data=merged);
plt.title('Count, by resident status', fontsize=15)


# In[82]:


# Count gender
sns.catplot(x="gender", kind="count", palette="ch:.25", data=merged);
plt.title('Count, by gender', fontsize=15)


# In[83]:


# Amount, by gender
sns.barplot(x=merged['gender'],y = merged['amount'])
plt.title('amount, by gender', fontsize=15)


# In[84]:


# Med Hist 1 by gender
sns.barplot(x=merged['gender'],y = merged['medical_history_1'])
plt.title('medical_history_1, by gender', fontsize=15)


# In[85]:


# Med Hist 6 by gender
sns.barplot(x=merged['gender'],y = merged['medical_history_6'])
plt.title('medical_history_6, by gender', fontsize=15)


# In[86]:


# Symptom 1 by gender
sns.barplot(x=merged['gender'],y = merged['symptom_1'])
plt.title('symptom_1, by gender', fontsize=15)


# In[87]:


# Symptom 2 by gender
sns.barplot(x=merged['gender'],y = merged['symptom_2'])
plt.title('symptom_2, by gender', fontsize=15)


# In[88]:


# Symptom 3 by gender
sns.barplot(x=merged['gender'],y = merged['symptom_3'])
plt.title('symptom_3, by gender', fontsize=15)


# In[89]:


# Symptom 4 by gender
sns.barplot(x=merged['gender'],y = merged['symptom_4'])
plt.title('symptom_4, by gender', fontsize=15)


# In[90]:


# Symptom 5 by gender
sns.barplot(x=merged['gender'],y = merged['symptom_5'])
plt.title('symptom_5, by gender', fontsize=15)


# In[91]:


# Count, by race
sns.catplot(x="race", kind="count", palette="ch:.25", data=merged);
plt.title('Count, by race', fontsize=15)


# In[92]:


# Amount by race
sns.barplot(x=merged['race'],y = merged['amount'])
plt.title('amount, by race', fontsize=15)


# In[93]:


# Med Hist 1 by race
sns.barplot(x=merged['race'],y = merged['medical_history_1'])
plt.title('medical_history_1, by race', fontsize=15)


# In[94]:


# Med hist 6 by race
sns.barplot(x=merged['race'],y = merged['medical_history_6'])
plt.title('medical_history_6, by race', fontsize=15)


# In[95]:


# Symptom 1 by race
sns.barplot(x=merged['race'],y = merged['symptom_1'])
plt.title('symptom_1, by race', fontsize=15)


# In[96]:


# Symptom 2 by race
sns.barplot(x=merged['race'],y = merged['symptom_2'])
plt.title('symptom_2, by race', fontsize=15)


# In[97]:


# Symptom 3 by race
sns.barplot(x=merged['race'],y = merged['symptom_3'])
plt.title('symptom_3, by race', fontsize=15)


# In[98]:


# Symptom_4 by race
sns.barplot(x=merged['race'],y = merged['symptom_4'])
plt.title('symptom_4, by race', fontsize=15)


# In[99]:


# Symptom 5 by race
sns.barplot(x=merged['race'],y = merged['symptom_5'])
plt.title('symptom_5, by race', fontsize=15)


# In[100]:


# Count, by race
sns.catplot(x="age_cat", kind="count", palette="ch:.25", data=merged);
plt.title('Count, by age category', fontsize=15)


# In[101]:


# Amount, by age cat
sns.barplot(x=merged['age_cat'],y = merged['amount'])
plt.title('amount, by age category', fontsize=15)


# In[102]:


# Med Hist 1, by age cat
sns.barplot(x=merged['age_cat'],y = merged['medical_history_1'])
plt.title('medical_history_1, by age category', fontsize=15)


# In[103]:


# Med Hist 6 by age cat
sns.barplot(x=merged['age_cat'],y = merged['medical_history_6'])
plt.title('medical_history_6, by age category', fontsize=15)


# In[104]:


# Symptom 1 by age cat
sns.barplot(x=merged['age_cat'],y = merged['symptom_1'])
plt.title('symptom_1, by age category', fontsize=15)


# In[105]:


# Symptom 2 by age cat
sns.barplot(x=merged['age_cat'],y = merged['symptom_2'])
plt.title('symptom_2, by age category', fontsize=15)


# In[106]:


# Symptom 3 by age cat
sns.barplot(x=merged['age_cat'],y = merged['symptom_3'])
plt.title('symptom_3, by age category', fontsize=15)


# In[107]:


# Symptom 4 by age cat
sns.barplot(x=merged['age_cat'],y = merged['symptom_4'])
plt.title('symptom_4, by age_cat', fontsize=15)


# In[108]:


# Symptom 5 by age cat
sns.barplot(x=merged['age_cat'],y = merged['symptom_5'])
plt.title('symptom_5, by age category', fontsize=15)


# In[109]:


# Count, by BMI category
sns.catplot(x="resident_status", kind="count", palette="ch:.25", data=merged);
plt.title('Count, by Resident Status', fontsize=15)


# In[110]:


# Count, by BMI category
sns.catplot(x="bmi_cat", kind="count", palette="ch:.25", data=merged);
plt.title('Count, by BMI category', fontsize=15)


# In[111]:


# Count, by BMI category
sns.catplot(x="bmi_cat", hue="gender", kind="count", palette="ch:.25", data=merged);
plt.title('Count, by BMI category + Gender', fontsize=15)


# In[112]:


# Amount by BMI Cat
sns.barplot(x=merged['bmi_cat'],y = merged['amount'])
plt.title('amount, by BMI category', fontsize=15)


# In[113]:


# Amt by BMI + Gender
sns.catplot(x="bmi_cat", y="amount", hue="gender", kind="bar", data=merged)
plt.title('Amount by BMI + Gender', fontsize=15)


# In[114]:


# Med Hist 1 by BMI Cat
sns.barplot(x=merged['bmi_cat'],y = merged['medical_history_1'])
plt.title('medical_history_1, by BMI category', fontsize=15)


# In[115]:


# Med Hist 2 by BMI Cat
sns.barplot(x=merged['bmi_cat'],y = merged['medical_history_2'])
plt.title('medical_history_2, by BMI category', fontsize=15)


# In[116]:


# Med Hist 3 by BMI cat
sns.barplot(x=merged['bmi_cat'],y = merged['medical_history_3'])
plt.title('medical_history_3, by BMI category', fontsize=15)


# In[117]:


# BMI Cat med hist 5
sns.barplot(x=merged['bmi_cat'],y = merged['medical_history_5'])
plt.title('medical_history_5, by BMI category', fontsize=15)


# In[118]:


# Med Hist 6 by BMI Cat
sns.barplot(x=merged['bmi_cat'],y = merged['medical_history_6'])
plt.title('medical_history_6, by BMI category', fontsize=15)


# In[119]:


# Preop med 4, by BMI cat
sns.barplot(x=merged['bmi_cat'],y = merged['preop_medication_4'])
plt.title('preop_medication_4, by BMI category', fontsize=15)


# In[120]:


# Med Hist 3 by BMI + Gender
sns.catplot(x="gender", y="medical_history_3", hue="bmi_cat", kind="bar", data=merged)
plt.title('Medical History 3 by BMI + Gender', fontsize=15)


# In[121]:


# MEd Hist 1 by BMI + Gender
sns.catplot(x="gender", y="medical_history_1", hue="bmi_cat", kind="bar", data=merged)
plt.title('Medical History 1 by BMI + Gender', fontsize=15)


# In[122]:


# Amt by Race + Gender
sns.catplot(x="race", y="amount", hue="gender", kind="bar", data=merged)
plt.title('Amount by Race + Gender', fontsize=15)


# In[130]:


bokeh serve --show myapp.py


# In[123]:


# Symptom 5 = NO
p = figure(plot_width=500, plot_height=450, title='Scatterplot, Amount Progression By Age, [Symptom 5 = NO]')
p.circle(x=merged[(merged.symptom_5 == 0)].age,y=merged[(merged.symptom_5 == 0)].amount, size=7, line_color="navy", fill_color="pink", fill_alpha=0.9)

show(p)


# In[124]:


# Symptom 5 = YES
p = figure(plot_width=500, plot_height=450, title='Scatterplot, Amount Progression By Age, [Symptom 5 = YES]')
p.circle(x=merged[(merged.symptom_5 == 1)].age,y=merged[(merged.symptom_5 == 1)].amount, size=7, line_color="navy", fill_color="red", fill_alpha=0.9)
show(p)


# In[125]:


# Symptom 5 = YES and Symptom 5 = NO
sns.lmplot(x="age", y="amount", hue="symptom_5", data=merged, palette = 'inferno_r', size = 7)
ax.set_title('symptom_5 YES and symptom_5 NO')


# In[126]:


# Healthy
p = figure(plot_width=500, plot_height=450, title="Scatterplot, Amount Progression By Age, Healthy Weight")
p.circle(x=merged[(merged.bmi_cat == 0)].age,y=merged[(merged.bmi_cat == 0)].amount, size=7, line_color="navy", fill_color="cyan", fill_alpha=0.9)

show(p)


# In[127]:


# Overweight
p = figure(plot_width=500, plot_height=450, title="Scatterplot, Amount Progression By Age, Overweight")
p.circle(x=merged[(merged.bmi_cat == 1)].age,y=merged[(merged.bmi_cat == 1)].amount, size=7, line_color="navy", fill_color="red", fill_alpha=0.9)

show(p)


# In[128]:


# Underweight
p = figure(plot_width=500, plot_height=450, title="Scatterplot, Amount Progression By Age, Underweight")
p.circle(x=merged[(merged.bmi_cat == 2)].age,y=merged[(merged.bmi_cat == 2)].amount, size=7, line_color="navy", fill_color="violet", fill_alpha=0.9)

show(p)


# In[129]:


# All Weights
sns.lmplot(x="age", y="amount", hue="bmi_cat", data=merged, palette = 'inferno_r', size = 7)
ax.set_title('Amount Progression By Age, All Weight Categories')

