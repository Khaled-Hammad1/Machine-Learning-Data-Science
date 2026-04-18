
'''
Prepared by: Khaled Hammad 		   ID: 1220857	  Sec: 1
Prepared by: Mohammad Shamasneh	   ID: 1220092    Sec: 1

'''

import pandas
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import title

###################################################################################
#############################--|  task 1  |---#####################################
###################################################################################

data = pandas.read_csv("customer_data.csv")

###################################################################################
# ------------| to show if theis non binary value at binary feature  |--------------

# for col in ['Gender', 'ProductType', 'ChurnStatus']:
#     unique_vals = data[col].unique()
#     print(f"Unique values in column '{col}': {unique_vals}")
#
#     if set(unique_vals).issubset({0, 1}):
#         print(f"column '{col}' contains only 0 and 1")
#     else:
#         print(f"column '{col}' contains values other than 0 and 1")

###################################################################################

print(data.head())
print("-----------------------------------------------------------")
print(data.info())
print("-----------------------------------------------------------")
print(data.describe())
print("----------------------| done task 1 |-------------------------------------")




###################################################################################
#############################--|  task 2  |---#####################################
###################################################################################

print(data.isnull().sum())

data['Age'] = data['Age'].fillna(data['Age'].median())
data['Tenure'] = data['Tenure'].fillna(data['Tenure'].median())
data['SupportCalls'] = data['SupportCalls'].fillna(data['SupportCalls'].median())
data['Income'] = data.groupby('ProductType')['Income'].transform(lambda x: x.fillna(x.median()))

###################################################################################
#############################--|  task 3  |---#####################################
###################################################################################

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle('Data before cleaning the outliers', fontsize=16, fontweight='bold')
sns.boxplot(x=data['Age'], ax=axes[0,0])
axes[0,0].set_title("Age outlier")
sns.boxplot(x=data['Tenure'], ax=axes[0,1])
axes[0,1].set_title("Tenure outlier")
sns.boxplot(x=data['SupportCalls'], ax=axes[1,0])
axes[1,0].set_title("SupportCalls outlier")
sns.boxplot(x=data['Income'], ax=axes[1,1])
axes[1,1].set_title("Income outlier")
plt.tight_layout()
plt.show()

######################################Clean Data from outlier############################################################

for col in ['Age', 'Tenure', 'SupportCalls','Income']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3- Q1
    lowerBound = Q1 - 1.5 * IQR
    upperBound = Q3 + 1.5 * IQR
    medianValue = data[col].median()
    data[col] =data[col].apply(lambda x: medianValue if x < lowerBound or x > upperBound else x) # we change the outlier by median of column instead of remove all row

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle('Data After cleaning the outliers', fontsize=16, fontweight='bold')
sns.boxplot(x=data['Age'], ax=axes[0,0])
axes[0,0].set_title("Age outlier")
sns.boxplot(x=data['Tenure'], ax=axes[0,1])
axes[0,1].set_title("Tenure outlier")
sns.boxplot(x=data['SupportCalls'], ax=axes[1,0])
axes[1,0].set_title("SupportCalls outlier")
sns.boxplot(x=data['Income'], ax=axes[1,1])
axes[1,1].set_title("Income outlier")
plt.tight_layout()
plt.show()


###################################################################################
#############################--|  task 4  |---#####################################
####################################################################################
 
for col in ['Age', 'Tenure', 'SupportCalls','Income']:
    minv = data[col].min()
    maxv = data[col].max()
    data[col] = (data[col]-minv)/(maxv-minv)



###################################################################################
#############################--|  task 5  |---#####################################
####################################################################################

# Visualize the distribution of key numerical features using histograms .

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle('Distribution of Data', fontsize=16, fontweight='bold')
sns.histplot(x=data['Age'], ax=axes[0,0])
axes[0,0].set_title("Age distribution")
sns.histplot(x=data['Tenure'], ax=axes[0,1])
axes[0,1].set_title("Tenure distribution")
sns.histplot(x=data['SupportCalls'], ax=axes[1,0])
axes[1,0].set_title("SupportCalls distribution")
sns.histplot(x=data['Income'], ax=axes[1,1])
axes[1,1].set_title("Income distribution")
plt.tight_layout()
plt.show()

# Analyze the distribution of categorical variables using bar plots

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribution of Categories', fontsize=16, fontweight='bold')
sns.countplot(x=data['Gender'], ax=axes[0, 0])
axes[0, 0].set_title("Gender Distribution")
sns.countplot(x=data['ProductType'], ax=axes[0, 1])
axes[0, 1].set_title("Product Type Distribution")
sns.countplot(x=data['ChurnStatus'], ax=axes[1, 0])
axes[1, 0].set_title("Churn Status Distribution")
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()


#  relationships between numerical features and the target variable using or box plots.

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribution by Churn Status', fontsize=16, fontweight='bold')
sns.boxplot(x=data['ChurnStatus'], y=data['Income'], ax=axes[0, 0])
axes[0, 0].set_title('Income Distribution by Churn Status')
sns.boxplot(x=data['ChurnStatus'], y=data['Age'], ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Churn Status')
sns.boxplot(x=data['ChurnStatus'], y=data['SupportCalls'], ax=axes[1, 0])
axes[1, 0].set_title('Support Calls Distribution by Churn Status')
sns.boxplot(x=data['ChurnStatus'], y=data['Tenure'], ax=axes[1, 1])
axes[1, 1].set_title('Tenure Distribution by Churn Status')
plt.tight_layout()
plt.show()

#Investigate relationships between categorical variables and the target using bar plots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x='Gender', hue='ChurnStatus', data=data, ax=axes[0])
axes[0].set_title('Churn Status by Gender')
sns.countplot(x='ProductType', hue='ChurnStatus', data=data, ax=axes[1])
axes[1].set_title('Churn Status by Product Type')
plt.tight_layout()
plt.show()

#  relationships between numerical features and the target variable using scatter plots (Triangular relationships). ( additional )

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Scatter Plots by Churn Status', fontsize=16, fontweight='bold')
sns.scatterplot(x=data['Tenure'], y=data['SupportCalls'], hue=data['ChurnStatus'], ax=axes[0, 0])
axes[0, 0].set_title('Tenure vs Support Calls by Churn Status')
sns.scatterplot(x=data['Income'], y=data['Tenure'], hue=data['ChurnStatus'], ax=axes[0, 1])
axes[0, 1].set_title('Income vs Tenure by Churn Status')
sns.scatterplot(x=data['Income'], y=data['SupportCalls'], hue=data['ChurnStatus'], ax=axes[1, 0])
axes[1, 0].set_title('Income vs Support Calls by Churn Status')
axes[1, 1].axis('off') 
plt.tight_layout()
plt.show()


# Correlation Analysis


correlation_matrix = data.select_dtypes(include=['number']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


#############################################################################################
###################################|  task 6  |##############################################
#############################################################################################

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sns.boxplot(x='ChurnStatus', y='Tenure', data=data, ax=axs[0, 0])
axs[0, 0].set_title('Tenure vs Churn Status (Box Plot)')
sns.boxenplot(x='ChurnStatus', y='Income', data=data, ax=axs[0, 1])
axs[0, 1].set_title('Income vs Churn Status (Boxen Plot)')
sns.swarmplot(x='ChurnStatus', y='Income', data=data, ax=axs[1, 0])
axs[1, 0].set_title('Income vs Churn Status (Swarm Plot)')
sns.histplot(x='Tenure', hue='ChurnStatus', data=data, ax=axs[1, 1], kde=True)
axs[1, 1].set_title('Tenure vs Churn Status (Hist Plot)')
plt.tight_layout()
plt.show()


sns.scatterplot(x=data['Income'], y=data['Tenure'], hue=data['ChurnStatus'])
plt.title('Income vs Tenure by Churn Status')
plt.show()
