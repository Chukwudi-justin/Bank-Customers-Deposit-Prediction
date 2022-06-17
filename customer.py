#%%
#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
import scipy
import missingno as mso
# %%
#More Libraries
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# %%
#importing Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
# %%
#Load Data
df = pd.read_csv('bank.csv')
# %%
df.head()
# %%
df.info
# %%
print('Rows: ', len(df))
print('Columns: ', df.shape[1])
# %%
#Count of missing values
df.isnull().sum()
# %%
#Data Types
df.dtypes
# %%
def Cat_dist(variable):
    var = df[variable]
    varValue = var.value_counts(dropna=False)
    print(varValue)
# %%
Cat_Var = ['education', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'deposit']
for v in Cat_Var:
    Cat_dist(v)
# %%
#Explaratory Data Analysis
labels = df['job'].value_counts(dropna=False).index
sizes = df['job'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Occupation',color = 'black',fontsize = 15)
# %%
labels = df['marital'].value_counts(dropna=False).index
sizes = df['marital'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Marital Status',color = 'black',fontsize = 15)
# %%
labels = df['education'].value_counts(dropna=False).index
sizes = df['education'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Educational Qualification',color = 'black',fontsize = 15)
# %%
labels = df['housing'].value_counts(dropna=False).index
sizes = df['housing'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Housing Loan',color = 'black',fontsize = 15)
# %%
def bar_plot(variable):
    var = df[variable]
    varValue = var.value_counts()
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n{}".format(variable,varValue))
# %%
sns.set_style('darkgrid')
categorical_variables = ['education', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'deposit']
for v in categorical_variables:
    bar_plot(v)
# %%
labels = df['deposit'].value_counts(dropna=False).index
sizes = df['deposit'].value_counts(dropna=False).values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of customers deposit',color = 'black',fontsize = 15)
# %%
sns.countplot(x="day", data=df, palette='inferno_r')
plt.show()
# %%
sns.countplot(x="month", data=df, palette="hls")
plt.show()
# %%
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with histogram".format(variable))
    plt.show()
# %%
numerical_variables = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for m in numerical_variables:
    plot_hist(m)
# %%
sns.countplot(x="contact", data=df, palette="hls")
plt.show()
# %%
#Further Analysis
#Relationship Between Variables
#Education vs Marriage
pd.crosstab(df.education,df.marital).plot(kind="bar", stacked=True, figsize=(5,5), color=['#f67f58','#12c2e8'])
plt.title('Education vs Marriage')
plt.xlabel('Education')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()
# %%
#Marital Status vs Deposit Status
pd.crosstab(df.marital,df.deposit).plot(kind="bar", stacked=True, figsize=(5,5), color=['#8A2BE2','#53868B'])
plt.title('Marital Status vs Deposit Status')
plt.xlabel('Marital Status')
plt.ylabel('Frequency')
plt.legend(["No Deposit", "Deposited"])
plt.xticks(rotation=0)
plt.show()
# %%
#Occupation vs Deposit Status
pd.crosstab(df.job,df.deposit).plot(kind="bar", stacked=True, figsize=(5,5), color=['#333333','#dd1818'])
plt.title('Occupation vs Deposit Status')
plt.xlabel('Occupation')
plt.ylabel('Frequency')
plt.legend(["No Deposit", "Deposited"])
plt.xticks(rotation=90)
plt.show()
# %%
#Education vs Deposit Status
pd.crosstab(df.education,df.deposit).plot(kind="bar", stacked=True, figsize=(5,5), color=['#654a7d','#ffd459'])
plt.title('Education vs Deposit Status')
plt.xlabel('Education')
plt.ylabel('Frequency')
plt.legend(["No Deposit", "Deposited"])
plt.xticks(rotation=0)
plt.show()
# %%
#Age vs Deposit Status
sns.boxplot(x="deposit", y="age", data=df, palette='twilight_r')
# %%
sns.swarmplot(x=df['deposit'],
              y=df['age'])
# %%
#Loan Vs Deposit
pd.crosstab(df.loan,df.deposit).plot(kind="bar", stacked=True, figsize=(5,5), color=['#333333','#dd1818'])
plt.title('Loan vs Deposit Status')
plt.xlabel('Loan Status')
plt.ylabel('Frequency')
plt.legend(["No Deposit", "Deposited"])
plt.xticks(rotation=0)
plt.show()
# %%
# EDucation VS Balance
sns.pointplot(y="balance", x="education", data=df)
# %%
# Jobs vs Balance
Occupation_type = ['admin.', 'technician', 'services', 'management', 'retired',
       'blue-collar', 'unemployed', 'entrepreneur', 'housemaid',
       'unknown', 'self-employed', 'student']

sns.boxenplot(x="job", y="balance",
              color="b", order=Occupation_type,
              scale="linear", data=df)
plt.xticks(rotation=90)
plt.show()
# %%
# Categorical Encoding
# One Hot Encoding
df = pd.get_dummies(df, columns=['default', 'housing', 'loan', 'deposit'])
# %%
# Drop categorical variable with pair strings columns
df = df.drop(['default_no', 'housing_no', 'loan_no', 'deposit_no'], axis = 1)
# %%
#LabelEncoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.marital = le.fit_transform(df.marital)
df.contact = le.fit_transform(df.contact)
#%%
df['month'].value_counts()
#%%
# Find and Replace for month
cleanup_month = {'month' : {'jan': 1, 'feb' : 2, 'mar' : 3, 'apr' : 4, 'may' : 5, 'jun' : 6, 'jul' : 7, 'aug' : 8, 'sep' : 9, 'oct' : 10, 'nov' : 11, 'dec' : 12}}
df = df.replace(cleanup_month)
#%%
#Categorical with more than pair string columns
df = pd.get_dummies(df, columns=['education', 'poutcome'])
#%%
from sklearn.preprocessing import OneHotEncoder
oe_style = OneHotEncoder(handle_unknown='ignore')
oe_result1 = oe_style.fit_transform(df[['job']])
J = pd.DataFrame(oe_result1.toarray(), columns=oe_style.categories_).head()
# %%
J.rename({'unknown':'job_unknown'}, inplace=True)
df = df.join(J)
# %%
df = df.drop('job', axis = 1)
# %%
#Target
sns.countplot(data=df,x='deposit_yes')
# %%
integer_columns=df.select_dtypes(include=['int64','float64']).columns.tolist()
for column in df:
    if df[column].isnull().any():
        if(column in integer_columns):
            df[column]=df[column].fillna(df[column].mode()[0])
#%%
#Model Preparation
#independent vs Target
X = df.drop(["deposit_yes"], axis=1)
y = df["deposit_yes"]
# %%
#Data Normalization
X = MinMaxScaler().fit_transform(X)
# %%
#Splitting Data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# %%
model_lgr = 'Logistic Regression'
lr = LogisticRegression(solver='saga', max_iter=500, random_state=1)
model = lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print("confussion matrix")
print(lr_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Logistic Regression: {:.2f}%".format(lr_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,lr_predict))
# %%
model_rfc = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=1000, random_state=12,max_depth=5)
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confussion matrix")
print(rf_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Random Forest:{:.2f}%".format(rf_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,rf_predicted))
# %%
model_egb = 'Extreme Gradient Boost'
xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
xgb.fit(X_train, y_train)
xgb_predicted = xgb.predict(X_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
xgb_acc_score = accuracy_score(y_test, xgb_predicted)
print("confussion matrix")
print(xgb_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Extreme Gradient Boost:{:.2f}%".format(xgb_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,xgb_predicted))
# %%
model_dtc = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)
print("confussion matrix")
print(dt_conf_matrix)
print("-------------------------------------------")
print("Accuracy of DecisionTreeClassifier:{:.2f}%".format(dt_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,dt_predicted))
#%%
from sklearn.ensemble import GradientBoostingClassifier
model_gbc = 'GradientBoostingClassifier'
gbc = GradientBoostingClassifier(random_state = 0)
gbc.fit(X_train, y_train)
gbc_predicted = gbc.predict(X_test)
gbc_conf_matrix = confusion_matrix(y_test, gbc_predicted)
gbc_acc_score = accuracy_score(y_test, gbc_predicted)
print("confussion matrix")
print(gbc_conf_matrix)
print("-------------------------------------------")
print("Accuracy of GradientBoostingClassifier:{:.2f}%".format(gbc_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,gbc_predicted))
#%%
import lightgbm
from lightgbm import LGBMClassifier
model_lgb = 'LightGradientBoosting'
lgb = LGBMClassifier(random_state = 0)
lgb.fit(X_train, y_train)
lgb_predicted = lgb.predict(X_test)
lgb_conf_matrix = confusion_matrix(y_test, lgb_predicted)
lgb_acc_score = accuracy_score(y_test, lgb_predicted)
print("confussion matrix")
print(lgb_conf_matrix)
print("-------------------------------------------")
print("Accuracy of LightGradientBoosting:{:.2f}%".format(lgb_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,lgb_predicted))
#%%
import catboost
from catboost import CatBoostClassifier
model_cbc = 'CatBoostClassifier'
cbc = CatBoostClassifier(verbose=0, random_state = 0)
cbc.fit(X_train, y_train)
cbc_predicted = cbc.predict(X_test)
cbc_conf_matrix = confusion_matrix(y_test, cbc_predicted)
cbc_acc_score = accuracy_score(y_test, cbc_predicted)
print("confussion matrix")
print(cbc_conf_matrix)
print("-------------------------------------------")
print("Accuracy of CatBoostClassifier:{:.2f}%".format(cbc_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,cbc_predicted))
#%%
from sklearn.ensemble import AdaBoostClassifier
model_ada = 'AdaBoostClassifier'
ada = AdaBoostClassifier(random_state = 0)
ada.fit(X_train, y_train)
ada_predicted = ada.predict(X_test)
ada_conf_matrix = confusion_matrix(y_test, ada_predicted)
ada_acc_score = accuracy_score(y_test, ada_predicted)
print("confussion matrix")
print(ada_conf_matrix)
print("-------------------------------------------")
print("Accuracy of AdaBoostClassifier:{:.2f}%".format(ada_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,ada_predicted))
#%%
from sklearn.neural_network import MLPClassifier
model_mlp = 'Multi-LayerPerception'
mlp = MLPClassifier(hidden_layer_sizes=(64,64), alpha=1, random_state=0)
mlp.fit(X_train, y_train)
mlp_predicted = mlp.predict(X_test)
mlp_conf_matrix = confusion_matrix(y_test, mlp_predicted)
mlp_acc_score = accuracy_score(y_test, mlp_predicted)
print("confussion matrix")
print(mlp_conf_matrix)
print("-------------------------------------------")
print("Accuracy of Multi-LayerPerception:{:.2f}%".format(mlp_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,mlp_predicted))
# %%
model_ev = pd.DataFrame({'Model': ['Logistic Regression','Random Forest','Extreme Gradient Boost','Decision Tree', 'GradientBoostingClassifier', 'LightGradientBoosting', 'CatBoostClassifier', 'AdaBoostClassifier', 'Multi-LayerPerception'], 'Accuracy': [round((lr_acc_score*100), 2),
                    round((rf_acc_score*100), 2),round((xgb_acc_score*100), 2),round((dt_acc_score*100), 2), round((gbc_acc_score*100), 2), round((lgb_acc_score*100), 2), round((cbc_acc_score*100), 2), round((ada_acc_score*100), 2), round((mlp_acc_score*100), 2)]})
# %%
model_ev.sort_values(by='Accuracy', ascending=False)
# %%
model_cbc = 'CatBoostClassifier'
cbc = CatBoostClassifier(verbose=0, random_state = 0, depth = 6, l2_leaf_reg = 7, learning_rate = 0.03)
cbc.fit(X_train, y_train)
cbc_predicted = cbc.predict(X_test)
cbc_conf_matrix = confusion_matrix(y_test, cbc_predicted)
cbc_acc_score = accuracy_score(y_test, cbc_predicted)
print("confussion matrix")
print(cbc_conf_matrix)
print("-------------------------------------------")
print("Accuracy of CatBoostClassifier:{:.2f}%".format(cbc_acc_score*100,'\n'))
print("-------------------------------------------")
print(classification_report(y_test,cbc_predicted))
# %%
