# Bank-Customers-Deposit-Prediction
### INTRODUCTION

The Bank's capital source is highly dependent on customers' deposits. The deposits have a very important meaning to the Bank because it is the basis for the bank to bring profits through investments, loans, reserves e.t.c. Therefore, it is important for commercial banks to ensure maximum customer satisfaction with the services that the bank provides.\
In Nigeria for example the apex bank called CENTRAL BANK OF NIGERIA (CBN) recently instructed that the minimum paid-up share capital to be maintained for National level banking license is N25 Billion Naira, or any such amount that may be prescribed by the CBN, while for Regional Banking License is N10 Billion Naira and International Commercial Banking License is N50 Billion.This is to ensure the required level of capital adequacy, liquidity, and cash reserve by banks the same step was taking by the apex bank in Ghana and this has led to improved marketing campaign by banks in order to increase customers deposit.\

Sourse of the data https://archive.ics.uci.edu/ml/datasets/bank+marketing

# DATASET DESCRIPTION
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# Output variable (desired target):
16 - y - has the client subscribed a term deposit? (binary: 'yes','no')

# This project intends to achieve the following:
1 Information about a marketing campaign of a financial institution in order to find ways to look for future strategies in order to improve future marketing campaigns for the bank.
2 To generate Machine Learning and Deep Learning models that have a high degree of accuracy when predicting if a customer will deposit or not, based on certain predictor variables.

## MODELS EXPLORED
1 Logistic Regression\
2 Random Forest\
3 Extreme Gradient Boost\
4 Decision Tree\
5 GradientBoostingClassifier\
6 LightGradientBoosting\
7 CatBoostClassifier\
8 AdaBoostClassifier\
9 Multi-LayerPerception

### EXPLORATORY DATA ANALYSIS OF DATA SET
## Occupation 
![job](distoccup.png).

![job](JobsVSBalance.png).

![job](occupationVSdeposit.png).

As we can see people with Management jobs are more likly to make deposit.\
We can also notice that most active balance and most deposit are mostly from managemnet and technician related jobs hence the need for the marketing department of the bank to increase marketing campaigns for people doing this jobs, one can also notice high balance of ritirees.

## Marital Status 
![marital](marrybar.png).

![marital](marryVSdeposit.png).

It can be observed that married people where more likely to make a deposit but same can not be said of the divorced, It can also be observed that single people will most likely make a deposit.

More soon...
