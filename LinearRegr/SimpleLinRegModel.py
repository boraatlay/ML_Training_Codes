# Import elementary modules
import pandas as pd
import numpy as np

# Import data visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

#Import ML tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Reading in the data file and a first look
df = pd.read_csv('Ecommerce Customers')
df.info()
df.head(5)
df.describe()

# A slightly deeper looks to have an insight of the data
jp_webtime_annualspent = sns.jointplot(df['Time on Website'],df['Yearly Amount Spent'])
jp_apptime_annualspent = sns.jointplot(df['Time on App'],df['Yearly Amount Spent'])
jp_apptime_mship       = sns.jointplot(df['Time on App'],df['Length of Membership'],kind='hex')
lmp_annualspent_mship  = sns.lmplot('Yearly Amount Spent','Length of Membership',df)


jp_webtime_annualspent.savefig("jp_webtime_annualspent.pdf")
jp_apptime_annualspent.savefig("jp_apptime_annualspent.pdf")
jp_apptime_mship.savefig("jp_apptime_mship.pdf")
lmp_annualspent_mship.savefig("lmp_annualspent_mship.pdf")

# Splitting data into test and train samples
# df.columns
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
X_test, X_train, y_test, y_train = train_test_split(X,y,test_size = 0.3, random_state=101)

# Initiating and training the linear model 
lr = LinearRegression()
lr.fit(X_train,y_train)

print(lr.coef_)

# Computing the predictions
predictions = lr.predict(X_test)
plt.scatter(y_test,predictions)

# Checking the metrics on predictions
mae = metrics.mean_absolute_error(y_test,predictions)
mse = metrics.mean_squared_error(y_test,predictions)
sre = np.sqrt(metrics.mean_squared_error(y_test,predictions))

print ('\n', mae, '\n', mse, '\n', sre)

sns.distplot(y_test - predictions)
pd.DataFrame(lr.coef_,X_test.columns)
