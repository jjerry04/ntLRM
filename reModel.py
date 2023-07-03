
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

####Import dataset#####
dir = "Data/Melbourne_housing_FULL.csv"
df = pd.read_csv(dir)

####Preprocess#####
#Remove variables
subsets = ['Address', 'Method', 'SellerG', 'Date', 'Postcode',
           'YearBuilt', 'Type', 'Lattitude', 'Longtitude', 'Regionname',
           'Suburb', 'CouncilArea']

for property in subsets:
    del df[property]

df.head()

#Display all missing NaN value/feature
df.isnull().sum()

#visualize heatmap excluding nulls
df_heat = df.corr()
sns.heatmap(df_heat, annot=True, cmap='coolwarm')
plt.show()
df.shape

##Filtering data##
del df['Bedroom2'] #High missing data
del df['Landsize'] # Not correlated with dependent var
del df['Propertycount'] #Not correlated with dependent var

del df['BuildingArea']

#Fill missing values with the mean
df['Car'].fillna(df['Car'].mean(), inplace = True)

#Drop remaining missing values on row-by-row basis
df.dropna(axis=0, how='any', subset = None, inplace = True)

df.shape

#After cleaning and filtering data - prep for LR model
##################################
#Set ml variables (x, y)
x = df[['Rooms', 'Distance', 'Bathroom', 'Car']] #Independent variable
y = df['Price'] #Dependant variable (what we want to predict or know)

#Using 70/30 rule shuffle and divide data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                           test_size=0.3, 
                                           random_state=10, 
                                           shuffle=True)
#Assign ml algorithm
model = LinearRegression()

#Link training data to algorithm stored in ml model
model.fit(x_train, y_train)

#Find y-intercept for the model and coefficients for each independent variable
#Find y
model.intercept_

#Find x coefficients
model.coef_

#Setup 2 col table for easy reference to X coefficients
model_results = pd.DataFrame(model.coef_, x.columns,
                             columns=['Coefficients'])
model_results

#############Predict##########################
#Predict value of a new property
new_house = [2, #Rooms
             2.5, #Distance
             1, #Bathroom
             1, #Car
             ]

new_house = [4, #Rooms
             5, #Distance
             4, #Bathroom
             2, #Car
             ]

new_house_predict = model.predict([new_house])
new_house_predict[0]

###############Evaluation###################
#Mean absolute value error (predicted price vs actual price)
prediction = model.predict(x_test)
mave = metrics.mean_absolute_error(y_test, prediction)
predicted_cost = new_house_predict[0] -  mave
new_house_predict[0]
mave
predicted_cost

#Write Readme about model, dataset, error, evaluation













