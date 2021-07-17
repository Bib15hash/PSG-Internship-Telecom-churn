
######################################################################################################


#        LOGISTIC REGRESSION



#   IMPORTING THE REQUIRED LIBRARIES

import numpy as np
import pandas as pd

#   READING THE DATASET (TELECOM_CHURN)

df = pd.read_csv('Churn.csv')

df.describe().info()

# LABEL ENCODING VARIABLES - GEOGRAPHY AND GENDER

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Geography_2'] = le.fit_transform(df['Geography'])
df['Gender_2'] = le.fit_transform(df['Gender'])

#   DATA CLEANING

df = df[(df['EstimatedSalary'] >= 30000)]
df = df[(df['Balance'] != 0)]

#   INDEPENDENT AND DEPENDENT VARIABLES

X = df[['CreditScore', 'Age', 'Tenure','Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
       'Geography_2','Gender_2']].values
        
Y = df[['Exited']].values

#   SPLITTING THE DATA INTO TRAINING AND TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#   LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, Y_train)
y_pred = lg.predict(X_test)


#   ACCURACY, PRECISION, RECALL AND F-MEASURE FOR LOGISTIC REGRESSION
 
from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(Y_test, y_pred))


########################################################################################################




#       LINEAR REGRESSION


#   IMPORTING THE REQUIRED LIBRARIES
import numpy as np
import pandas as pd

#   READING THE DATASET (TELECOM_CHURN)

df = pd.read_csv('Churn.csv')

df.describe().info()

# LABEL ENCODING VARIABLES - GEOGRAPHY AND GENDER

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Geography_2'] = le.fit_transform(df['Geography'])
df['Gender_2'] = le.fit_transform(df['Gender'])

#   DATA CLEANING

df = df[(df['Balance'] != 0)]

#   INDEPENDENT AND DEPENDENT VARIABLES

X = df[['CreditScore', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember',
       'Geography_2','Gender_2']].values
        
Y = df[['EstimatedSalary']].values

#   SPLITTING THE DATA INTO TRAINING AND TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

#   LINEAR REGRESSION (ON ESTIMATED SALARY)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

y_pred = lr.predict(X_test)    

from sklearn.metrics import r2_score

print("Mean absolute error           : %.2f" % np.mean(np.absolute(y_pred - Y_test)))
print("Residual sum of squares (MSE) : %.2f" % np.mean((y_pred - Y_test) ** 2))



########################################################################################################################



#       DECISION TREE ALGORITHM



#   IMPORTING THE REQUIRED LIBRARIES    
  
import numpy as np
import pandas as pd

#   READING THE DATASET (TELECOM_CHURN)

df = pd.read_csv('Churn.csv')

df.describe().info()

# LABEL ENCODING VARIABLES - GEOGRAPHY AND GENDER

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Geography_2'] = le.fit_transform(df['Geography'])
df['Gender_2'] = le.fit_transform(df['Gender'])

#   DATA CLEANING

df = df[(df['EstimatedSalary'] >= 30000)]
df = df[(df['Balance'] != 0)]

#   INDEPENDENT AND DEPENDENT VARIABLES

X = df[['CreditScore', 'Age', 'Tenure','Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
       'Geography_2','Gender_2']].values
        
Y = df[['Exited']].values


#   SPLITTING THE DATA INTO TRAINING AND TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#   FITTING TO DECISION TREE ALGORITHM TO THE DATASET

from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(random_state=0)
dc.fit(X_train,Y_train)

y_pred = dc.predict(X_test)


#   ACCURACY, PRECISION, RECALL AND F-MEASURE FOR DECISION TREE ALGORITHM
 
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))


######################################################################################################



#      RANDOM FOREST ALGORITHM


#   IMPORTING THE DATASET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Churn.csv')

df.describe().info()

# LABEL ENCODING VARIABLES - GEOGRAPHY AND GENDER

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Geography_2'] = le.fit_transform(df['Geography'])
df['Gender_2'] = le.fit_transform(df['Gender'])

#   DATA CLEANING

df = df[(df['EstimatedSalary'] >= 30000)]
df = df[(df['Balance'] != 0)]

#   INDEPENDENT AND DEPENDENT VARIABLES

X = df[['CreditScore', 'Age', 'Tenure','Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
       'Geography_2','Gender_2']].values
        
Y = df[['Exited']].values


#   SPLITTING THE DATA INTO TRAINING AND TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#   FITTING TO RANDOM FOREST ALGORITHM TO THE DATASET

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train,Y_train)

y_pred = rfc.predict(X_test)


#   ACCURACY, PRECISION, RECALL AND F-MEASURE FOR RANDOM FOREST ALGORITHM
 
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))    


##################################################################################################



#      ARTIFICIAL NEURAL NETWORK (ANN)


#   IMPORTING THE REQUIRED LIBRARIES  

import numpy as np
import pandas as pd

df = pd.read_csv('Churn.csv')

X = df.iloc[:, 3:13].values
Y = df.iloc[:, 13].values

# LABEL ENCODING AND ONE HOT ENCODING VARIABLES - GEOGRAPHY AND GENDER

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_1 = LabelEncoder()
le_2 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])
X[:, 2] = le_2.fit_transform(X[:, 2]) 
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]


#   SPLITTING THE DATA INTO TRAINING AND TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#   APPLYING FEATURE SCALING

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#     IMPORTING THE KERAS LIBRARIES AND PACKAGES

import keras
from keras.models import Sequential
from keras.layers import Dense

#     INITIALISING THE ANN

classifier = Sequential()

#   ADDING THE INPUT LAYER AND THE FIRST HIDDEN LAYER
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim = 11))

#   ADDING THE SECOND HIDDEN LAYER
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))

#   ADDING THE OUTPUT LAYER
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))

#   COMPILING THE ARTIFICIAL NEURAL NETWORK

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#   FITTING THE ARTIFICIAL NEURAL NETWORK TO THE TRAINING SET

classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

#   PREDICTING THE TEST SET RESULTS

y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5


#   ACCURACY, PRECISION, RECALL AND F-MEASURE FOR ANN
 
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))


