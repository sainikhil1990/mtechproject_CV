import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('motion_result_training.csv')
print(df.head())

#Insights on Data
#Number of Features
print("Number of Features: ", len(df.iloc[:, :-1].columns))
print("Possible values of Output: " , df.iloc[:, -1].unique())

print("Original Data set size: ", df.size)
df_new = df.drop_duplicates() #Removal of duplicate values
print("Dataset size after removal of duplicates: ",df_new.size)
print("Any NULL values in Dataset? : ", df_new.isnull().values.any()) #Removal of null values
#No Categorical Features
X = df_new.iloc[:, :-1] #Features
y = df_new.iloc[:, -1] #Target Labels

#Spiliting of data done with 80% training data and 20% test data
X = df.iloc[:, :-1] #Features
y = df.iloc[:, -1] #Target Output
X_train, X_test, y_train, y_test = train_test_split(
 X,y , random_state=0,test_size=0.20, shuffle=True)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test data
df1 = pd.read_csv('validate_data.csv')
print(df1.head())
X1 = df1.iloc[:, :-1] #Features
y1 = df1.iloc[:, -1] #Target Labels

y_pred = rf_classifier.predict(X1)

# Calculate the accuracy
accuracy = accuracy_score(y1, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print(classification_report(y1, y_pred))


