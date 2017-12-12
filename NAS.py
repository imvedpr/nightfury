import pandas as pd

# import excel files 
df1 = pd.read_csv("nas-columns.csv")
df2 = pd.read_csv("nas-labels.csv")
df4 = pd.read_csv("nas-pupil-marks.csv")
df5 = pd.get_dummies( df4['State'] )

# deleting string type columns
del df4['State']
del df4['Use computer']
del df4['Subjects']

# concating original and dummy variables datasets
df3 = pd.concat([df5, df4],axis=1)

# make two datasets: 1 for missing values (mathswise) and other not missing_values
notmaths = df3['Maths %'].notnull()
df_notmaths = df3[notmaths]
nullmaths = df3['Maths %'].isnull()
df_nullmaths = df3[nullmaths]

# make two datasets: 1 for missing values (readingwise) and other not missing_values
notread = df3['Reading %'].notnull()
df_notread = df3[notread]
nullread = df3['Reading %'].isnull()
df_nullread = df3[nullread]

# make two datasets: 1 for missing values (sciencewise) and other not missing_values
notsci = df3['Science %'].notnull()
df_notsci = df3[notsci]
nullsci = df3['Science %'].isnull()
df_nullsci = df3[nullsci]

# make two datasets: 1 for missing values (socialwise) and other not missing_values
notsocial = df3['Social %'].notnull()
df_notsocial = df3[notsocial]
nullsocial = df3['Social %'].isnull()
df_nullsocial = df3[nullsocial]

# Split the dataset into train and test sets
X_train_maths = df_notmaths.ix[:,0:90]
y_train_maths = np.asarray(df_notmaths['Maths %'], dtype="|S6")
X_test_maths = df_nullmaths.ix[:,0:90]

# Import and fit Random Forest model
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier()
# fit the model
model.fit(X_train_maths, y_train_maths)
# predict the values in test set
rf_predict = model.predict(X_test_maths)
df_nullmaths['Maths %'] = rf_predict


# feature_importancefrom sklearn import tree
import matplotlib.pyplot as plt
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = df_notmaths.columns[indices]
f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 10)
plt.bar(range(X_train_maths.shape[1]), importances[indices],color="b", align="center")
plt.xticks(range(X_train_maths.shape[1]), feature_names)
plt.xlim([-1, X_train_maths.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 0.02)
plt.xticks(rotation=90)












