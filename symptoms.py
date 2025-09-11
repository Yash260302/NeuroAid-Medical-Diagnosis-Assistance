import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
import joblib


df = pd.read_csv('dataset_sorted.csv')
print(df.head())
print('Shape:',df.shape)
print('Columns:',df.columns)
print(df.info())

print(df.isnull().sum())
print('Unique Diseases:', df['diseases'].unique())
print(df['diseases'].value_counts().head(10))  #top 10 diseases

df = df.drop_duplicates()
df = df.fillna('No Symptom')
print(df)

# model training
df = df.groupby('diseases').filter(lambda x: len(x) > 1)

x = df.drop(columns=['diseases'])
y =  df['diseases']

# split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2, stratify=y,random_state=42
)

model = RandomForestClassifier(
    n_estimators=250,       # more trees → better accuracy
    max_depth=25,           # deeper splits allowed
    max_features="sqrt",    # consider subset of features per split
    min_samples_split=3,    # finer splits
    class_weight="balanced",# handle imbalance
    n_jobs=-1,              # use all CPU cores
    random_state=42  

)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
report = classification_report(y_test,y_pred)
print(report)

joblib.dump(model,'rf_disease_model.pkl')
joblib.dump(list(x.columns),'symptom_list.pkl')
joblib.dump(list(model.classes_), "class_names.pkl")

