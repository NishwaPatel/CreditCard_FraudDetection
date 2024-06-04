import pandas as pd
cdf = pd.read_csv("creditcard.csv")
print(cdf)

print(cdf.info())
print(cdf.describe())

print(cdf.isnull().sum())

# preprocessing

from sklearn.preprocessing import StandardScaler

cdf['scaled_amount'] = StandardScaler().fit_transform(cdf['Amount'].values.reshape(-1, 1))
cdf = cdf.drop(['Amount'], axis=1)

from sklearn.model_selection import train_test_split

X = cdf.drop('Class', axis=1)
y = cdf['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_test)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_test,y_pred)

new_data = {'Time': 0,'V1': -1.3598071336738, 'V2': -0.0727811733098497, 'V3': 2.53634673796914, 'V4': 1.37815522427443,
            'V5': -0.338320769942518, 'V6': 0.462387777762292, 'V7': 0.239598554061257, 'V8': 0.0986979012610507,
            'V9': 0.363786969611213, 'V10': 0.0907941719789316, 'V11': -0.551599533260813, 'V12': -0.617800855762348,
            'V13': -0.991389847235408, 'V14': -0.311169353699879, 'V15': 1.46817697209427, 'V16': -0.470400525259478,
            'V17': 0.207971241929242, 'V18': 0.0257905801985591, 'V19': 0.403992960255733, 'V20': 0.251412098239705,
            'V21': -0.018306777944153, 'V22': 0.277837575558899, 'V23': -0.110473910188767, 'V24': 0.0669280749146731,
            'V25': 0.128539358273528, 'V26': -0.189114843888824, 'V27': 0.133558376740387, 'V28': -0.0210530534538215,
            'scaled_amount': 119.62}  # This amount should be preprocessed similarly

new_data_df = pd.DataFrame([new_data])


# Predict
prediction = model.predict(new_data_df)
print(prediction)

print(f"Prediction: {'Fraudulent' if prediction[0] == 1 else 'Non-Fraudulent'}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

