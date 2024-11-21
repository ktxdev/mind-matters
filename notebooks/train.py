import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils import  standardize_numerical_features, encode_categorical_features

data = pd.read_csv("/Users/ktxdev/Developer/mind-matters/data/cleaned/train.csv")

# Standardize numerical features
# data = standardize_numerical_features(data)
# # Encode categorical variables
# data = encode_categorical_features(data)

X = data.drop(columns=['Depression'])
y = data['Depression']

# Step 2: Split Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the model
xgb_model = XGBClassifier(eval_metric='logloss')

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=3)
grid_search.fit(X, y)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

from sklearn.externals import joblib  # Note: joblib is included in sklearn for older versions

joblib.dump(grid_search.best_estimator_, 'model.pkl')


# Predict and Evaluate
# xgb_pred = grid_search.best_estimator_.predict(X_test)
# print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
# print("\nClassification Report:\n", classification_report(y_test, xgb_pred))

test_data = pd.read_csv("https://raw.githubusercontent.com/ktxdev/mind-matters/refs/heads/master/data/cleaned/test.csv")

X_test_data = test_data.drop(columns=['id', 'Name'])

X_test_data['Academic Pressure'] = X_test_data['Academic Pressure'].astype('object')
X_test_data['Work Pressure'] = X_test_data['Work Pressure'].astype('object')
X_test_data['Study Satisfaction'] = X_test_data['Study Satisfaction'].astype('object')
X_test_data['Job Satisfaction'] = X_test_data['Job Satisfaction'].astype('object')
X_test_data['Work/Study Hours'] = X_test_data['Work/Study Hours'].astype('object')
X_test_data['Financial Stress'] = X_test_data['Financial Stress'].astype('object')

from sklearn.preprocessing import LabelEncoder
# Initialize LabelEncoder
label_encoder = LabelEncoder()

columns_to_encode = ['Academic Pressure','Work Pressure', 'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']

# Apply LabelEncoder to each column
for col in columns_to_encode:
    X_test_data[col] = label_encoder.fit_transform(X_test_data[col])

X_test_data = pd.get_dummies(X_test_data, columns=['Gender', 'City', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree',
                                     'Working Professional or Student', 'Have you ever had suicidal thoughts ?',
                                     'Family History of Mental Illness'], drop_first=True)

X_test_data['Financial_Work_Interaction'] = X_test_data['Financial Stress'] * X_test_data['Work/Study Hours']
X_test_data['Academic_Financial_Interaction'] = X_test_data['Academic Pressure'] * X_test_data['Financial Stress']
X_test_data['Family_Suicidal_Interaction'] = X_test_data['Have you ever had suicidal thoughts ?_Yes'] * X_test_data['Family History of Mental Illness_Yes']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_test_data[['Age','CGPA']] = scaler.fit_transform(X_test_data[['Age','CGPA']])


y_pred = grid_search.best_estimator_.predict(X_test_data)

x = pd.concat([test_data['id'], pd.Series(y_pred, name='Depression')], axis=1)
x.to_csv('submission.csv', index=False)