from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Custom transformer to add Isolation Forest anomaly scores as a new feature
class IsolationForestScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination)

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X):
        # Calculate the anomaly scores
        scores = self.model.score_samples(X)
        # Add the scores as a new column to X
        return np.column_stack((X, scores))

# Paths to the data files
path = r"C:\Users\BHARAT\Downloads\credit-card-fraud-prediction\train.csv"
path1 = r"C:\Users\BHARAT\Downloads\credit-card-fraud-prediction\test.csv"

# Load data
data = pd.read_csv(path)
test = pd.read_csv(path1)

# Preprocessing
data.columns = data.columns.str.lower()
test.columns = test.columns.str.lower()
data.drop(['id'], axis=1, inplace=True)

# Split data into features and target
X = data.drop('isfraud', axis=1)
y = data['isfraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define preprocessing steps and models
sampling = RandomUnderSampler(random_state=42)
scaler = RobustScaler()
rf1 = RandomForestClassifier(n_jobs=-1, random_state=42, max_depth=8, max_samples=0.8, max_features=3, min_samples_leaf=2)
rf2 = RandomForestClassifier(n_jobs=-1, random_state=42, max_depth=7, max_samples=0.8, max_features=3, min_samples_leaf=2)
rf3 = RandomForestClassifier(n_jobs=-1, random_state=42, max_depth=12, max_samples=0.8, max_features=3, min_samples_leaf=2)
base = [('m1', rf1), ('m2', rf2), ('m3', rf3)]

# Define the ensemble model
model = VotingClassifier(base, n_jobs=-1, voting='soft')

# Create the pipeline
pipe = Pipeline([
    ('sampling', sampling),
    ('scaler', scaler),
    ('isolation', IsolationForestScoreTransformer(contamination=0.15)),
    ('model', model)
])

# Train the pipeline
pipe.fit(X_train, y_train)

# Predictions and evaluation
prediction = pipe.predict(X_test)
probability = pipe.predict_proba(X_test)
print('Accuracy', accuracy_score(prediction, y_test))
print('Classification Report', classification_report(prediction, y_test))
print('ROC Score', roc_auc_score(y_test, probability[:, 1])*100)

# Prediction on the test set
id = test['id']
test.drop(['id'], axis=1, inplace=True)
prob = pipe.predict_proba(test)

# Create a DataFrame for probabilities
prob_df = pd.DataFrame(prob, columns=pipe.named_steps['model'].classes_)
test = pd.concat([id, prob_df], axis=1)
test = test[['id', 1]]
test = test.round(1)

# Save predictions to a CSV file
test.to_csv('C:/Users/BHARAT/Desktop/credit_prediction.csv', index=False)

# Cross-validation for model evaluation
cv = cross_val_score(pipe, X_train, y_train, cv=10, scoring='roc_auc', n_jobs=-1)
print(cv, '\n', cv.mean()*100)
