import pandas as pd
import numpy as np
from tqdm import tqdm

import sklearn
sklearn.set_config(transform_output="pandas")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from catboost import CatBoostClassifier


from sklearn.metrics import accuracy_score
import joblib

#import optuna

df = pd.read_csv('/home/zemfira/Фаза 1/ds-phase-1/06-supervised/aux/heart.csv')
num_features = df.select_dtypes(exclude='object')
cat_features = df.select_dtypes(include='object')

X, y = df.drop('HeartDisease', axis=1), df['HeartDisease']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, stratify=y, random_state=12)

pipe_ml = Pipeline(
    [
        ('imputer', ColumnTransformer(transformers = 
        [
        ('num_imputer', SimpleImputer(strategy='mean'), ['Age']),
        ],
        verbose_feature_names_out = False,
        remainder = 'passthrough' 
        )),
        
        ('preprocessor', ColumnTransformer([
        ('ordinal_encoding', OrdinalEncoder(), ['Sex', 'ExerciseAngina']),
        ('target_encoding', TargetEncoder(), ['ChestPainType', 'RestingECG', 'ST_Slope']),
        ('scaling_num', StandardScaler(), ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'])
        ]
        )),
        ("cat_boost", CatBoostClassifier(eval_metric='Accuracy', iterations=100))
    ]
)

pipe_ml.fit(X_train, y_train)

y_pred = pipe_ml.predict(X_valid)


joblib.dump(pipe_ml, '/home/zemfira/Фаза 1/ds-phase-1/06-supervised/prediction-of-heart-disease/ml_pipeline.pkl')
