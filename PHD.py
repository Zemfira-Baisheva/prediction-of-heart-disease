import pandas as pd
import numpy as np
import streamlit as st
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
from joblib import load


st.write("""# Предсказание сердечных заболеваний""")

st.write("""### Как правильно вводить данные
         
* __Возраст__: возраст пациента [количество полных лет]
* __Пол__: пол пациента [M: мужчина, F: женщина]
* __Тип боли в груди__: тип боли в груди [ТА: типичная стенокардия, ATA: атипичная стенокардия, NAP: нестенокардитическая боль, ASY: бессимптомная]
* __АД в покое__: артериальное давление в покое [мм рт. ст.]
* __Холестерин__: уровень холестерина в сыворотке [мг/дл]
* __FastingBS__: уровень сахара в крови натощак [1: если уровень сахара в крови натощак > 120 мг/дл, 0: в противном случае]
* __RestingECG__: результаты электрокардиограммы в состоянии покоя [Normal: в норме, 
        ST: наличие аномалии зубцов ST-T (инверсия зубца T и/или подъем или депрессия зубца ST > 0,05 мВ),
        LVH: наличие вероятной или определенной * гипертрофии левого желудочка по критериям Эстеса]
* __MaxHR__: максимальная частота сердечных сокращений [числовое значение от 60 до 202]
* __Стенокардия при физической нагрузке__: стенокардия, вызванная физической нагрузкой [Y: да, N: нет]
* __Oldpeak__: oldpeak = ST [числовое значение, измеренное в депрессии]
* __ST_Slope__: наклон сегмента ST на пике физической нагрузки [Up: восходящий, Flat: горизонтальный, Down: нисходящий]
""")



data_user = {
    "Age": st.text_input('Введите возраст'),
    "Sex": st.selectbox('Выберите пол', ['M', 'F']),
    "ChestPainType": st.selectbox('Выберите тип боли в груди', ['TA', 'ATA', 'NAP', 'ASY']),
    "RestingBP": st.text_input('Введите АД в покое'),
    "Cholesterol": st.text_input('Введите уровень холестерина'),
    "FastingBS": st.selectbox('Уровень сахара в крови натощак', [0, 1]),
    "RestingECG": st.selectbox('Выберите результаты ЭКГ', ['Normal', 'ST', 'LVH']),
    "MaxHR": st.text_input('Введите максимальную ЧСС'),
    "ExerciseAngina": st.selectbox('Наличие стенокардии при физической нагрузке', ['Y', 'N']),
    "Oldpeak": st.text_input('Введите значение oldpeak'),
    "ST_Slope": st.selectbox('Выберите наклон сегмента ST', ['Up', 'Flat', 'Down'])
}

df_predict = pd.DataFrame(data_user, index=[0])

numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
df_predict[numeric_columns] = df_predict[numeric_columns].apply(pd.to_numeric)

ml_pipe_cb = load('/home/zemfira/Фаза 1/ds-phase-1/06-supervised/prediction-of-heart-disease/ml_pipeline.pkl')

predictions = ml_pipe_cb.predict_proba(df_predict)

proba = int(predictions[0][0]*100)

st.write("### Predictions:")
st.write(f'##### С вероятностью в {proba}% у Вас не имеется середечных заболеваний')