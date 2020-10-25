import streamlit as st
import pandas as pd
import random
import pickle
import numpy as np

model = pickle.load(open('model/model.p', 'rb'))
mean = pickle.load(open('model/mean.p', 'rb'))
min_ = pickle.load(open('model/min.p', 'rb'))
max_ = pickle.load(open('model/max.p', 'rb'))

sidebars = {k: st.sidebar.slider(k, min_value=min_[k], max_value=max_[k], value=v) for k,v in mean.items()}

st.write(f"""
# Risk of Diabetes Checker
""")

st.write("""
#### Pregnancies
Number of times pregnant

#### Glucose
Plasma glucose concentration a 2 hours in an oral glucose tolerance test

#### BloodPressure
Diastolic blood pressure (mm Hg)

#### SkinThickness
Triceps skin fold thickness (mm)

#### Insulin
2-Hour serum insulin (mu U/ml)

#### BMI
Body mass index (weight in kg/(height in m)^2)

#### Age
Age (years)
""")

pred = model.predict([[sidebars['Pregnancies'],sidebars['Glucose'],sidebars['BloodPressure'],sidebars['SkinThickness'],sidebars['Insulin'],sidebars['BMI'],sidebars['Age']]])
result = "Diabetic" if pred >= .5 else "Not diabetic"


st.write(f"""
## Results:
### {result}
""")