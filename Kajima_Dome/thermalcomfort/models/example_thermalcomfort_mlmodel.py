import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor
XGBRegr2 = XGBRegressor()
XGBRegr2.load_model(
        '/home/lakshmi/Desktop/demo/FY22/JULY/model/XGBRegr_26k_9_race40.json')

'''
Encoding

Age
no encoding

Gender
{'Female': 0, 'Male': 1}

Race
{'Asian': 0, 'Caucasian': 1, 'Indian': 2}

'''


### all personal details as Nan

modelip1 = pd.DataFrame([[1.11, 1.0, 0.01,55.2,24.2,24.2,
                         np.nan, np.nan, np.nan]], 
                        
                       columns=['gtClo', 'gtMet', 'gtAS', 'gtRelHum','gtTA', 'gtRT',
                                'Age', 'Gender','Race'])
tc1 = XGBRegr2.predict(modelip1)
print(tc1)

# few personal details as Nan

modelip2 = pd.DataFrame([[1.11, 1.0, 0.01,55.2,24.2,24.2,
                         40, 1, np.nan]], 
                        
                       columns=['gtClo', 'gtMet', 'gtAS', 'gtRelHum','gtTA', 'gtRT',
                                'Age', 'Gender','Race'])
tc2 = XGBRegr2.predict(modelip2)
print(tc2)
# no personal details as Nan

modelip2 = pd.DataFrame([[1.11, 1.0, 0.01,55.2,24.2,24.2,
                         40, 1, 1]], 
                        
                       columns=['gtClo', 'gtMet', 'gtAS', 'gtRelHum','gtTA', 'gtRT',
                                'Age', 'Gender','Race'])
tc2 = XGBRegr2.predict(modelip2)
print(tc2)