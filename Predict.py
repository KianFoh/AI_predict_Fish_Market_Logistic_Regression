import joblib
import pandas as pd
Fish_Market = joblib.load('model_1.1.2.pkl')

data = {'Weight':  [390.0],
        'Length1': [27.6],
        'Length2': [30.0],
        'Length3': [35.0],
        'Height': [12.6700],
        'Width': [4.6900]}

d = {0: 'Bream', 1: 'Parkki', 2: 'Perch', 3: 'Pike', 4: 'Roach', 5: 'Smelt', 6: 'Whitefish'}
prediction = Fish_Market.predict(pd.DataFrame(data))[0]
print (d[prediction])