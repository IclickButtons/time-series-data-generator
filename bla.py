from data_generator import DataGeneratorTimeSeries 
import numpy as np 
import pandas as pd 

df = pd.DataFrame(np.random.randint(0,50,size=(50, 2)), columns=list('AB'))
data = df.values 
print(data.shape[1]) 



gen = DataGeneratorTimeSeries(df, 5, 1, 15)
