import pickle

import pandas as pd
import numpy as numpy
import time
import psutil
import mpl_toolkits
import scipy
import warnings
import matplotlib.pyplot as plt
import math
import sys
pd.set_option('display.max_columns', None) ## 모든 열을 출력한다.
pd.set_option('display.max_rows', None) ## 모든 열을 출력한다.
warnings.filterwarnings(action='ignore') 

 
# plot을 위한 library
from matplotlib import font_manager, rc
import seaborn as sns
from matplotlib import gridspec
import matplotlib.font_manager as fm  # 폰트 관련 용도

# 자동화를 위한 library
import os 
import os.path
import shutil
import re
import itertools
from sdv.tabular import CTGAN
import csv

loop_num = 16

transformed_data = ["transformed_data" + str(i) for i in range(0, loop_num)]
transformed_data_names = ["transformed_data" + str(i) + ".csv" for i in range(0,loop_num)]

for k in range(0,loop_num): 
    transformed_data[k] = pd.read_csv(transformed_data_names[k])

list_df = ["" for _ in range(loop_num)]

myepoch = 500

model1 = CTGAN(primary_key = 'id', verbose= True, epochs=myepoch, batch_size = 10)
model2 = CTGAN(primary_key = 'id', verbose= True, epochs=myepoch, batch_size = 100)
model3 = CTGAN(primary_key = 'id', verbose= True, epochs=myepoch, batch_size = 300)
model4 = CTGAN(primary_key = 'id', verbose= True, epochs=myepoch, batch_size = 500)
model5 = CTGAN(primary_key = 'id', verbose= True, epochs=myepoch, batch_size = 1000)

for i in range(0,loop_num) : 

    if len(transformed_data[i]) < 10 :
       list_df[i] = transformed_data[i]    

    elif (len(transformed_data[i])>= 10) & (len(transformed_data[i]) < 100) :
        model1.fit(transformed_data[i])
        list_df[i] = model1.sample(len(transformed_data[i])) 

    elif (len(transformed_data[i]) >= 100) & (len(transformed_data[i]) < 1000) :
        model2.fit(transformed_data[i])    
        list_df[i] = model2.sample(len(transformed_data[i])) 

    elif (len(transformed_data[i]) >= 1000) & (len(transformed_data[i]) < 5000) :
        model3.fit(transformed_data[i])    
        list_df[i] = model3.sample(len(transformed_data[i])) 

    elif (len(transformed_data[i]) >= 5000) & (len(transformed_data[i]) < 10000) :
        model4.fit(transformed_data[i])    
        list_df[i] = model4.sample(len(transformed_data[i])) 

    elif len(transformed_data[i]) >= 10000:
        model5.fit(transformed_data[i])    
        list_df[i] = model5.sample(len(transformed_data[i])) 


syn_file_names = ["syn_data_group_" + str(i) + ".csv" for i in range(0,loop_num)]

for j in range(0,loop_num):
    list_df[j].to_csv(syn_file_names[j], index = False)


