# from fileinput import filename
# import os
# import joblib
# import pandas as pd
# import time
# import numpy as np


# from utils.logger import debug

# # SQL Connection
# from nilmlab.resultLogger import *
# import nest_asyncio
# nest_asyncio.apply()

# LOG_DIR = "/l/users/roberto.guillen/nilm/logs/"
# PRETRAINED_DIR  = "/l/users/roberto.guillen/nilm/pretrained_models/"
# RESULTS_DIR = "/l/users/roberto.guillen/nilm/results/"
# time.sleep(np.random.rand(1))


# exp_name = "cluster_n7_test" + str(np.random.rand(1))
# exp_dir = PRETRAINED_DIR + exp_name

# # Connect/Create SQLite DB
# resultLogger = ResultLogger(LOG_DIR,"cluster_test") #TODO delet 
# resultLogger.makeConnection()
# print("Connection made")
# resultLogger.lockSetting(exp_name) 
# resultLogger.close()
# time.sleep(100)
# #SAVE FILE
# arr = ["cluster_test_Data"]
# df = pd.DataFrame(arr)
# file_name = exp_dir + "_emb.pkl"
# joblib.dump(df,file_name)
# print("Saved df ",df ," in ", file_name)
# time.sleep(1)

# emb = joblib.load(file_name)
# print("Embedding laoded: ", emb," in ", file_name)

# print("storing in db ",exp_name)
# resultLogger.makeConnection()

# resultLogger.updateSetting(exp_name,df)
# resultLogger.close()

import pickle
from nilmlab.sql_manager import ResultLogger

result_logger = ResultLogger(db_name='nilm')

result_logger.create_db()
for row in  result_logger.get_results():
    #print(row[0],pickle.loads(row[1]))
    print(row[0])
    #print(pickle.loads(row[1]))
# print(joblib.load(res))
# result_logger.insert_result('tacos')
# # df =["tienes_el_valor_o_te_vale"]
# # result_logger.update_result(key='drogas', value=df)
# row = result_logger.check_if_exists('drogas')
# if row:
#     print( "exists")
# else:
#     print(row)