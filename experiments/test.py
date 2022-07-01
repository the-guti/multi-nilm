import os
import traceback
import joblib
import pickle
import pandas as pd
from itertools import product
import time
import numpy as np

from sklearn.neural_network import MLPClassifier
from nilmlab.factories import TransformerFactory

from datasources.datasource import DatasourceFactory
from experiments import GenericExperiment
from nilmlab.factories import EnvironmentFactory
from nilmlab.lab import TimeSeriesLength
from utils.logger import debug

time.sleep(np.random.rand(1))

LOG_DIR = "/l/users/roberto.guillen/nilm/logs/"
PRETRAINED_DIR  = "/l/users/roberto.guillen/nilm/pretrained_models/"
dirname = os.path.abspath('')

    # Results folder
dirname_res = os.path.join(dirname, "results/")
if not os.path.exists(dirname_res):
    os.mkdir(dirname_res)

#     # Pretrained KNN weights folder
# dirname_pre = os.path.join(dirname, "../pretrained_models/")
# if not os.path.exists(dirname_pre):
#     os.mkdir(dirname_pre)

    # Log file folder
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

    # Pretrained models folder 2
if not os.path.exists(PRETRAINED_DIR):
    os.mkdir(PRETRAINED_DIR)

def get_time_series_length(ts_id: str = "Hour"):
    if ts_id == "10Min":
        ts = TimeSeriesLength.WINDOW_10_MINS
    elif ts_id == "Hour":
        ts = TimeSeriesLength.WINDOW_1_HOUR
    elif ts_id == "Day":
        ts = TimeSeriesLength.WINDOW_1_DAY
    return ts


def get_classifier(id: int = 0):
    if id == 0:
        classifier = MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam',early_stopping=True,validation_fraction=0.2)
    elif id == 1:
        classifier = MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam',early_stopping=True,validation_fraction=0.2)
    elif id == 2:
        classifier = MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',early_stopping=True,validation_fraction=0.2)
    elif id == 3:
        classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam',early_stopping=True,validation_fraction=0.2)
    return classifier

def get_datasource(datasource_ix = 0): # 0,false,ukdale    / 1,true,redd
    if datasource_ix:
        datasource_name = 'redd'
        datasource = DatasourceFactory.create_redd_datasource()
        appliances = [
            'unknown', 'electric oven','sockets', 'electric space heater', 'microwave', 
            'washer dryer', 'light', 'electric stove', 'dish washer', 'fridge'
        ]
        
        env = EnvironmentFactory.create_env_single_building(
            datasource=datasource,
            building=1,
            sample_period=6,
            train_year="2011-2011",
            train_start_date="4-1-2011",
            train_end_date="4-30-2011",
            test_year="2011",
            test_start_date="5-1-2011",
            test_end_date="5-2-2011",
            appliances=appliances
        )
    else:
        datasource_name = 'ukdale' # Review name change
        datasource = DatasourceFactory.create_uk_dale_datasource()
        appliances = [
            'microwave', 'dish washer', 'fridge', 'kettle', 'washer dryer',
            'toaster', 'television'
        ]
        env = EnvironmentFactory.create_env_single_building(
            datasource=DatasourceFactory.create_uk_dale_datasource(),
            building=1,
            sample_period=6,
            # train_year="2013-2013",
            # train_start_date="6-2-2013",
            # train_end_date="6-3-2013",
            train_year="2013-2014",
            train_start_date="4-12-2013",
            train_end_date="6-01-2014",
            test_year="2014",
            # test_start_date="6-2-2014",
            # test_end_date="6-3-2014",
            test_start_date="6-2-2014",
            test_end_date="12-30-2014",
            appliances=appliances
        )
    experiment = GenericExperiment(env)
    return datasource_name, appliances, experiment


datasource_ix = 0
datasource_name = "ukdale"

# Comment
# i = 99999999
# components = 8
# ts_length = "Day"
# num_rep_vec = 1
# classifier_type = 0 

for i, (components, ts_length, num_rep_vec,classifier_type) in enumerate(product([2**p for p in range(2, 9)], ["10Min","Hour","Day"], [1,2,4,8],[0,1,2,3])):
    # Review if log file exists
    exp_name = "mys2v_components_%d_tsLength_%s_numRepVec_%d_classifier_type_%d"%(components, ts_length, num_rep_vec,classifier_type) 
    log_file_name = datasource_name + "/" + exp_name + ".log" 
    log_file_path = os.path.join(LOG_DIR, log_file_name)
    if os.path.exists(log_file_path):
        # Skip if exists
        continue
    # Create file
    pickle.dump({},open(log_file_path,"wb"))
    print("\n Working on setting", i, " PATH: " ,log_file_path)

    # Names for files 
    PRETRAINED_DIR = PRETRAINED_DIR + datasource_name + "/" 
    dirname_res = dirname_res + datasource_name + "/" 
    mys2v_knn_weights = os.path.join(PRETRAINED_DIR, f'{exp_name}_weight.pkl')
    mys2v_embedding = os.path.join(PRETRAINED_DIR, f'{exp_name}_emb.pkl')
    results_file_name = os.path.join(dirname_res, f'{exp_name}.csv')
    print(mys2v_knn_weights,mys2v_embedding,results_file_name)

    # Other params
    window_size = 10
    window_step = 1
    epochs  = 1

    exp_name = PRETRAINED_DIR + exp_name

    models =  infer_mysignal2vec_experiment = {
        'MYSIGNAL2VEC_Build' : {
            'CLF_MODELS' : [ 
                get_classifier(classifier_type),
            ],
            'TRANSFORMER_MODELS': [
                TransformerFactory.build_mysignal2vec_train(num_rep_vec, window_size, window_step, components, components, epochs, exp_name),
            ]
        },
        'MYSIGNAL2VEC_Infer' : {
            'CLF_MODELS' : [ 
                get_classifier(classifier_type),
            ],
            'TRANSFORMER_MODELS': [
                TransformerFactory.build_mysignal2vec_infer(mys2v_knn_weights, mys2v_embedding, num_rep_vec),
            ]
        }
    }

    datasource_name, appliances, experiment = get_datasource(datasource_ix)
    for k in models.keys():      
        experiment.setup_running_params(
            transformer_models=models[k]['TRANSFORMER_MODELS'],
            classifier_models=models[k]['CLF_MODELS'],
            train_appliances=appliances,
            test_appliances=appliances,
            ts_len=get_time_series_length(ts_length),
            repeat=1
        )

        experiment.set_checkpoint_file(results_file_name)
        tb = "No error"
        
        try:
            experiment.run()
        except Exception as e:
            tb = traceback.format_exc()
            debug(tb)
            debug(f"Failed for {k}")
            debug(f"{e}")
        
    # TODO change how saving is handled
    df = pd.read_csv(results_file_name)
    joblib.dump(df, log_file_path)