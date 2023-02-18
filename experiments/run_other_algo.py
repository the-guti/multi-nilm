import os
import time
import traceback
import numpy as np
import pandas as pd
from itertools import product

from sklearn.neural_network import MLPClassifier

from nilmlab.factories import TransformerFactory
from datasources.datasource import DatasourceFactory
from experiments import GenericExperiment
from nilmlab.factories import EnvironmentFactory
from nilmlab.lab import TimeSeriesLength
from utils.logger import debug
from datasources.paths_manager import SAVED_MODEL, PATH_SIGNAL2VEC

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from skmultilearn.adapt import MLkNN
from skmultilearn.ensemble import RakelD

# SQL Connection
from nilmlab.sql_manager import ResultLogger

time.sleep(np.random.rand(1)*5)

LOG_DIR = "/l/users/roberto.guillen/nilm/logs/"
PRETRAINED_DIR  = "/l/users/roberto.guillen/nilm/pretrained_models/"
RESULTS_DIR = "/l/users/roberto.guillen/nilm/results/"

SAX = 'SAX'
SAX1D = 'SAX1D'
SFA = 'SFA'
DFT = 'DFT'
PAA = 'PAA'
WEASEL = 'WEASEL'
SIGNAL2VEC = 'SIGNAL2VEC'
MYSIGNAL2VEC = 'MYSIGNAL2VEC'
TRANSFORMER_MODELS = 'TRANSFORMER_MODELS'
CLF_MODELS = 'CLF_MODELS'
BOSS = 'BOSS'
TIME_DELAY_EMBEDDING = 'TIME_DELAY_EMBEDDING'
WAVELETS = 'WAVELETS'


# Results folder CSV
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

# Log file folder
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

# Pretrained models folder 2
if not os.path.exists(PRETRAINED_DIR):
    os.mkdir(PRETRAINED_DIR)

# ----- Helper Functions -----
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
            train_start_date="4-18-2011",
            train_end_date="4-19-2011",
            test_year="2011",
            test_start_date="4-20-2011",
            test_end_date="4-21-2011",
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
            sample_period=6, #TODO delete 
            # train_year="2013-2013",
            # train_start_date="6-1-2013",
            # train_end_date="6-3-2013",
            # train_year="2013-2014",
            # train_start_date="4-12-2013",
            # train_end_date="6-01-2014",
            train_year="2013-2013",
            train_start_date="4-12-2013",  #TODO delete
            train_end_date="6-13-2013",            
            test_year="2014",
            # test_start_date="6-1-2014",
            # test_end_date="6-2-2014",
            test_start_date="6-1-2014",
            test_end_date="6-30-2014", #TODO delete
            # test_start_date="6-2-2014",
            # test_end_date="12-30-2014",
            appliances=appliances
        )
    experiment = GenericExperiment(env)
    return datasource_name, appliances, experiment

# ----- Main Function -----
datasource_ix = 0 

if datasource_ix: # 1
    datasource_name = "redd" 
else: # 0
    datasource_name = "ukdale" 


models =  infer_mysignal2vec_experiment = {
    SIGNAL2VEC: {
        CLF_MODELS        : [
            get_classifier(0),
            get_classifier(1),
            get_classifier(2),
            get_classifier(3)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=5),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=2)
        ]
    },
    WAVELETS            : {
        CLF_MODELS        : [MLkNN(ignore_first_neighbours=0, k=3, s=1.0),
                                RakelD(get_classifier(3), labelset_size=5)],
        TRANSFORMER_MODELS: [TransformerFactory.build_wavelet(), TransformerFactory.build_wavelet()]
    },
    BOSS      : {
        CLF_MODELS        : [
            get_classifier(0),
            get_classifier(1),
            get_classifier(2),
            get_classifier(3)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=5, window_size=10, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=5, window_size=10, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=10, window_size=10, norm_mean=False,
                                                norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=4, window_size=10, norm_mean=False, norm_std=False)
        ]
    },
    PAA       : {
        CLF_MODELS        : [
            get_classifier(0),
            get_classifier(1),
            get_classifier(2),
            get_classifier(3)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True)
        ]
    },
    DFT       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=200)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                                supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                                supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                                supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                                supports_approximation=True)
        ]
    },
    SFA       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam'),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=1000),
            RandomForestClassifier(n_jobs=-1, n_estimators=200),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=3, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=2, norm_mean=False, norm_std=False)
        ]
    },
    SAX1D     : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=20),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=50),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=100)
        ]
    },
    SAX       : {
        CLF_MODELS        : [
            get_classifier(0),
            get_classifier(1),
            get_classifier(2),
            get_classifier(3)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=10, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=50, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=10, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=50, supports_approximation=True)
        ]
    },
    TIME_DELAY_EMBEDDING: {
        CLF_MODELS        : [
            MLkNN(ignore_first_neighbours=0, k=3, s=1.0),
            RakelD(MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive',
                                    solver='adam'), labelset_size=5),
            MLkNN(ignore_first_neighbours=0, k=3, s=1.0),
            RakelD(MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive',
                                    solver='adam'), labelset_size=5)
        ],
        TRANSFORMER_MODELS: [TransformerFactory.build_delay_embedding(delay_in_seconds=30, dimension=6),
                                TransformerFactory.build_delay_embedding(delay_in_seconds=30, dimension=6),
                                TransformerFactory.build_delay_embedding(delay_in_seconds=15, dimension=6),
                                TransformerFactory.build_delay_embedding(delay_in_seconds=15, dimension=6),
                                ]
    },
    WEASEL    : {
        CLF_MODELS        : [
            get_classifier(0),
            get_classifier(1),
            get_classifier(2),
            get_classifier(3)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=4, n_bins=8, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=2, norm_mean=False, norm_std=False)
        ]
    },
}

for i, (k, ts_length) in enumerate(product(models.keys(), ["10Min","Hour","Day"])):
    result_logger = ResultLogger(db_name='algo')
    result_logger.create_db()

    # Prepare name, used as key for sql   #TODO delete
    exp_name = "SQLday_dataset_%s_algo%s_tsLength_%s"%(datasource_name, k, ts_length) 
    
    # Review if key exists, skip 
    rows = result_logger.check_if_exists(exp_name)
    if rows:
        print("skipping", exp_name)
        continue
    # Put key
    result_logger.insert_result(exp_name)

    # Add datasource name to paths 
    pretrained_dir = PRETRAINED_DIR + datasource_name + "/" 
    results_file_dir = RESULTS_DIR + datasource_name + "/" 

    # Names ofr files 
    results_file_dir = os.path.join(results_file_dir, f'{exp_name}.csv')
    print(results_file_dir)

    # Other params
    window_size = 10
    window_step = 1
    epochs  = 1

    exp_name_dir = pretrained_dir + exp_name
        
    # Get dataset
    _, appliances, experiment = get_datasource(datasource_ix)

    # Run experiments
    experiment.setup_running_params(
        transformer_models=models[k]['TRANSFORMER_MODELS'],
        classifier_models=models[k]['CLF_MODELS'],
        train_appliances=appliances,
        test_appliances=appliances,
        ts_len=get_time_series_length(ts_length),
        repeat=1
    )

    experiment.set_checkpoint_file(results_file_dir)
    tb = "No error"
    
    try:
        experiment.run()
    except Exception as e:
        tb = traceback.format_exc()
        debug(tb)
        debug(f"Failed for {k}")
        debug(f"{e}")
        
    # TODO change how saving is handled
    df = pd.read_csv(results_file_dir)

    print("storing",exp_name)
    result_logger.update_result(key=exp_name,value=df)
