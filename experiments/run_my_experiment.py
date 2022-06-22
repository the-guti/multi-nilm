import os
import traceback

from sklearn.neural_network import MLPClassifier
from nilmlab.factories import TransformerFactory

from datasources.datasource import DatasourceFactory
from experiments import GenericExperiment
from nilmlab.factories import EnvironmentFactory
from nilmlab.lab import TimeSeriesLength
from utils.logger import debug

# Prepare Folders
dirname = os.path.abspath('')

    # Results folder
dirname_res = os.path.join(dirname, "results/")
if not os.path.exists(dirname_res):
    os.mkdir(dirname_res)

    # Pretrained KNN weights folder
dirname_pre = os.path.join(dirname, "pretrained_models/")
if not os.path.exists(dirname_pre):
    os.mkdir(dirname_pre)

#TODO add paths for both datasets in pretrained and resultsS

datasource_ix = 1 # 0,false,ukdale    / 1,true,redd

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
        train_start_date="4-19-2011",
        train_end_date="4-20-2011",
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
        'toaster', 'television', 'hair dryer', 'vacuum cleaner'
    ]
    env = EnvironmentFactory.create_env_single_building(
        datasource=DatasourceFactory.create_uk_dale_datasource(),
        building=1,
        sample_period=6,
        train_year="2013-2013",
        train_start_date="3-1-2013",
        train_end_date="5-30-2013",
        test_year="2014",
        test_start_date="3-1-2014",
        test_end_date="5-30-2014",
        appliances=appliances
    )


# Configure environment parameters
experiment = GenericExperiment(env)

window = TimeSeriesLength.WINDOW_1_HOUR  

# MyS2V hyper-param
num_of_representative_vectors = 1
window_size = 10
window_step = 1
min_n_components = 20
max_n_components = 20
epochs  = 2

# File names 
exp_name =  "/" + datasource_name + "/" + str(window)[24:] + "_" + str(max_n_components) # Remove timeseries from string

mys2v_knn_weights = os.path.join(dirname_pre, f'{exp_name}.pkl')
mys2v_embedding = os.path.join(dirname_pre, f'{exp_name}.csv')
results_file_name = os.path.join(dirname_res, f'{exp_name}.csv')


models =  infer_mysignal2vec_experiment = {
    'MYSIGNAL2VEC_Build' : {
        'CLF_MODELS' : [ 
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam'),
        ],
        'TRANSFORMER_MODELS': [
            TransformerFactory.build_mysignal2vec_train(num_of_representative_vectors, 
                                                        window_size, window_step, min_n_components,
                                                        max_n_components, epochs, exp_name),
        ]
    },
    'MYSIGNAL2VEC_Infer' : {
        'CLF_MODELS' : [ 
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam'),
        ],
        'TRANSFORMER_MODELS': [
           TransformerFactory.build_mysignal2vec_infer(mys2v_knn_weights, mys2v_embedding, num_of_vectors=1),
        ]
    }
}

for k in models.keys():
    # Create fake/temporary 
    
    experiment.setup_running_params(
        transformer_models=models[k]['TRANSFORMER_MODELS'],
        classifier_models=models[k]['CLF_MODELS'],
        train_appliances=appliances,
        test_appliances=appliances,
        ts_len=window,
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
