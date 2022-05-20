import os
import traceback

from datasources.datasource import DatasourceFactory
from experiments import GenericExperiment
from nilmlab import exp_model_list
from nilmlab.factories import EnvironmentFactory
from nilmlab.lab import TimeSeriesLength
from nilmlab.exp_model_list import CLF_MODELS, TRANSFORMER_MODELS
from utils.logger import debug

dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname, "../results")
if not os.path.exists(dirname):
    os.mkdir(dirname)

window = TimeSeriesLength.WINDOW_1_DAY  
datasource_ix = 0 # 0,false,ukdale    / 1,true,redd

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
        train_end_date="4-30-2011",
        test_year="2011",
        test_start_date="5-1-2011",
        test_end_date="5-2-2011",
        appliances=appliances
    )
else:
    datasource_name = 'uk_dale'
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

same_datasource_exp_checkpoint = os.path.join(dirname, f'results_from_my_{window}_{datasource_name}.csv')

# Configure environment parameters

experiment = GenericExperiment(env)

# models = exp_model_list.my_experiment
# models = exp_model_list.mysignal2vec_experiment
# models = exp_model_list.gmm_experiment

models = {}
if window == TimeSeriesLength.WINDOW_10_MINS:
    models = exp_model_list.selected_models_10mins
elif window == TimeSeriesLength.WINDOW_1_HOUR:
    models = exp_model_list.selected_models_1h
elif window == TimeSeriesLength.WINDOW_2_HOURS:
    models = exp_model_list.selected_models_2h
elif window == TimeSeriesLength.WINDOW_8_HOURS:
    models = exp_model_list.selected_models_8h
elif window == TimeSeriesLength.WINDOW_4_HOURS:
    models = exp_model_list.selected_models_4h
elif window == TimeSeriesLength.WINDOW_1_DAY:
    models = exp_model_list.selected_models_24h

for k in models.keys():
    
    experiment.setup_running_params(
        transformer_models=models[k][TRANSFORMER_MODELS],
        classifier_models=models[k][CLF_MODELS],
        train_appliances=appliances,
        test_appliances=appliances,
        ts_len=window,
        repeat=1
    )

    experiment.set_checkpoint_file(same_datasource_exp_checkpoint)
    tb = "No error"
    
    try:
        experiment.run()
    except Exception as e:
        tb = traceback.format_exc()
        debug(tb)
        debug(f"Failed for {k}")
        debug(f"{e}")
