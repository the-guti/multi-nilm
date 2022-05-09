import os
import traceback

from datasources.datasource import DatasourceFactory
from experiments import GenericExperiment
from nilmlab import exp_model_list
from nilmlab.factories import EnvironmentFactory
from nilmlab.lab import TimeSeriesLength
from nilmlab.exp_model_list import CLF_MODELS, TRANSFORMER_MODELS, SAX
from utils.logger import debug

dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname, "../results")
if not os.path.exists(dirname):
    os.mkdir(dirname)
same_datasource_exp_checkpoint = os.path.join(dirname, 'results_from_my_exp.csv')

appliances = [
    'unknown', 'electric oven','sockets', 'electric space heater', 'microwave', 
    'washer dryer', 'light', 'electric stove', 'dish washer', 'fridge'
 ]

# Configure environment parameters

env = EnvironmentFactory.create_env_single_building(
    datasource=DatasourceFactory.create_redd_datasource(),
    building=1,
    sample_period=6,
    train_year="2011-2011",
    train_start_date="4-19-2011",
    train_end_date="4-29-2011",
    test_year="2011",
    test_start_date="5-19-2011",
    test_end_date="5-20-2011",
    appliances=appliances
)

experiment = GenericExperiment(env)

window = TimeSeriesLength.WINDOW_1_DAY
models = exp_model_list.gmm_experiment

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
