"""
Model selection environment parameters
- UKDALE Building 1   1/1/2014 - 30/6/2014
- REDD   Building 1,3 1/4/2011 - 30/5/2011
Train-test environment parameters
- UKDALE Building 1
  Train: 1/3/2013 - 30/6/2014
  Test : 1/7/2014 - 31/12/2014
- REDD   Building 1
  Train: 18/4/2011 - 17/5/2011
  Test : 18/5/2011 - 25/5/2011
- REDD   Building 3
  Train: 16/4/2011 - 30/4/2011
  Test : 17/5/2011 - 30/5/2011

  redd 3
  2011-04-16 01:11:24-04:00 - 2011-05-30 20:19:54-04:00

  redd 1
  2011-04-18 09:22:06-04:00 - 2011-05-24 15:57:00-04:00
"""
import os
import traceback

from datasources.datasource import DatasourceFactory
from experiments import GenericExperiment
from nilmlab import exp_model_list
from nilmlab.factories import EnvironmentFactory
from nilmlab.lab import TimeSeriesLength
from nilmlab.exp_model_list import CLF_MODELS, TRANSFORMER_MODELS, BOSS, SIGNAL2VEC, TIME_DELAY_EMBEDDING
from utils.logger import debug

dirname = os.path.dirname(__file__)

RUN_NEW = os.path.join(dirname, '../results/run_new.csv')
APPLIANCES_UK_DALE_BUILDING_1 = ['oven', 'microwave', 'dish washer', 'fridge freezer',
                                 'kettle', 'washer dryer', 'toaster', 'boiler', 'television',
                                 'hair dryer', 'vacuum cleaner', 'light']

ukdale_train_year_start = '2013'
ukdale_train_year_end = '2014'
ukdale_train_month_start = '3'
ukdale_train_month_end = '5'
ukdale_train_end_date = "{}-30-{}".format(ukdale_train_month_end, ukdale_train_year_end)
ukdale_train_start_date = "{}-1-{}".format(ukdale_train_month_start, ukdale_train_year_start)

ukdale_test_year_start = '2014'
ukdale_test_year_end = '2014'
ukdale_test_month_start = '6'
ukdale_test_month_end = '12'
ukdale_test_end_date = "{}-30-{}".format(ukdale_test_month_end, ukdale_test_year_end)
ukdale_test_start_date = "{}-1-{}".format(ukdale_test_month_start, ukdale_test_year_start)

env_ukdale_building_1 = EnvironmentFactory.create_env_single_building(
    datasource=DatasourceFactory.create_uk_dale_datasource(),
    building=1,
    sample_period=6,
    train_year=ukdale_train_year_start + "-" + ukdale_train_year_end,
    train_start_date=ukdale_train_start_date,
    train_end_date=ukdale_train_end_date,
    test_year=ukdale_test_year_start + "-" + ukdale_test_year_end,
    test_start_date=ukdale_test_start_date,
    test_end_date=ukdale_test_end_date,
    appliances=APPLIANCES_UK_DALE_BUILDING_1)

ukdale_building1_experiment = GenericExperiment(env_ukdale_building_1)

def run_experiments(experiment, appliances, window):
    models = exp_model_list.run_new
    for k in models.keys():
        experiment.setup_running_params(
            transformer_models=models[k][TRANSFORMER_MODELS],
            classifier_models=models[k][CLF_MODELS],
            train_appliances=appliances,
            test_appliances=appliances,
            ts_len=window,
            repeat=1)
        experiment.set_checkpoint_file(RUN_NEW)
        tb = "No error"
        try:
            experiment.run()
        except Exception as e:
            tb = traceback.format_exc()
            debug(tb)
            debug(f"Failed for {k}")
            debug(f"{e}")

run_experiments(env_ukdale_building_1, APPLIANCES_UK_DALE_BUILDING_1, TimeSeriesLength.WINDOW_1_HOUR)
