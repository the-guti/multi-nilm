import os

dirname = os.path.dirname(__file__)
# UK_DALE = os.path.join(dirname, '../Datasets/UKDALE/ukdale.h5')
# REDD = os.path.join(dirname, '../Datasets/REDD/redd.h5')
UK_DALE = os.path.join(dirname, '../Datasets/UKDALE/ukdale.h5')
REDD = os.path.join(dirname, '../Datasets/REDD/redd.h5')

SAVED_MODEL = os.path.join(dirname, "../pretrained_models/s2v/clf-v1.pkl")
PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/s2v/signal2vec-v1.csv')

# MY_SAVED_MODEL = os.path.join(dirname, "../pretrained_models/first_mys2v/xtrachromo-clf-mysignal2vec-v1.pkl")
# MY_PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/first_mys2v/xtrachromo-mysignal2vec-v1.csv')
MY_SAVED_MODEL = os.path.join(dirname, "../pretrained_models/ukdale/KNNweights_max75comp_ ukdale_1day_00_23_57-20_05_2022.pkl")
MY_PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/ukdale/emb_max75comp_ ukdale_1day_2ep_11_49_21-20_05_2022.csv')