import os

dirname = os.path.dirname(__file__)
# UK_DALE = os.path.join(dirname, '../Datasets/UKDALE/ukdale.h5')
# REDD = os.path.join(dirname, '../Datasets/REDD/redd.h5')
UK_DALE = os.path.join(dirname, '../Datasets/UKDALE/ukdale4.h5')
REDD = os.path.join(dirname, '../Datasets/REDD/redd4.h5')

SAVED_MODEL = os.path.join(dirname, "../pretrained_models/s2v/clf-v1.pkl")
PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/s2v/signal2vec-v1.csv')

# MY_SAVED_MODEL = os.path.join(dirname, "../pretrained_models/first_mys2v/xtrachromo-clf-mysignal2vec-v1.pkl")
# MY_PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/first_mys2v/xtrachromo-mysignal2vec-v1.csv')
MY_SAVED_MODEL = os.path.join(dirname, "../pretrained_models/ukdale/KNNweights_max15comp ukdale_1hr_14_23_56-24_05_2022.pkl")
MY_PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/ukdale/emb_max15comp ukdale_1hr_2ep_00_44_54-25_05_2022.csv')