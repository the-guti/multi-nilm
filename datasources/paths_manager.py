import os

dirname = os.path.dirname(__file__)
# UK_DALE = os.path.join(dirname, '../Datasets/UKDALE/ukdale.h5')
# REDD = os.path.join(dirname, '../Datasets/REDD/redd.h5')
UK_DALE = os.path.join(dirname, '../Datasets/UKDALE/ukdale.h5')
REDD = os.path.join(dirname, '../Datasets/REDD/redd.h5')

SAVED_MODEL = os.path.join(dirname, "../pretrained_models/s2v/clf-v1.pkl")
PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/s2v/signal2vec-v1.csv')

MY_SAVED_MODEL = os.path.join(dirname, "../pretrained_models/first_mys2v/clf-mysignal2vec-v1.pkl")
MY_PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/first_mys2v/mysignal2vec-v1.csv')
# MY_SAVED_MODEL = os.path.join(dirname, "../pretrained_models/knn_texmex.pkl")
# MY_PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/mysignal2vec_texmex.csv')