import os
from doctest import debug_script
import time
from typing import Union, Iterable
from unittest import skip

from matplotlib import pyplot as plt
from numpy import linalg, ndarray

import numpy as np
import pandas as pd
import psutil
import pywt
from loguru import logger
from pyts import approximation, transformation
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import TransformerMixin
# from sklearn.externals import joblib
import joblib
from tslearn import utils as tsutils
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from nilmlab.lab import TimeSeriesTransformer, TransformerType
from utils import chaotic_toolkit
from utils.logger import debug, timing, debug_mem, info

import torch
from torch import Tensor
from torch.nn import NLLLoss
from torch.optim import Adam

from src.models.skip_gram import SkipGram

from datetime import datetime

torch.set_default_dtype(torch.float64)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 60 * 60 * 24

CAPACITY15GB = 1024 * 1024 * 1024 * 15


# Train
class MySignal2VecTrain(TimeSeriesTransformer):

    def __init__(
        self, num_of_representative_vectors: int = 1, window_size: int = 10, 
        window_step: int = 1, min_n_components: int = 1, 
        max_n_components: int = 10, epochs: int = 1,
        exp_name: str = ""
    ):

        super().__init__()

        self.window_size = window_size
        self.window_step = window_step
        self.num_of_representative_vectors = num_of_representative_vectors
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.exp_name =  exp_name
        self.epochs = epochs
        self.name = "mySignal2VecTrain"

        # Initialize components

        self.vocabulary = None
        self.quant_clf = None
        self.skipgram_mlp = None

    def transform(self, series: np.ndarray, sample_period: int = 6) -> np.ndarray:

        # Transform series to appriate dimmension
        series = series.reshape(-1,1)

        # Normalize data  TODO do it in the datasource, review keras normalization layer
        data_min = np.min(series)
        data_max = np.max(series)

        scaled_series = (series - data_min) / (data_max - data_min)

        # Extract the vocabulary and data corpus
        n_clusters, token_sequence = self.get_token_sequence(series=scaled_series, sample_period=sample_period)

        self.vocabulary = np.array(range(n_clusters))

        # Train kNN to quantize in the future
        self.train_quantization_clf(n_neighbors=n_clusters, X=scaled_series, y=token_sequence)

        # Train skip_gram data
        self.train_skipgram(token_sequence=token_sequence)

        raise Exception('MySignal2Vec Infer ended')

    def approximate(self, data_in_batches: np.ndarray, window: int = 1, should_fit: bool = True) -> list:  
        # print(data_in_batches)
       
        raise Exception('MySignal2Vec doesn\'t support approximate yet.')

    def reconstruct(self, series: np.ndarray) -> list:
        raise Exception('MySignal2Vec doesn\'t support reconstruct yet.')

    def get_type(self) -> TransformerType:
        return self.type

    def set_type(self, method_type: TransformerType):
        if method_type == TransformerType.approximate:
            raise Exception('MySignal2vec does not support only approximation. The series has to be transformed firstly')
        self.type = method_type

    # def get_name(self):
    #     raise Exception('MySignal2Vec doesn\'t support get_name yet.')

    def get_name(self):
        return type(self).__name__

    def train_skipgram(self, token_sequence, lr: float = 1e-3) -> None:

        start_time = time.time()

        debug(f'MySignal2Vec.train_skipgram: Start training.')

        self.skipgram_mlp = SkipGram(n_vocabulary=len(self.vocabulary)).to(device)
        criterion = NLLLoss()
        optimizer = Adam(self.skipgram_mlp.parameters(), lr=lr)

        loss_history = []

        for _ in range(self.epochs):

            for i in range(len(token_sequence) - self.window_size):

                batch_skip_gram = self.get_batch_skip_gram(token_sequence, i)

                input  = batch_skip_gram['input'].tolist()
                target = batch_skip_gram['target'].tolist()

                encoded_input = Tensor(self.one_hot_encoder(words=input)).long().to(device)
                encoded_target = Tensor(self.one_hot_encoder(words=target)).long().to(device)

                # Use SkipGram nn

                log_ps = self.skipgram_mlp(encoded_input)
                loss = criterion(log_ps, encoded_target)
                if i % 10000 == 0:
                    loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Checkpoint save skipgram dict
            debug(f'MySignal2Vec.train_skipgram: Checkpoint saving.')
            emb = self.skipgram_mlp.state_dict()['model.0.weight']
            emb_np = [em.detach().cpu().numpy() for em in emb.T]
            df = pd.DataFrame(emb_np)

            # Prepare name of file and save
            dirname = os.path.join(os.path.abspath(''), "pretrained_models/")
            file_name = self.exp_name + "_emb.pkl"
            joblib.dump(df,file_name)

        timing('MySignal2Vec.build_skip_gram: Finished building : {}'.format(round(time.time() - start_time, 2)))


    def get_batch_skip_gram(self, token_sequence: ndarray, index: int) -> ndarray:

        skip_gram = pd.DataFrame(columns=['input', 'target'])

        # Extract information of the segment

        segment = token_sequence[index:index + self.window_size]
        target_ix = int((self.window_size - 1) / 2)
        target = segment[target_ix]
        context = np.delete(segment, target_ix)
        
        # Add positive neighbor records to the skip_gram

        for neighbor in context:

            skip_gram = pd.concat([skip_gram, pd.DataFrame([[target, neighbor]], columns=['input', 'target'])])

        # Add negative neighbor records to the skip_gram

        # neg_neighbors_set = np.concatenate((token_sequence[0:index], token_sequence[index + self.window_size:len(token_sequence)-1]))
        # negative_neighbors = np.random.choice(neg_neighbors_set, 2)

        # for neg_neighbor in negative_neighbors:

        #     skip_gram = pd.concat([skip_gram, pd.DataFrame([[target, neg_neighbor]], columns=['input', 'output', 'target'])])

        # timing('MySignal2Vec.build_skip_gram: Finished building : {}'.format(round(time.time() - start_time, 2)))

        return skip_gram

    def one_hot_encoder(self, words: ndarray) -> ndarray:

        return np.eye(len(self.vocabulary))[words]

    def train_quantization_clf(self, n_neighbors: int, X: ndarray, y: ndarray):

        debug(f'MySignal2Vec.train_quantization_clf: Train.')

        self.quant_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        debug(f'Quantization N tokens {n_neighbors}')

        start_time = time.time()
        self.quant_clf.fit(X=X, y=y)
        # Save file
        debug(f'MySignal2Vec.train_quantization_clf: saving checkpoint.')
        dirname = os.path.join(os.path.abspath(''), "pretrained_models/")

        file_name = self.exp_name + "_weight.pkl"
        joblib.dump(self.quant_clf,file_name)
        

        timing('MySignal2Vec.train_quantization_clf: Finished : {}'.format(round(time.time() - start_time, 2)))
        

    def get_token_sequence(self, series: ndarray, sample_period: int):

        debug(f'MySignal2Vec.get_token_sequence: Transform to discrete space.')

        data_chunks = self.split_in_chunks(series, sample_period)
        debug_mem('Time series {} MB', series)
        debug_mem('Data chunks {} MB', data_chunks)

        start_time = time.time()
        clustering_model = self.get_best_clustering_model(data_chunks=data_chunks)
        timing('MySignal2Vec.get_token_sequence: Getting best GMM : {}'.format(round(time.time() - start_time, 2)))
        
        # Pass from the continuous series to a discrete series

        sequence_of_tokens = clustering_model.predict(series)

        return clustering_model.n_components, sequence_of_tokens

    def get_best_clustering_model(self, data_chunks: ndarray) -> GaussianMixture:
        # TODO review normalize for bic 
        lowest_bic = np.infty
        bic = []
        n_components_range = range(self.min_n_components, self.max_n_components+1) 
        # cv_types = [ "spherical", "tied", "diag", "full" ]

        # for cv_type in cv_types:

        for n_components in n_components_range:

            gmm = GaussianMixture(
                n_components=n_components, covariance_type='full', warm_start=True, reg_covar=1e-6
            )

            # number of chunks depends on the available memory, usually just one
            for chunk in data_chunks:

                # Fit a Gaussian mixture with EM
                gmm.fit(chunk)
                bic.append(gmm.bic(chunk))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm 
            # debug(f'Finished cv_type {cv_type}' )


        bic = np.array(bic)

        debug(f'Best BIC {lowest_bic}' )
        return best_gmm

    def map_into_vectors(self, sequence):
        start_time = time.time()

        # sequence_of_vectors = [self.embedding[str(i)] for i in sequence]
        # timing('Appending vectors to list : {}'.format(round(time.time() - start_time, 2)))
        # return sequence_of_vectors
        raise Exception('MySignal2Vec doesn\'t support map_into_vectors yet.')

    def split_in_chunks(self, sequence, sample_period: int = 6):
        memory = psutil.virtual_memory()
        debug('Memory: {}'.format(memory))
        chunk_size = sample_period * SECONDS_PER_DAY
        if memory.total >= CAPACITY15GB:
            chunk_size = chunk_size * 2
        # seq = list()
        split_n = max(int(len(sequence) / chunk_size), 1)
        rem = len(sequence) % split_n
        if rem != 0:
            sequence = sequence[:-rem]

        debug('Spliting data into {} parts for memory efficient clustering'.format(split_n))
        return [chunk.reshape(-1,1) for chunk in np.split(sequence, split_n)]

# Infer
class MySignal2VecInfer(TimeSeriesTransformer):

    def __init__(self, classifier_path: str, embedding_path: str, num_of_representative_vectors: int = 1):
        super().__init__()
        # Save directory of weights and embeddings
        self.classifier_path = classifier_path
        self.embedding_path = embedding_path
        self.type = TransformerType.transform_and_approximate
        self.num_of_representative_vectors = num_of_representative_vectors
        self.name = "mySignal2VecInfer"

        # Init empty 
        self.clf = None
        self.embedding = None

    def __repr__(self):
        return f"Signal2Vec num_of_representative_vectors: {self.num_of_representative_vectors}"

    def load_weights(self):
        self.clf = joblib.load(self.classifier_path)
        self.embedding = joblib.load(self.embedding_path)
        #TODO review
        # embedding = pd.read_csv(self.embedding_path)
        # self.embedding = embedding.reset_index().to_dict('list')

    def transform(self, series: np.ndarray, sample_period: int = 6) -> np.ndarray:
        # TODO move function elsewhere Load weights
        self.load_weights()

        discrete_series = self.discretize_in_chunks(series, sample_period)
        debug_mem('Time series {} MB', series)
        debug_mem('Discrete series {} MB', discrete_series)

        vector_representation = self.map_into_vectors(discrete_series)
        debug_mem('Sequence of vectors : {} MB', vector_representation)

        return np.array(vector_representation)

    def approximate(self, data_in_batches: np.ndarray, window: int = 1, should_fit: bool = True) -> list:
        # TODO: Window is used only by signal2vec, move it to constructor or extract it as len(segment).
        if self.num_of_representative_vectors > 1:
            window = int(window / self.num_of_representative_vectors)
            data_in_batches = np.reshape(data_in_batches,
                                         (len(data_in_batches), window, 256 * self.num_of_representative_vectors))
        # Squeeze vectors into 1 of 256 size
        squeezed_seq = np.sum(data_in_batches, axis=1)
        vf = np.vectorize(lambda x: x / window)
        squeezed_seq = vf(squeezed_seq)
        return squeezed_seq

    def reconstruct(self, series: np.ndarray) -> list:
        raise Exception('Signal2Vec doesn\'t support reconstruct yet.')

    def get_name(self):
        return type(self).__name__

    def get_type(self):
        return self.type

    def set_type(self, method_type: TransformerType):
        if method_type == TransformerType.approximate:
            raise Exception('Signal2vec does not support only approximation. The series has to be transformed firstly')
        self.type = method_type

    def discretize(self, data):
        debug('Length of data {}'.format(len(data)))
        start_time = time.time()

        pred = self.clf.predict(data.reshape(-1, 1))

        timing('clf.predict: {}'.format(round(time.time() - start_time, 2)))
        debug('Length of predicted sequence {}'.format(len(pred)))
        debug('Type of discrete sequence {}'.format(type(pred)))

        return pred

    def map_into_vectors(self, sequence):
        start_time = time.time()
        sequence_of_vectors = [self.embedding[i] for i in sequence]
        # sequence_of_vectors = [self.embedding[str(i)] for i in sequence] 

        timing('Appending vectors to list : {}'.format(round(time.time() - start_time, 2)))
        return sequence_of_vectors

    def discretize_in_chunks(self, sequence, sample_period: int = 6):
        memory = psutil.virtual_memory()
        debug('Memory: {}'.format(memory))
        chunk_size = sample_period * SECONDS_PER_DAY
        if memory.total >= CAPACITY15GB:
            chunk_size = chunk_size * 2
        seq = list()
        split_n = max(int(len(sequence) / chunk_size), 1)
        rem = len(sequence) % split_n
        if rem != 0:
            sequence = sequence[:-rem]

        debug('Spliting data into {} parts for memory efficient classification'.format(split_n))
        for d in np.split(sequence, split_n):
            debug('Discretising time series...')
            s = self.discretize(d)
            seq.append(s)

        return np.concatenate(seq)

    def get_name(self):
        pass

class Signal2Vec(TimeSeriesTransformer):

    def __init__(self, classifier_path: str, embedding_path: str, num_of_representative_vectors: int = 1):
        super().__init__()
        self.clf = joblib.load(classifier_path)
        embedding = pd.read_csv(embedding_path)
        self.embedding = embedding.reset_index().to_dict('list')
        self.type = TransformerType.transform_and_approximate
        self.num_of_representative_vectors = num_of_representative_vectors

    def __repr__(self):
        return f"Signal2Vec num_of_representative_vectors: {self.num_of_representative_vectors}"

    def transform(self, series: np.ndarray, sample_period: int = 6) -> np.ndarray:
        discrete_series = self.discretize_in_chunks(series, sample_period)
        debug_mem('Time series {} MB', series)
        debug_mem('Discrete series {} MB', discrete_series)

        vector_representation = self.map_into_vectors(discrete_series)
        debug_mem('Sequence of vectors : {} MB', vector_representation)

        return np.array(vector_representation)

    def approximate(self, data_in_batches: np.ndarray, window: int = 1, should_fit: bool = True) -> list:
        # TODO: Window is used only by signal2vec, move it to constructor or extract it as len(segment).
        if self.num_of_representative_vectors > 1:
            window = int(window / self.num_of_representative_vectors)
            data_in_batches = np.reshape(data_in_batches,
                                         (len(data_in_batches), window, 300 * self.num_of_representative_vectors))

        squeezed_seq = np.sum(data_in_batches, axis=1)
        vf = np.vectorize(lambda x: x / window)
        squeezed_seq = vf(squeezed_seq)
        return squeezed_seq

    def reconstruct(self, series: np.ndarray) -> list:
        raise Exception('Signal2Vec doesn\'t support reconstruct yet.')

    def get_name(self):
        return type(self).__name__

    def get_type(self):
        return self.type

    def set_type(self, method_type: TransformerType):
        if method_type == TransformerType.approximate:
            raise Exception('Signal2vec does not support only approximation. The series has to be transformed firstly')
        self.type = method_type

    def discretize(self, data):
        debug('Length of data {}'.format(len(data)))
        start_time = time.time()

        pred = self.clf.predict(data.reshape(-1, 1))

        timing('clf.predict: {}'.format(round(time.time() - start_time, 2)))
        debug('Length of predicted sequence {}'.format(len(pred)))
        debug('Type of discrete sequence {}'.format(type(pred)))

        return pred

    def map_into_vectors(self, sequence):
        start_time = time.time()
        sequence_of_vectors = [self.embedding[str(i)] for i in sequence]
        timing('Appending vectors to list : {}'.format(round(time.time() - start_time, 2)))
        return sequence_of_vectors

    def discretize_in_chunks(self, sequence, sample_period: int = 6):
        memory = psutil.virtual_memory()
        debug('Memory: {}'.format(memory))
        chunk_size = sample_period * SECONDS_PER_DAY
        if memory.total >= CAPACITY15GB:
            chunk_size = chunk_size * 2
        seq = list()
        split_n = max(int(len(sequence) / chunk_size), 1)
        rem = len(sequence) % split_n
        if rem != 0:
            sequence = sequence[:-rem]

        debug('Spliting data into {} parts for memory efficient classification'.format(split_n))
        for d in np.split(sequence, split_n):
            debug('Discretising time series...')
            s = self.discretize(d)
            seq.append(s)

        return np.concatenate(seq)


class WaveletAdapter(TimeSeriesTransformer):
    """
    http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
    """

    def __init__(self, wavelet_name: str = 'haar', filter_bank: str = None, mode='symmetric', level=None,
                 drop_cA=False):
        super().__init__()
        self.wavelet_name = wavelet_name
        self.filter_bank = filter_bank
        self.mode = mode
        self.level = level
        self.drop_cA = drop_cA
        self.type = TransformerType.approximate

    def __repr__(self):
        return str(f"Wavelet {self.wavelet_name}, level {self.level}, filter_bank {self.filter_bank}, "
                   f"mode {self.mode}, drop_cA {self.drop_cA}")

    def transform(self, series: np.ndarray, sample_period: int = 6) -> list:
        debug('WaveletAdapter series shape {}'.format(series.shape))
        coeffs = pywt.wavedec(data=series, wavelet=self.wavelet_name, level=self.level, mode=self.mode)
        ts_representation = pywt.waverec(coeffs, wavelet=self.wavelet_name, mode=self.mode)
        debug('WaveletAdapter series shape after inverse {}'.format(series.shape))
        return ts_representation[0].ravel()

    def approximate(self, series: np.ndarray, window: int = 1, should_fit: bool = True) -> np.ndarray:
        ts_representation = list()
        debug(f'WaveletAdapter.approximate: param series \n{series} ')
        for segment in series:
            coeffs = pywt.wavedec(data=segment, wavelet=self.wavelet_name,
                                  level=self.level, mode=self.mode)
            if self.drop_cA:
                coeffs = coeffs[0]
            else:
                coeffs = np.concatenate(coeffs)

            ts_representation.append(coeffs)
        # debug('TSLearnApproximatorWrapper.approximate: ts_representation \n{}'.format(ts_representation))
        debug('WaveletAdapter.approximate: ts_representation shape {}'.format(np.shape(ts_representation)))
        # ts_representation = np.reshape(ts_representation, (
        #     np.shape(ts_representation)[0], np.shape(ts_representation)[1] * np.shape(ts_representation)[2]))
        # debug('WaveletAdapter.approximate: ts_representation \n{}'.format(ts_representation))
        return np.asarray(ts_representation)

    def reconstruct(self, series: np.ndarray) -> list:
        raise Exception('WaveletAdapter doesn\'t support reconstruct yet.')

    def get_type(self) -> TransformerType:
        return self.type

    def set_type(self, method_type: TransformerType):
        self.type = method_type

    def get_name(self):
        return type(self).__name__


class TimeDelayEmbeddingAdapter(TimeSeriesTransformer):
    """
    http://eprints.maths.manchester.ac.uk/175/1/embed.pdf
    """

    def __init__(self, delay_in_seconds: int, dimension: int, sample_period: int = 6):
        super().__init__()
        self.delay_in_seconds = delay_in_seconds
        self.dimension = dimension
        self.sample_period = sample_period
        self.type = TransformerType.approximate

    def __repr__(self):
        return f"TimeDelayEmbedding delay={self.delay_in_seconds} dim={self.dimension}"

    def transform(self, series: np.ndarray, sample_period: int = 6) -> list:
        """
        Given a whole time series, it is automatically segmented into segments with size
        window_size = delay_items * self.dimension. Next, delay embeddings are extracted for each segment.
        """
        delay_items = int(self.delay_in_seconds / sample_period)
        window_size = delay_items * self.dimension
        num_of_segments = int(len(series) / window_size)
        delay_embeddings = []
        for i in range(num_of_segments):
            segment = series[i * window_size:(i + 1) * window_size]
            embedding = chaotic_toolkit.takens_embedding(segment, delay_items, self.dimension)
            delay_embeddings.append(embedding)
        return delay_embeddings

    def approximate(self, series_in_segments: np.ndarray, window: int = 1, should_fit: bool = True) -> np.ndarray:
        """
        The time series is given as segments. For each segment we extract the delay embeddings.
        """
        delay_items = int(self.delay_in_seconds / self.sample_period)
        window_size = delay_items * self.dimension

        if window_size > len(series_in_segments[0]):
            raise Exception(
                f'Not enough data for the given delay ({self.delay_in_seconds} seconds) and dimension ({self.dimension}).'
                f'\ndelay_items * dimension > len(data): {window_size} > {len(series_in_segments[0])}')

        if window_size == len(series_in_segments[0]):
            info(f"TimeDelayEmbeddingAdapter is applied with delay embeddings equavalent to the length of each segment"
                 f" {window_size} == {len(series_in_segments[0])}")

        if window_size < len(series_in_segments[0]):
            info(f"TimeDelayEmbeddingAdapter is applied with delay embeddings covering less than the length of each "
                 f"segment. {window_size} < {len(series_in_segments[0])}")

        delay_embeddings = []
        for segment in series_in_segments:
            embedding = chaotic_toolkit.takens_embedding(segment, delay_items, self.dimension)
            delay_embeddings.append(embedding)
        return np.asarray(delay_embeddings)

    def reconstruct(self, series: np.ndarray) -> list:
        raise Exception('TimeDelayEmbeddingAdapter doesn\'t support reconstruct yet.')

    def get_type(self) -> TransformerType:
        return self.type

    def set_type(self, method_type: TransformerType):
        self.type = method_type

    def get_name(self):
        return type(self).__name__


class TSLearnTransformerWrapper(TimeSeriesTransformer):

    def __init__(self, transformer: TransformerMixin, supports_approximation: bool = True):
        super().__init__()
        if not isinstance(transformer, TransformerMixin):
            raise Exception('Invalid type of approximator. It should be an instance of TransformerMixin.')
        self.transformer = transformer
        self.supports_approximation = supports_approximation
        if supports_approximation:
            self.type = TransformerType.approximate
        else:
            self.type = TransformerType.transform

    def __repr__(self):
        return str(self.transformer)

    def transform(self, series: np.ndarray, sample_period: int = 6) -> list:
        debug('TSLearnApproximatorWrapper series shape {}'.format(series.shape))
        ts_representation = self.transformer.inverse_transform(self.transformer.fit_transform(series))
        return ts_representation[0].ravel()

    def approximate(self, series: np.ndarray, window: int = 1, should_fit: bool = True) -> np.ndarray:
        # series is already in batches
        debug('TSLearnApproximatorWrapper.approximate: series shape {}'.format(series.shape))
        debug('TSLearnApproximatorWrapper.approximate: to_time_series shape {}'.format(series.shape))
        ts_representation = list()
        debug(f'TSLearnApproximatorWrapper.approximate: param series \n{series} ')
        for segment in series:
            if isinstance(self.transformer, SymbolicAggregateApproximation) or isinstance(self.transformer, OneD_SymbolicAggregateApproximation):
                logger.info("Scaling the data so that they consist a normal distribution.")
                scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
                segment = scaler.fit_transform(segment.reshape(-1,1))
            ts_representation.append(self.transformer.fit_transform(segment))
        # debug('TSLearnApproximatorWrapper.approximate: ts_representation \n{}'.format(ts_representation))
        debug('TSLearnApproximatorWrapper.approximate: ts_representation shape {}'.format(np.shape(ts_representation)))
        ts_representation = np.reshape(ts_representation, (
            np.shape(ts_representation)[0], np.shape(ts_representation)[1] * np.shape(ts_representation)[2]))
        debug('TSLearnApproximatorWrapper.approximate: ts_representation \n{}'.format(ts_representation))
        debug('TSLearnApproximatorWrapper.approximate: ts_representation shape {}'.format(ts_representation.shape))
        return ts_representation

    def reconstruct(self, series: np.ndarray) -> list:
        raise Exception('Pyts doesn\'t support reconstruct.')

    def get_type(self):
        return self.type

    def get_name(self):
        return type(self.transformer).__name__

    def set_type(self, method_type: TransformerType):
        if not self.supports_approximation and method_type == TransformerType.approximate:
            raise Exception('{} does not support approximation.'.format(type(self.transformer).__name__))
        self.type = method_type


class PytsTransformerWrapper(TimeSeriesTransformer):

    def __init__(self, transformer, supports_approximation: bool = True):
        super().__init__()
        if not isinstance(transformer, TransformerMixin):
            raise Exception('Invalid type of approximator. It should be an instance of TransformerMixin.')
        self.transformer = transformer
        self.supports_approximation = supports_approximation
        if supports_approximation:
            self.type = TransformerType.approximate
        else:
            self.type = TransformerType.transform

    def __repr__(self):
        return str(self.transformer)

    def transform(self, series: np.ndarray, sample_period: int = 6) -> Union[np.ndarray, Iterable, int, float]:
        if isinstance(self.transformer, approximation.DiscreteFourierTransform):
            n_coefs = self.transformer.n_coefs
            series = tsutils.to_time_series(series)
            series = np.reshape(series, (1, -1))
            n_samples, n_timestamps = series.shape
            self.transformer.drop_sum = True
            X_dft = self.transformer.fit_transform(series)

            # Compute the inverse transformation
            if n_coefs % 2 == 0:
                real_idx = np.arange(1, n_coefs, 2)
                imag_idx = np.arange(2, n_coefs, 2)
                X_dft_new = np.c_[
                    X_dft[:, :1],
                    X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx],
                                                    np.zeros((n_samples,))]
                ]
            else:
                real_idx = np.arange(1, n_coefs, 2)
                imag_idx = np.arange(2, n_coefs + 1, 2)
                X_dft_new = np.c_[
                    X_dft[:, :1],
                    X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
                ]
            X_irfft = np.fft.irfft(X_dft_new, n_timestamps)
            debug('PytsTransformerWrapper ts_representation shape {}'.format(np.shape(X_irfft)))

            return np.ravel(X_irfft)
        else:
            raise Exception('Pyts doesn\'t support trasform')

    def approximate(self, series: np.ndarray, window: int = 1, target=None, should_fit: bool = True) -> list:
        # series is already in batches
        debug('PytsTransformerWrapper series shape {}'.format(series.shape))
        if isinstance(self.transformer, transformation.WEASEL):
            labels = list()
            for t in target:
                l: str = ''
                for i in range(target.shape[1]):
                    l = l + str(int(t[i]))
                labels.append(l)

            if should_fit:
                ts_representation = self.transformer.fit_transform(series, labels)
            else:
                ts_representation = self.transformer.transform(series)
        else:
            if should_fit:
                ts_representation = self.transformer.fit_transform(series)
            elif isinstance(self.transformer, transformation.BOSS):
                debug("BOSS instance, only transform")
                ts_representation = self.transformer.transform(series)
            else:
                debug("Fit transform.")
                ts_representation = self.transformer.fit_transform(series)

        # debug('PytsTransformerWrapper ts_representation \n{}'.format(ts_representation))
        debug('PytsTransformerWrapper ts_representation shape {}'.format(np.shape(ts_representation)))
        return ts_representation

    def reconstruct(self, series: np.ndarray) -> list:
        raise Exception('Pyts doesn\'t support reconstruct.')

    def get_type(self):
        return self.type

    def set_type(self, method_type: TransformerType):
        if not self.supports_approximation and method_type == TransformerType.approximate:
            raise Exception('{} does not support approximation.'.format(type(self.transformer).__name__))
        self.type = method_type

    def get_name(self):
        return type(self.transformer).__name__

    def uses_labels(self):
        if isinstance(self.transformer, transformation.WEASEL):
            return True
        return False
