# Author: German A. Garcia Ferrando 
# Contact me: www.github.com/gergf

from abc import ABC, abstractmethod
import warnings

class base_loader(ABC):
    def __init__(self):
        pass

    """
        Args: 
            batch_size: Number of samples in each batch. 
        
        Returns a tuple of three elements (samples, labels, last_batch)
            samples: A numpy array with shape (batch_size, height, width, sample_channels) 
            labels: A numpy array with shape (batch_size, height, width, label_channels)
            last_batch: A bool which represents if the current batch is the last one of the epoch. 
    """
    @abstractmethod
    def get_train_batch(self, batch_size): 
        pass

    """
        Args: 
            batch_size: Number of samples in each batch. 
        
        Returns a tuple of three elements (samples, labels, last_batch)
            samples: A numpy array with shape (batch_size, height, width, sample_channels) 
            labels: A numpy array with shape (batch_size, height, width, label_channels)
            last_batch: A bool which represents if the current batch is the last one of the epoch. 
    """
    @abstractmethod
    def get_valid_batch(self, batch_size): 
        pass 

    """
        Args: 
            batch_size: Number of samples in each batch. 
        
        Returns a tuple of three elements (samples, labels, last_batch)
            samples: A numpy array with shape (batch_size, height, width, sample_channels) 
            labels: A numpy array with shape (batch_size, height, width, label_channels)
            last_batch: A bool which represents if the current batch is the last one of the epoch. 
    """
    @abstractmethod
    def get_test_batch(self, batch_size): 
        pass 
    