# Author: German A. Garcia Ferrando 
# Contact me: www.github.com/gergf

from .base_loader import base_loader
import nibabel as nib 
import numpy as np 
import os 

class nifti_loader(base_loader):    
    """
        Documentation goes here. 
    """
    def __init__(self, corpus_path, img_height, img_width, img_channels, lab_channels, shuffle_data=True,
        numpy_seed=4669):
        # Set up numpy seed #
        np.random.seed(numpy_seed)

        # Set up corpus specifications #
        self.height   = img_height
        self.width    = img_width 
        self.sample_channels = img_channels
        self.label_channels  = lab_channels
        self.shuffle_data    = shuffle_data 

        self.corpus_set_id = {
            'TRAIN': 0, 
            'VALID': 1, 
            'TEST': 2
        }

        # Create paths #
        self.train_samples_dir = corpus_path + 'train/samples/'
        self.train_labels_dir  = corpus_path + 'train/labels/'
        self.valid_samples_dir = corpus_path + 'valid/samples/'
        self.valid_labels_dir  = corpus_path + 'valid/labels/'
        self.test_samples_dir  = corpus_path + 'test/samples/'
        self.test_labels_dir   = corpus_path + 'test/labels/'

        # Load the directions to the samples #
        self.train_sample_files = np.asarray(sorted([f for f in os.listdir(self.train_samples_dir) if f[-6:] == 'nii.gz']))
        self.train_label_files  = np.asarray(sorted([f for f in os.listdir(self.train_labels_dir)  if f[-6:] == 'nii.gz']))
        self.valid_sample_files = np.asarray(sorted([f for f in os.listdir(self.valid_samples_dir) if f[-6:] == 'nii.gz']))
        self.valid_label_files  = np.asarray(sorted([f for f in os.listdir(self.valid_labels_dir)  if f[-6:] == 'nii.gz']))
        self.test_sample_files  = np.asarray(sorted([f for f in os.listdir(self.test_samples_dir)  if f[-6:] == 'nii.gz']))
        self.test_label_files   = np.asarray(sorted([f for f in os.listdir(self.test_labels_dir)   if f[-6:] == 'nii.gz']))
        
        # Get how many samples there are in each corpus  #
        self.num_train_samples = len(self.train_sample_files)
        self.num_valid_samples = len(self.valid_sample_files)
        self.num_test_samples  = len(self.test_sample_files)

        # Initialize generators #
        self.train_current_index = 0
        self.valid_current_index = 0
        self.test_current_index  = 0 
        try:
            self.train_generator = self._create_train_generator()
            self.valid_generator = self._create_valid_generator()
            self.test_generator  = self._create_test_generator()
        except:
            exit("Something wrong happend while creating the generators. Please check that all paths exists.")

    def _union_shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def _create_train_generator(self):
        return self._sample_label_generator(self.train_sample_files, self.train_label_files, self.train_samples_dir, self.train_labels_dir)

    def _create_valid_generator(self):
        return self._sample_label_generator(self.valid_sample_files, self.valid_label_files, self.valid_samples_dir, self.valid_labels_dir)

    def _create_test_generator(self):
        return self._sample_label_generator(self.test_sample_files, self.test_label_files, self.test_samples_dir, self.test_labels_dir)

    def _sample_label_generator(self, sample_files, label_files, sample_dir, label_dir):
        if self.shuffle_data:
            sample_files, label_files = self._union_shuffle(sample_files, label_files)
        
        for s, l in zip(sample_files, label_files): 
            sample = nib.load(sample_dir + s).get_data()
            label = nib.load(label_dir + l).get_data()
            yield sample, label

    """
        Returns: 
            A tuple (int, bool), where the int is the proper batch_size, and the bool represents 
            if this is the last batch of the epoch or not. 
    """
    def _check_batch_size(self, batch_size, corpus_flag=0):
        if corpus_flag == self.corpus_set_id['TRAIN']: 
            remaining_samples = self.num_train_samples - self.train_current_index

        elif corpus_flag == self.corpus_set_id['VALID']: 
            remaining_samples = self.num_valid_samples - self.valid_current_index

        elif corpus_flag == self.corpus_set_id['TEST']: 
            remaining_samples = self.num_test_samples  - self.test_current_index 
        
        if remaining_samples <= batch_size: 
            return remaining_samples, True
        else: 
            return batch_size, False

    """
        If there is only one channel, the returned shape is (shape[0], shape[1], shape[2]) 
    """
    def _create_zeros_numpy_arrays(self, shape):
        if len(shape) > 3 and shape[3] > 1: 
            return np.zeros(shape)
        else:
            return np.zeros(shape[0:3])

    """
        Args: 
            batch_size: Number of samples in each batch. 
        
        Returns a tuple of three elements (samples, labels, last_batch)
            samples: A numpy array with shape (batch_size, height, width, sample_channels) 
            labels: A numpy array with shape (batch_size, height, width, label_channels)
            last_batch: A bool which represents if the current batch is the last one of the epoch. 
    """
    def get_train_batch(self, batch_size):
        assert batch_size > 0 
        # Check if it is enough samples to fill the batch, if not, change batch_size # 
        batch_size, last_batch = self._check_batch_size(batch_size, self.corpus_set_id['TRAIN']) 
        
        # Arrays to store the batch #
        sam = self._create_zeros_numpy_arrays((batch_size, self.height, self.width, self.sample_channels))
        lab = self._create_zeros_numpy_arrays((batch_size, self.height, self.width, self.label_channels))

        i = 0
        while i < batch_size:
            sam[i], lab[i] = next(self.train_generator)
            self.train_current_index += 1; 
            i += 1; 
        
        # If we have use all the samples, then create a new generator #
        if self.train_current_index == self.num_train_samples:
            self.train_current_index = 0 
            self.train_generator = self._create_train_generator()

        return sam, lab, last_batch

    """
        Args: 
            batch_size: Number of samples in each batch. 
        
        Returns a tuple of three elements (samples, labels, last_batch)
            samples: A numpy array with shape (batch_size, height, width, sample_channels) 
            labels: A numpy array with shape (batch_size, height, width, label_channels)
            last_batch: A bool which represents if the current batch is the last one of the epoch. 
    """
    def get_valid_batch(self, batch_size):
        assert batch_size > 0 
        # Check if it is enough samples to fill the batch, if not, change batch_size # 
        batch_size, last_batch = self._check_batch_size(batch_size, self.corpus_set_id['VALID']) 
        
        # Arrays to store the batch #
        sam = self._create_zeros_numpy_arrays((batch_size, self.height, self.width, self.sample_channels))
        lab = self._create_zeros_numpy_arrays((batch_size, self.height, self.width, self.label_channels))

        i = 0
        while i < batch_size:
            sam[i], lab[i] = next(self.valid_generator)
            self.valid_current_index += 1; 
            i += 1; 
        
        # If we have use all the samples, then create a new generator #
        if self.valid_current_index == self.num_valid_samples:
            self.valid_current_index = 0 
            self.valid_generator = self._create_valid_generator()
        
        return sam, lab, last_batch

    """
        Args: 
            batch_size: Number of samples in each batch. 
        
        Returns a tuple of three elements (samples, labels, last_batch)
            samples: A numpy array with shape (batch_size, height, width, sample_channels) 
            labels: A numpy array with shape (batch_size, height, width, label_channels)
            last_batch: A bool which represents if the current batch is the last one of the epoch. 
    """
    def get_test_batch(self, batch_size): 
        assert batch_size > 0 
        # Check if it is enough samples to fill the batch, if not, change batch_size # 
        batch_size, last_batch = self._check_batch_size(batch_size, self.corpus_set_id['TEST']) 
        
        # Arrays to store the batch #
        sam = self._create_zeros_numpy_arrays((batch_size, self.height, self.width, self.sample_channels))
        lab = self._create_zeros_numpy_arrays((batch_size, self.height, self.width, self.label_channels))
        
        i = 0
        while i < batch_size:
            sam[i], lab[i] = next(self.test_generator)
            self.test_current_index += 1; 
            i += 1; 
        
        # If we have use all the samples, then create a new generator #
        if self.test_current_index == self.num_test_samples:
            self.test_current_index = 0 
            self.test_generator = self._create_test_generator()
        
        return sam, lab, last_batch

    def get_ite_per_epoch(self, batch_size, corpus='train'):
        if corpus.lower() == 'train':
            return np.ceil(self.num_train_samples / batch_size)
        if corpus.lower() == 'valid':
            return np.ceil(self.num_valid_samples / batch_size)
        if corpus.lower() == 'test': 
            return np.ceil(self.num_test_samples  / batch_size)
        raise Exception('Invalid name for corpus. The valid values are: "train", "valid", "test".')

if __name__ == '__main__': 
    print("This script is not meant to be run.")