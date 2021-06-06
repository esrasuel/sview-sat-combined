import numpy as np
import partitioning
import pickle
import h5py
import pandas as pd
import gdal_tools
import sys

VGGM = -5.24
VGGS = 8.17
SATM = 8.00
SATS = 0.25

def map_labels(labels):
    return labels - 1

def soften_ordinal_labels(labels, m=0.05): # this function softens the ordinal labels for better training.
    labels_ = labels.copy()
    labels_[labels==1] = 1.0 - m
    for l in range(labels.shape[0]): # assuming first dimension is batch and second dimension is classes
        maxindex = np.argmax(labels[l])
        if maxindex == 0:
            labels_[l,1] = m
        elif maxindex == labels.shape[1]-1:
            labels_[l,-2] = m
        else:
            labels_[l,maxindex-1] = m / 2.0
            labels_[l,maxindex+1] = m / 2.0
    return labels_.astype(np.float32)

def soften_ordinal_labels_2D(labels, m=0.05): # this function softens the ordinal labels for better training.
    labels_ = labels.copy()
    labels_[labels==1] = 1.0 - m
    for l in range(labels.shape[0]): # assuming first dimension is batch and fourth dimension is classes
        maxindex = np.argmax(labels[l], axis=2)
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                if maxindex[j,k] == 0:
                    labels_[l,j,k,1] = m
                elif maxindex[j,k] == labels.shape[3]-1:
                    labels_[l,j,k,-2] = m
                else:
                    labels_[l,j,k,maxindex[j,k]-1] = m / 2.0
                    labels_[l,j,k,maxindex[j,k]+1] = m / 2.0
    return labels_.astype(np.float32)

def normalize_features(x):
    return (x - VGGM) / VGGS

def normalize_satellite(x):
    return (np.log(x+1e-4) - SATM) / SATS

class Dataset:
    def __init__(self,
                sat_hdf5_file, 
                sat_lab_hdf5_file,
                label_name, 
                sat_patchsize,
                label_set):
        # allowed label names: dincome, dcrime, dcrowd, denv, dmeanincome
        self.satf = h5py.File(sat_hdf5_file, 'r')
        self.satims = self.satf['satellite_tiles']
        self.psat_size = sat_patchsize
        
        self.labf = h5py.File(sat_lab_hdf5_file, 'r')
        self.labels = self.labf[label_name]
                
        self.label_name=label_name
        self.label_set = label_set
        self.label_num = label_set.size
        
        # = indices to keep track of which samples are used so far within
        # = training. important to count epochs, rather than iterations.
        self.batch_ind = 0
        self.batch_ind_test = 0
        self.batch_ind_valid = 0
        # = place holders
        self.train_part = []
        self.test_part = []
        self.validation_part = []

    # = this function gets features and labels of samples with ids in the
    # = list rows.
    # = in the current implementation, this only works with batchsize = 1
    # = the reasons is that every image will have different number of sview
    # = points in them. Hence, it is not trivial to create a tensor with such
    # = varying number of sview codes.
    def get_data_part(self, rows, noise_std=None, lx_=None, ly_=None, psat_size=None):
        srows = sorted(rows)[0]
        if psat_size is None:
            psat_size_ = self.psat_size
        else:
            psat_size_ = psat_size
        if np.isscalar(srows):
            srows = [srows]
        # getting the satellite images
        if (lx_ is None) and (ly_ is None): 
            lx = np.random.randint(0, self.satims.shape[1] - psat_size_ + 1)
            ly = np.random.randint(0, self.satims.shape[2] - psat_size_ + 1)
        else:
            lx = lx_
            ly = ly_
        img = self.satims[srows, lx:lx+psat_size_, ly:ly+psat_size_, :]
        img = normalize_satellite(img[:,:,:,:4])

        # getting the labels exactly the same way.
        labs = self.labels[srows, lx:lx+psat_size_, ly:ly+psat_size_]
        l = [] 
        for j in range(self.label_num): 
            l_ = np.zeros([len(srows), psat_size_, psat_size_])
            l_[labs == self.label_set[j]] = 1.0
            l = l + [l_]
        l = np.asarray(l).transpose(1,2,3,0).astype(np.float32)
        # setting the sview map of the img to 0. 
        return img, l

    def get_lsoa_info(self, rows, lx_=None, ly_=None, psat_size=None): 
        srows = sorted(rows)[0]
        if psat_size is None:
            psat_size_ = self.psat_size
        else:
            psat_size_ = psat_size
        if np.isscalar(srows):
            srows = [srows]
        # getting the satellite images
        if (lx_ is None) and (ly_ is None): 
            lx = np.random.randint(0, self.satims.shape[1] - psat_size_ + 1)
            ly = np.random.randint(0, self.satims.shape[2] - psat_size_ + 1)
        else:
            lx = lx_
            ly = ly_
        lsoa_info = self.labf['lsoa_ids'][srows, lx:lx+psat_size_, ly:ly+psat_size_]
        return lsoa_info

    def get_train_batch(self, batch_size):
        rows = self.train_part[self.batch_ind: self.batch_ind + batch_size]
        img, l = self.get_data_part(rows)
        self.batch_ind += batch_size
        if self.batch_ind >= len(self.train_part):
            self.batch_ind = 0
        return img, l

    def get_balanced_train_batch(self, batch_size):
        # there is no way of getting balanced sets. 
        return self.get_train_batch(batch_size)

    def get_train_data(self):
        rows = self.train_part
        img, l = self.get_data_part(rows)
        return img, l

    def get_validation_batch(self, batch_size, lx_=None, ly_=None, psat_size=None):
        rows = self.validation_part[self.batch_ind_valid: self.batch_ind_valid + batch_size]
        img, l = self.get_data_part(rows, lx_=lx_, ly_=ly_, psat_size=psat_size)
        self.batch_ind_valid += batch_size
        if self.batch_ind_valid >= len(self.validation_part):
            self.batch_ind_valid = 0
        return img, l

    def get_balanced_validation_batch(self, batch_size, lx_=None, ly_=None):
        return self.get_validation_batch(batch_size, lx_=lx_, ly_=ly_)

    def get_validation_data(self, lx_=None, ly_=None):
        rows = self.validation_part
        img, l = self.get_data_part(rows, lx_=lx_, ly_=ly_)
        return img, l

    def get_test_batch(self, batch_size):
        rows = self.test_part[self.batch_ind_test: self.batch_ind_test + batch_size]
        img, l = self.get_data_part(rows)
        self.batch_ind_test += batch_size
        if self.batch_ind_test >= len(self.test_part):
            self.batch_ind_test = 0
        return img, l

    def get_test_data(self):
        rows = self.test_part
        img, l = self.get_data_part(rows)
        return img, l

    def test_iterator(self, batch_num=1, psat_size=None):
        num_iter = np.int(np.ceil(len(self.test_part) / batch_num))
        for n in range(num_iter):
            rows = self.test_part[n*batch_num : (n+1)*batch_num]
            yield self.get_data_part(rows, psat_size=psat_size), self.get_lsoa_info(rows, psat_size=psat_size)
    
    def validation_iterator(self, batch_num=1, psat_size=None):
        num_iter = np.int(np.ceil(len(self.validation_part) / batch_num))
        for n in range(num_iter): 
            rows = self.validation_part[n*batch_num : (n+1)*batch_num]
            yield self.get_data_part(rows, psat_size=psat_size)

    def training_iterator(self, batch_num=1):
        num_iter = np.int(np.ceil(len(self.train_part) / batch_num))
        for n in range(num_iter): 
            rows = self.train_part[n*batch_num : (n+1)*batch_num]
            yield self.get_data_part(rows)   
            
    def write_preds(self, preds, fname):
        # not done yet. 
        return -1.0

class Dataset_CrossValidation(Dataset):
    def __init__(self,                 
                sat_hdf5_file, 
                sat_lab_hdf5_file,
                label_name, 
                sat_patchsize,
                label_set):
        Dataset.__init__(self,
                         sat_hdf5_file, 
                         sat_lab_hdf5_file,
                         label_name, 
                         sat_patchsize,
                         label_set)

    def pick_label(self, part_gen, part_file, part_kn=5, part_kp=0, vsize=0.1, seed=None):
        '''
            This runs at every creation instance
            label_type: 'cat' (categorial), 'cont' (continuous)
        '''
        if part_gen == 1:
            # = this part creates partitions from the data and saves them in a specified file.
            # = the partitioning is k-folds and stratified.
            # = it also allows constraints, see above comment, as clabels.
            print('==================================== generating partitions from selected classes =====================================')
            self.kpartitions=partitioning.partition_stratified_kfold(part_kn,
                                                                     self.labels,
                                                                     seed=seed,
                                                                     clabels=self.clabels)

            print('==================================== generating partitions =====================================')
            self.kpartitions=partitioning.partition_stratified_kfold(part_kn,
                                                                     self.labels,
                                                                     seed=seed,
                                                                     clabels=self.clabels)
            pickle.dump(self.kpartitions, open(part_file, 'wb'))
        else:
            # = reads a partitioning that was written before.
            # = e.g. if 5 fold cross-validation is used, then this file simply
            # = will have 5 partitions written in it. self.kpartitions is a
            # = list with 5 members and each member has a list of data sample
            # = ids.
            self.kpartitions=pickle.load(open(part_file, 'rb'))

        # = creates training and test part from the self.kpartitions.
        # = e.g. in 5 fold cross validation, it uses the fold part_kp (1...5)
        # = as test and combines the remaining 4 as training data.
        _train_part, self.test_part = partitioning.get_partition_stratified_kfold(part_kp, self.kpartitions)

        
        # = vsize indicates the portion of the training set that will be used as validation. 
        # = the default value is set to 0.1, meaning 10% of all training examples.
        if vsize > 0.0:
            self.train_part, self.validation_part = partitioning.decimate_partition(_train_part, psize=1.0-vsize)
        else:
            self.train_part = _train_part

# = this class simply divides the dataset into three classes:
# == Train (T)
# == Validation (V)
# == Test (T)
class Dataset_TVT(Dataset):
    def __init__(self,                 
                sat_hdf5_file, 
                sat_lab_hdf5_file,
                label_name, 
                sat_patchsize,
                label_set):
        Dataset.__init__(self,
                         sat_hdf5_file, 
                         sat_lab_hdf5_file,
                         label_name, 
                         sat_patchsize,
                         label_set)

    def pick_label(self, part_gen, part_file, train_size, valid_size, psize=1.0, seed=None):
        '''
            This runs at every creation instance
            label_type: 'cat' (categorial), 'cont' (continuous)
        '''
        if part_gen == 1:
            # = this part creates partitions from the data and saves them in a specified file.
            # = the partitioning is stratified and only 3 parts: train, test and validation.
            # = it also allows constrains, see above comment, as clabels.
            print('==================================== generating partitions =====================================')
            _train_part, self.validation_part, self.test_part = partitioning.partition_stratified_validation(self.labels,
                                                                                                                 train_size,
                                                                                                                 valid_size,
                                                                                                                 seed=seed,
                                                                                                                 clabels=self.clabels)
            pickle.dump(_train_part, open(part_file + '_train', 'wb'))
            pickle.dump(self.validation_part, open(part_file + '_validation', 'wb'))
            pickle.dump(self.test_part, open(part_file + '_test', 'wb'))
        else:
            # = reads a partitioning that was written before.
            # = there are three files: validation, test and train
            # = this part reads all of them.
            _train_part=pickle.load(open(part_file + '_train', 'rb'))
            self.validation_part=pickle.load(open(part_file + '_validation', 'rb'))
            self.test_part=pickle.load(open(part_file + '_test', 'rb'))


        # = psize indicates the percentage of training data to be used during
        # = training. If it is 1.0, then we use all the training data. So,
        # = self.train_part = _train_part
        # = if it is less then 1.0 then we take a subset of the training data
        # = with the same proportions of classes, i.e. stratified.
        # = Note that inside decimate_partition_stratified code, we randomly
        # = permute over the samples. So, every time you run this code,
        # = training will happen with another subset of size psize.
        if psize < 1.0:
            self.train_part = partitioning.decimate_partition_stratified(_train_part, self.labels, psize=psize)
        else:
            self.train_part = _train_part

    def write_preds_validation(self, preds, fname):
        # not done yet. 
        return -1.0

# = this class simply divides the dataset into two classes:
# == Train (T)
# == Test (T)
class Dataset_TT(Dataset):
    def __init__(self,                 
                sat_hdf5_file, 
                sat_lab_hdf5_file,
                label_name, 
                sat_patchsize, 
                label_set):
        Dataset.__init__(self,
                         sat_hdf5_file, 
                         sat_lab_hdf5_file,
                         label_name, 
                         sat_patchsize,
                         label_set)

    def pick_label(self, part_gen, part_file, train_size, psize=1.0, seed=None):
        '''
            This runs at every creation instance
            label_type: 'cat' (categorial), 'cont' (continuous)
        '''

        if part_gen == 1:
            # = this part creates partitions from the data and saves them in a specified file.
            # = the partitioning is stratified and only 2 parts: train, and test .

            print('==================================== generating partitions =====================================')
            _train_part, self.test_part = partitioning.partition_stratified(self.labels,
                                                                            train_size,
                                                                            seed=seed,
                                                                            clabels=self.clabels)
            pickle.dump(_train_part, open(part_file + '_train', 'wb'))
            pickle.dump(self.test_part, open(part_file + '_test', 'wb'))
        else:
            # = reads a partitioning that was written before.
            # = there are three files: validation, test and train
            # = this part reads all of them.
            _train_part=pickle.load(open(part_file + '_train', 'rb'))
            self.test_part=pickle.load(open(part_file + '_test', 'rb'))


        # = psize indicates the percentage of training data to be used during
        # = training. If it is 1.0, then we use all the training data. So,
        # = self.train_part = _train_part
        # = if it is less then 1.0 then we take a subset of the training data
        # = with the same proportions of classes, i.e. stratified.
        # = Note that inside decimate_partition_stratified code, we randomly
        # = permute over the samples. So, every time you run this code,
        # = training will happen with another subset of size psize.
        if psize < 1.0:
            self.train_part = partitioning.decimate_partition_stratified(_train_part, self.labels, psize=psize)
        else:
            self.train_part = _train_part

    def get_validation_batch(self, batch_size):
        return self.get_train_batch(batch_size)

    def get_balanced_validation_batch(self, batch_size):
        return self.get_balanced_train_batch(batch_size)

    def write_preds_validation(self, preds, fname):
        print('This function is not written yet.')
        return -1.0


# = this class simply divides the dataset into two classes using a class label:
# == Train (T)
# == Test (T) : test set consists of only the selected class label

class Dataset_TT_byclass(Dataset):
    def __init__(self,                 
                sat_hdf5_file, 
                sat_lab_hdf5_file,
                label_name, 
                sat_patchsize, 
                label_set, 
                label_test=None):
        Dataset.__init__(self,
                         sat_hdf5_file, 
                         sat_lab_hdf5_file,
                         label_name, 
                         sat_patchsize,
                         label_set)
        self.label_test = label_test

    def pick_label(self, part_gen, part_file, train_size, psize=1.0, seed=None):
        '''
            This runs at every creation instance
            label_type: 'cat' (categorial), 'cont' (continuous)
        '''


        if part_gen == 1:
            # = this part creates partitions from the data and saves them in a specified file.
            # = the partitioning is stratified and only 2 parts: train, and test .

            print('==================================== generating partitions =====================================')
            _train_part, self.test_part = partitioning.partition_by_class(self.labels,
                                                                            self.label_test,
                                                                            seed=seed)
            pickle.dump(_train_part, open(part_file + '_train', 'wb'))
            pickle.dump(self.test_part, open(part_file + '_test', 'wb'))

        else:
            # = reads a partitioning that was written before.
            # = there are three files: validation, test and train
            # = this part reads all of them.
            _train_part=pickle.load(open(part_file + '_train', 'rb'))
            self.test_part=pickle.load(open(part_file + '_test', 'rb'))


        # = psize indicates the percentage of training data to be used during
        # = training. If it is 1.0, then we use all the training data. So,
        # = self.train_part = _train_part
        # = if it is less then 1.0 then we take a subset of the training data
        # = with the same proportions of classes, i.e. stratified.
        # = Note that inside decimate_partition_stratified code, we randomly
        # = permute over the samples. So, every time you run this code,
        # = training will happen with another subset of size psize.
        if psize < 1.0:
            self.train_part = partitioning.decimate_partition_stratified(_train_part, self.labels, psize=psize)
        else:
            self.train_part = _train_part

    def get_validation_batch(self, batch_size):
        return self.get_train_batch(batch_size)

    def get_balanced_validation_batch(self, batch_size):
        return self.get_balanced_train_batch(batch_size)

    def write_preds_validation(self, preds, fname):
        print('This function is not done yet.')
        return -1.0
