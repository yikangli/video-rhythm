import random
import torch
import torch.multiprocessing as multiprocessing
import signal
import functools
import collections
import re
import sys
import threading
import traceback
from torch._six import string_classes, int_classes
import pdb
import numpy as np

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


def default_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))



class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class BatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class _DataLoaderIter(object):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_sampler = loader.batch_sampler
        self.collate_fn = loader.collate_fn

        self.sample_iter = iter(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self.sample_iter)
        app_batch = []
        #of_batch = []
        label_batch = np.zeros((len(indices),))
        #pdb.set_trace()
        for i in range(len(indices)):
            #(app,of,label) = self.dataset[indices[i]]
            (app,label) = self.dataset[indices[i]]
            app_batch.append(app)
            #of_batch.append(of)
            label_batch[i] = label
        ## Padding Zeros
        max_aplen = 0
        #max_oplen = 0
        app_lengths = []
        #of_lengths = []
        for j in range(len(indices)):
            app_lengths.append(app_batch[j].shape[1])
            #of_lengths.append(of_batch[j].shape[1])
            max_aplen = max(max_aplen,app_batch[j].shape[1])
            #max_oplen = max(max_oplen,of_batch[j].shape[1])
        #pdb.set_trace()
        app_indexs = np.argsort(app_lengths)
        #of_indexs = np.argsort(of_lengths)
        app_indexs = app_indexs[::-1]
        #of_indexs = of_indexs[::-1]
        app_lengths.sort(reverse=True)
        #of_lengths.sort(reverse=True)
        app_feats = np.zeros((max_aplen,len(indices),4096))
        #of_feats = np.zeros((max_oplen,len(indices),4096))
        #mask_app = np.zeros((max_aplen,len(indices)))
        #mask_of  = np.zeros((max_oplen,len(indices)))
        for j in range(len(indices)):
            app_feats[0:app_batch[app_indexs[j]].shape[1],j,:] = app_batch[app_indexs[j]].transpose()
            #of_feats[0:of_batch[of_indexs[j]].shape[1],j,:] = of_batch[of_indexs[j]].transpose()
            #mask_app[0:app_batch[app_indexs[j]].shape[1],j]= 1
            #mask_of[0:of_batch[of_indexs[j]].shape[1],j] = 1
        #batch = self.collate_fn([self.dataset[i] for i in indices])
        #return (app_feats,of_feats,mask_app,mask_of,label_batch)
        #return (app_feats,of_feats,app_lengths,of_lengths,mask_app,mask_of,label_batch)
        return (app_feats,app_lengths,label_batch)

    next = __next__

    def __iter__(self):
        return self


class DataLoader(object):

    def __init__(self, dataset, batch_size=1, collate_fn=default_collate,shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.drop_last = drop_last

        if self.shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        return _DataLoaderIter(self)


