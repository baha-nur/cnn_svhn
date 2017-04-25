import numpy as np

class DataSet(object):
  def __init__(self, images, labels, normalize=True):
    assert images.shape[0] == labels.shape[0], (
        "images.shape: %s labels.shape: %s" % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]

    if normalize:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)
      
    self._images = np.array(images)
    self._labels = np.array(labels)
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
  @property
  def images(self):
    return self._images
  
  @property
  def labels(self):
    return self._labels
  
  @property
  def num_examples(self):
    return self._num_examples
  
  @property
  def epochs_completed(self):
    return self._epochs_completed
  
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    assert batch_size <= self._num_examples
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    
    # Finished epoch
    if self._index_in_epoch > self._num_examples:
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      
    end = self._index_in_epoch
    batch = self._images[start:end], self._labels[start:end]
    return batch

# Example:
# class DataSets(object):
#   pass

# data_sets = DataSets()
# data_sets.train = DataSet(train_data, dense_to_one_hot(train_labels), normalize=True)
# data_sets.test = DataSet(test_data, dense_to_one_hot(test_labels), normalize=True)