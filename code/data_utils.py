from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread, imsave
from scipy.ndimage import gaussian_filter, rotate
import platform
from tqdm import tqdm
import random

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(filepath, num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Package data into a dictionary
    return {
      'X_train': X_train,
      'y_train': y_train,
      'X_val': X_val,
      'y_val': y_val,
      'X_test': X_test,
      'y_test': y_test,
    }


def load_tiny_imagenet(path, is_training=True, dtype=np.float32, subtract_mean=True, debug=False, debug_nclass=3):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Note: The original implementation loaded data as NCHW, I (tyler) changed it to NHWC

    Inputs:
    - path: String giving path to the directory to load.
    - is_training: If True, dont load testing data, if False, dont load training and val data
        Note: Must always load training data in order to subtract_mean.
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.
    - debug: Whether or not to load a small number of classes for debugging

    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 64, 64, 3) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 64, 64, 3) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 64, 64, 3) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (64, 64, 3) array giving mean training image
    - label_to_wnid: dictionary with mapping from integer class label to wnid
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
    label_to_wnid = {v: k for k, v in wnid_to_label.items()}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    if debug:
        print('Debug is on! Only loading %d / %d training classes.'
                  % (debug_nclass, len(wnids)))

    # Next load training data.
    X_train, y_train = [], []
    train_wnids = wnids[:debug_nclass] if debug else wnids
    for i, wnid in tqdm(enumerate(train_wnids), total=len(train_wnids)):
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 64, 64, 3), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * \
                        np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:   ## grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    X_val, y_val = None, None
    if is_training:
        print('loading validation data')
        with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
            img_files = []
            val_wnids = []
            for line in f:
                img_file, wnid = line.split('\t')[:2]
                img_files.append(img_file)
                val_wnids.append(wnid)
            num_val = len(img_files)
            y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
            X_val = np.zeros((num_val, 64, 64, 3), dtype=dtype)
            for i, img_file in tqdm(enumerate(img_files), total=len(img_files)):
                img_file = os.path.join(path, 'val', 'images', img_file)
                img = imread(img_file)
                if img.ndim == 2:
                    img.shape = (64, 64, 1)
                X_val[i] = img

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    X_test, test_image_names = None, None
    if not is_training:
        print('loading testing data')
        img_files = os.listdir(os.path.join(path, 'test', 'images'))
        X_test = np.zeros((len(img_files), 64, 64, 3), dtype=dtype)
        test_image_names = ["unk"]*len(img_files)
        for i, img_file in tqdm(enumerate(img_files), total=len(img_files)):
            img_file = os.path.join(path, 'test', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_test[i] = img
            test_image_names[i] = img_file

    mean_image = None
    if subtract_mean:
        mean_image = X_train.mean(axis=0)
        if is_training:
            X_train -= mean_image[None]
            X_val -= mean_image[None]
        else:
            X_test -= mean_image[None]

    if not is_training:
        X_train = None
        y_train = None

    return {
      'class_names': class_names,
      'X_train': X_train,
      'y_train': y_train,
      'X_val': X_val,
      'y_val': y_val,
      'X_test': X_test,
      'mean_image': mean_image,
      'label_to_wnid': label_to_wnid,
      'test_image_names': test_image_names,
    }


def augment(dataset, fliplr = True, cropAndScale = True, doRotation = True, verbose = True):
    X_train = dataset['X_train']
    y_train = dataset['y_train']    

    if verbose:
        N = X_train.shape[0]
        print(" - Pre augmentation size: " + str(N))
    
    '''
    if doZoom:
        N = X_train.shape[0]
        
        X_train_list = np.split(X_train , N)
        X_train_zoom = []
        for img in X_train_list:
            zoomAmount = (np.random.rand(1) - 0.5) * 60   # Generate random numbers between 30 and -30
            img_rotated = clipped_zoom(img, angle = degree, axes = (1, 2), reshape = False)
            X_train_zoom.append(img_rotated)
        X_train_zoom = np.concatenate(X_train_zoom, axis=0)

        X_train = np.concatenate([X_train, X_train_zoom], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)
    '''

    if doRotation:
        N = X_train.shape[0]
        
        X_train_list = np.split(X_train , N)
        X_train_rotated = []
        for img in X_train_list:
            degree = (np.random.rand(1) - 0.5) * 60   # Generate random numbers between 30 and -30
            img_rotated = rotate(img, angle = degree, axes = (1, 2), reshape = False)
            X_train_rotated.append(img_rotated)
        X_train_rotated = np.concatenate(X_train_rotated, axis=0)

        X_train = np.concatenate([X_train, X_train_rotated], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)

    if fliplr:
        X_train_flipped = np.fliplr(X_train)
        X_train = np.concatenate([X_train, X_train_flipped], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)

    dataset['X_train'] = X_train
    dataset['y_train'] = y_train

    if verbose:
        N = X_train.shape[0]
        print(" - Post augmentation size: " + str(N))

    return dataset


# Random crops and reflection
def augment_batch(X_batch, H2, W2):
    H1, W1, C1 = X_batch[0].shape
    N = len(X_batch)

    X_batch = np.stack(X_batch, axis=0)   # Make into one numpy array


    if random.random() > 0.5:   # 50-50 chance the batch is flipped
        X_batch = np.fliplr(X_batch)

    gapH = H1 - H2
    gapW = W1 - W2

    randH = random.randint(0, gapH)
    randW = random.randint(0, gapW)

    X_batch = X_batch[:, gapH:gapH + H2, gapW:gapW + W2, :] # Random crop
    return X_batch


# Get ten crops of the image
def crop_10(image, H2, W2):
    _, H1, W1, C1 = image.shape

    image_flipped = np.fliplr(image)
    image = np.concatenate([image, image_flipped], axis=0)

    gapH = H1 - H2
    gapW = W1 - W2
    halfGapH = gapH / 2
    halfGapW = gapW / 2

    ul = X_train[:, :H2, :W2, :]   # Upper Left
    br = X_train[:, gapH:, gapW:, :]   # Bottom Right
    ur = X_train[:, :H2, gapW:, :]   # Upper Right
    bl = X_train[:, gapW:, :W2, :]   # Bottom Left
    c = X_train[:, halfGapH:-halfGapH, halfGapW:-halfGapW, :]   # Center

    X_train = np.concatenate([ul, br, ur, bl, c], axis=0)
    return X_train


    

    

        
            