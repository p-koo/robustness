import os, h5py
import numpy as np



def load_data(filepath):
  with h5py.File(filepath, 'r') as dataset:
    x_train = np.array(dataset['X_train']).astype(np.float32)
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32)
    y_valid = np.array(dataset['Y_valid']).astype(np.int32)
    x_test = np.array(dataset['X_test']).astype(np.float32)
    y_test = np.array(dataset['Y_test']).astype(np.int32)
    model_test = np.array(dataset['model_test']).astype(np.float32)
 
  model_test = model_test.transpose([0,2,1])
  x_train = x_train.transpose([0,2,1])
  x_valid = x_valid.transpose([0,2,1])
  x_test = x_test.transpose([0,2,1])
 
  N, L, A = x_train.shape
  return x_train, y_train, x_valid, y_valid, x_test, y_test
 


def make_directory(path, foldername, verbose=1):
  """make a directory"""

  if not os.path.isdir(path):
    os.mkdir(path)
    print("making directory: " + path)

  outdir = os.path.join(path, foldername)
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
    print("making directory: " + outdir)
  return outdir