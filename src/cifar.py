# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Downloads and extracts the binary version of the CIFAR-100 dataset."""
import numpy as np
import pickle
import os
import urllib.request
import sys
import tarfile

def unpickle(file):
    print("loading " + file)
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

    return dict

def load_cifar100():
    DATA_PATH = "./data"
    DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(DATA_PATH,filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            print('\r>> Downloading %s %.1f%%'%(
                filename, 100.0 * count * block_size / total_size), end='')

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    os.chdir(DATA_PATH)
    data = {}
    tarfile.open(filename, 'r:gz').extractall()
    os.chdir("./cifar-100-python")
    for file in ['train', 'test']:
        dict = unpickle(file)
        data[file] = dict

    return data
