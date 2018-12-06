''' Classify flower and vehicles1 in cifar100
    2500 images for flowers
    2500 images for vehicles1
'''
import cifar
import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_file(data, usage, supercls):
    ''' load specific data
        here, 
        usage:
            'train',
            'test',
        supercls:
            2 to be flowers,
            18 to be vehicles,
    '''
    imgs = []
    for idx, val in enumerate(data[usage][b'coarse_labels']):
        if val == supercls:
            imgs.append(data[usage][b'data'][idx])

    return np.vstack(imgs)

def main():
    data = cifar.load_cifar100()
    # load flowers train and valid
    flower = load_file(data, 'train', 2)
    vehicle = load_file(data, 'train', 18)
    flower_test = load_file(data, 'test', 2)
    vehicle_test = load_file(data, 'test', 18)
    
    train_img = np.vstack((flower,vehicle))
    train_label = np.array([0]*len(flower) + [1]*len(vehicle), dtype=np.uint8)

    test_img = np.vstack((flower_test, vehicle_test))
    test_label = np.array([0]*len(flower_test) + [1]*len(vehicle_test), dtype=np.uint8)

    cls_names = ['flower', 'vehicle']
    print("{} train images and {} test images loaded."
            .format(len(train_img), len(test_img)))
    train_img = train_img / 255.0
    test_img = test_img / 255.0

    # build model
    model = tf.keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    # training
    model.fit(train_img, train_label, epochs=5)

    # Evaluate
    test_loss, test_acc = model.evaluate(test_img, test_label)
    print('Test accuracy:', test_acc)
if __name__ == "__main__":
    main()
