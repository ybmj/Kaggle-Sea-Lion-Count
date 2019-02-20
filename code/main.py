import numpy as np

import os
from functions import data_process, rmse
import tensorflow as tf

from multiprocessing import Process, Manager, Pool


def add(trainX, trainY, path1, path2):
    tempX, tempY = data_process(path1, path2)
    trainX.extend(tempX)
    trainY.extend(tempY)
    del tempX, tempY


if __name__ == '__main__':

    r = 0.4     #scale down
    width = 100 #patch size
    batch_size = 100 # batch size
    root = '../input/'
    model_path = '../model'
    filenames = os.listdir(os.path.join(root, 'TrainDotted'))
    print(filenames)
    pool = Pool(3)
    manager = Manager()
    trainX = manager.list()
    trainY = manager.list()
    for filename in filenames:
        path1 = os.path.join(root, 'TrainDotted', filename)
        path2 = os.path.join(root, 'Train', filename)
        p = pool.apply_async(add, (trainX, trainY, path1, path2))
    pool.close()
    pool.join()
    print("data process end")
    print(np.array(trainX).shape)

    import cv2
    import matplotlib.pyplot as plt
    import skimage.feature
    import keras
    from keras.models import Sequential
    from keras import applications, Model
    from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU

    import keras.backend.tensorflow_backend as KTF
    KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


    # np.random.seed(1004)
    # randomize = np.arange(len(trainX))
    # print(randomize)
    # np.random.shuffle(randomize)
    # print(randomize)
    # trainX = trainX[randomize]
    # trainY = trainY[randomize]

    # n_train = int(len(trainX) * 0.7)
    # testX = trainX[n_train:]
    # testY = trainY[n_train:]
    # trainX = trainX[:n_train]
    # trainY = trainY[:n_train]
    #
    #
    # print(trainY.shape, trainY[0])
    # print(testY.shape, testY[0])

    # initial_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(100,100,3))
    # last = initial_model.output
    # x = Flatten()(last)
    # x = Dense(1024)(x)
    # x = LeakyReLU(alpha=.1)(x)
    # preds = Dense(5, activation='linear')(x)
    # model = Model(initial_model.input, preds)
    # 载入数据
    # model_name = 'my_model_weights.h5'
    # model.load_weights(model_name)
    #
    # optim = keras.optimizers.SGD(lr=1e-5, momentum=0.2)
    # model.compile(loss='mean_squared_error', optimizer=optim)
    # step1_epochs = 8
    # model.fit(trainX, trainY, epochs=step1_epochs, verbose=2, batch_size=batch_size)
    # model_name = '%s/step1-%s.h5' % (model_path,step1_epochs)
    # model.save(model_name)

    # optim = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
    # model.compile(loss='mean_squared_error', optimizer=optim)
    # step2_epochs = 30
    # model.fit(trainX, trainY, epochs=step2_epochs, verbose=2, batch_size=batch_size)
    # model_name = '%s/step2-%s.h5' % (model_path,step2_epochs)
    # model.save(model_name)
    # step3_epochs = 20
    # model.fit(trainX, trainY, epochs=step3_epochs, verbose=2, batch_size=batch_size)
    # model_name = '%s/step3-%s.h5' % (model_path,step3_epochs)
    # model.save(model_name)

    # result = model.predict(trainX)
    # print('Training set --')
    # print('    ground truth: ', np.sum(trainY, axis=0))
    # print('  evaluate count: ', np.sum(result*(result>0.3), axis=0).astype('int'))
    #
    # result = model.predict(testX)
    # print('Testing set --')
    # print('    ground truth: ', np.sum(testY, axis=0))
    # print('   predict count: ', np.sum(result*(result>0.3), axis=0).astype('int'))

    # 保存数据, 建议每个fit后面都保存一下
    # model_name = '%s/my_model_weights.h5' % model_path
    # model.save_weights(model_name)

