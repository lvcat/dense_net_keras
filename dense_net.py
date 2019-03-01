
import tensorflow as tf
import get_data
import os
import h5py
import keras
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # ROOT = "/dataset/cifar-10-batches-py/"
# train_data,train_label,test_data,test_label=get_data.load_CIFAR10(ROOT)
list_train = '/home/m18_lkg/PycharmProjects/dense_1//hd5_list_train.txt'
list_val = '/home/m18_lkg/PycharmProjects/dense_1/hd5_list_val.txt'
batchsize = 100
growthrate = 12
epochs = 50

CLASS=10000

val_data = []
val_label = []

v = h5py.File("/home/m17_sxp/Documents/Tensorflow-implementation-of-LCNN-master/data_process/hd5/train_121.h5")
val_data.extend(np.float32(v['data']))
val_label.extend(np.int32(v['label']))
print(type(val_data))
v.close()

val_data = np.float32(val_data)
val_label_ = np.array(val_label)
val_data = val_data / 255
val_label = np.zeros([5000, CLASS])
for index in range(5000):
    val_label[index][val_label_[index]] = 1



f = h5py.File("/dataset/100k/train_0.h5")
train_data = np.float32(f['data'])
train_label_ = np.int32(f['label'])
f.close()


train_data = train_data.astype("float32")

train_data = train_data / 255

train_label = np.zeros([100000, CLASS])

for index in range(100000):
    train_label[index][train_label_[index]] = 1


def add_layer(data):
    c = keras.layers.BatchNormalization(epsilon=1e-4)(data)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2D(growthrate, 3, strides=1,
                            activation=tf.identity,
                            use_bias=False,
                            padding='same')(c)

    data = keras.layers.concatenate([c, data], axis=-1)

    return data


def transition(data):
    # shape = tf.shape(data)
    print(data.shape)

    in_channel = int(data.shape[3])
    print(in_channel)
    # bushi12

    data = keras.layers.BatchNormalization(epsilon=1e-4)(data)
    data = keras.layers.ReLU()(data)
    data = keras.layers.Conv2D(in_channel, 1, use_bias=False, activation=tf.nn.relu)(data)
    data = keras.layers.MaxPooling2D(2)(data)

    return data


def dense_net():
    image = keras.Input(shape=[144, 144, 3])
    data = keras.layers.Conv2D(16, 3, padding='same')(image)
    data = keras.layers.MaxPooling2D(2)(data)
    # with tf.variable_scope('block1') as scope:
    for i in range(6):
        data = add_layer(data)
    data = transition(data)
    print(data.shape)

    # with tf.variable_scope('block2') as scope:
    for i in range(12):
        data = add_layer(data)
    data = transition(data)
    print(data.shape)
    # with tf.variable_scope('block3') as scope:
    """for i in range(12):
        data = add_layer(data)
    print(data.shape)"""
    data = keras.layers.Conv2D(256, 3, padding='same')(data)
    data = keras.layers.BatchNormalization(epsilon=1e-4, )(data)
    data = keras.layers.Activation('relu')(data)
    data = keras.layers.MaxPooling2D(2)(data)
    print(data.shape)

    data = keras.layers.Conv2D(512, 1, padding='same')(data)
    data = keras.layers.BatchNormalization(epsilon=1e-4, )(data)
    data = keras.layers.Activation('relu')(data)
    data = keras.layers.MaxPooling2D(2)(data)
    print(data.shape)

    data = keras.layers.Conv2D(512, 2, padding='same')(data)
    data = keras.layers.BatchNormalization(epsilon=1e-4, )(data)
    data = keras.layers.Activation('relu')(data)
    data = keras.layers.GlobalAveragePooling2D()(data)
    print(data.shape)
    # data=keras.layers.Flatten()(data)
    logits = keras.layers.Dense(10000, activation='softmax')(data)
    print(logits)
    return keras.Model(inputs=image, outputs=logits)


adam = keras.optimizers.Adam(lr=0.01, beta_2=0.99)
model = dense_net()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
check_point_path="/home/m18_lkg/PycharmProjects/dense_1/weights/cp-hd_1.ckpt"
check_point_dir=os.path.dirname(check_point_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(check_point_path,verbose=1,save_weights_only=True,period=50)


datagen = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, horizontal_flip=True,
                                                           rotation_range=120, width_shift_range=0.3,
                                                           height_shift_range=0.3)
datagen.fit(train_data)

model.fit_generator(datagen.flow(train_data, train_label, batch_size=batchsize),steps_per_epoch=1000,
                        epochs=epochs, validation_data=(val_data,val_label),callbacks=[cp_callback],verbose=1, shuffle=True)
model.save("/home/m18_lkg/PycharmProjects/dense_1/dense_net.h5")
# scores = model.evaluate(test_data, test_label)
# print('test loss:', scores[0])
# print('test acc:', scores[1])