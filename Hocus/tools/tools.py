import numpy as np
import cPickle as pickle
import os
import gzip

def collage(data):
    if type(data) is not list:
        if data.shape[3] != 3:
            data = data.transpose(0, 2, 3, 1)

        images = [img for img in data]
    else:
        images = list(data)

    side = int(np.ceil(len(images)**0.5))
    for i in range(side**2 - len(images)):
        images.append(images[-1])
    collage = [np.concatenate(images[i::side], axis=0)
               for i in range(side)]
    collage = np.concatenate(collage, axis=1)
    #collage -= collage.min()
    #collage = collage / np.absolute(collage).max() * 256
    return collage

def mapLabelsOneHot(data, manual_class_number="no", class_number="24"):
    data = np.asarray(data)
    if manual_class_number == "yes":
        class_no = class_number
    else:
        class_no = int(data.max()+1)
    out = np.zeros((data.shape[0], class_no)).astype(np.float32)
    out[range(out.shape[0]), data.astype(int)] = 1
    return out

def readCIFAR(path):
    trnData = []
    trnLabels = []
    for i in range(1,6):
        with open(os.path.join(path,'data_batch_{}'.format(i))) as f:
            data = pickle.load(f)
            trnData.append(data['data'])
            trnLabels.append(data['labels'])

    trnData = np.concatenate(trnData).reshape(-1, 3, 32, 32)
    trnData = np.concatenate([trnData[:,:,:,::-1], trnData[:,:,:,:]])
    trnLabels = np.concatenate(trnLabels)
    trnLabels = np.concatenate([trnLabels, trnLabels])

    with open(os.path.join(path,'test_batch'.format(i))) as f:
        data = pickle.load(f)
        tstData = data['data']
        tstLabels = data['labels']

    tstData = tstData.reshape(-1, 3, 32, 32)
    tstData = np.concatenate([tstData[:,:,:,::-1], tstData[:,:,:,:]])
    tstLabels = np.concatenate([tstLabels, tstLabels])

    trnData = trnData.transpose(0, 2, 3, 1)
    tstData = tstData.transpose(0, 2, 3, 1)


    return trnData, tstData, trnLabels, tstLabels

def load_mnist(images_path, labels_path):
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
