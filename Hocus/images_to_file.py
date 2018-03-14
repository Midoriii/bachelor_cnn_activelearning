import h5py
import numpy as np
import glob
import cv2
from random import shuffle

# Path to resulting file
hdf5_path = 'vgg2f/vgg2f.hdf5'

addrs = []
labels = []

# Paths to data
addr_path_template = 'vgg2faces/n0000'
for i in range(1,78):
    # Get the image addresses for preprocessing
    img_addrs = glob.glob(addr_path_template + str(i) + '/*.jpg')
    addrs.extend(img_addrs)
    # Create labels for data
    img_labels = [i-1] * len(img_addrs)
    labels.extend(img_labels)

# Print to check correct sizes
print(len(addrs))
print(len(labels))

# Shuffle the data
c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)

# Divide data into training and test set
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]

tst_addrs = addrs[int(0.8*len(addrs)):]
tst_labels = labels[int(0.8*len(labels)):]

print(len(train_addrs))
print(len(tst_addrs))

# Array shapes for tensorflow
train_shape = (len(train_addrs), 64, 64, 3)
tst_shape = (len(tst_addrs), 64, 64, 3)

# Open hdf5 file and create arrays
hdf5_file = h5py.File(hdf5_path, mode='w')

# Datasets for images
hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("tst_img", tst_shape, np.int8)
# Datasets for labels
hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("tst_labels", (len(tst_addrs),), np.int8)
hdf5_file["tst_labels"][...] = tst_labels

# loop over train addresses
for i in range(len(train_addrs)):
    # Read an image and resize to (64, 64)
    # Cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    print(addr)
    #print(i)
    #print(addr.encode("utf-8"))
    img = cv2.imread(addr)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Save to the file
    hdf5_file["train_img"][i, ...] = img[None]

# loop over test addresses
for i in range(len(tst_addrs)):
    addr = tst_addrs[i]
    print(addr)
    #print(i)
    #print(addr.encode("utf-8"))
    img = cv2.imread(addr)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hdf5_file["tst_img"][i, ...] = img[None]

# Close the file
hdf5_file.close()
