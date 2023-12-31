import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

def load_data(dataset, resize_shape):
    multilabel_datasets = ["voc"]
    bbox_datasets = ["cars196"]
    irregular_datasets = ["visual_domain_decathlon/aircraft"]
    small_datasets = irregular_datasets + ["cifar10", "cifar100"]
    big_datasets = multilabel_datasets + bbox_datasets + \
                   ["sun397", "food101", "dtd", "oxford_iiit_pet", "caltech101", "oxford_flowers102"]
    tf_datasets = small_datasets + big_datasets
    other_datasets = ["birdsnap"]
    
    # Datasets not available in tensorflow
    if dataset in other_datasets:
        # Load data
        x = np.load(f"other_datasets/{dataset}/{dataset}_images.npy")
        y = np.load(f"other_datasets/{dataset}/{dataset}_labels.npy") - 1  # To solve in .npy
        input_shape = x.shape[1:]
        # Get 5 samples of each class
        classes, counts = np.unique(y, return_counts=True)
        num_classes = classes.shape[0]
        intervals = np.insert(np.cumsum(counts), 0, 0)
        test_idx = []
        for i in range(num_classes):
            start, end = intervals[i], intervals[i+1]
            class_idx = np.random.choice(np.arange(start, end), size=5, replace=False)
            test_idx.extend(class_idx)
        test_idx = np.array(test_idx)
        # Separate train and test
        x_test, y_test = x[test_idx], y[test_idx]
        train_mask = np.ones(y.shape[0], dtype=bool)
        train_mask[test_idx] = False
        x_train, y_train = x[train_mask], y[train_mask]
        # Shuffle train data
        np.random.seed(196418)
        perm = np.random.permutation(y_train.shape[0])
        x_train, y_train = x_train[perm], y_train[perm]

    
    elif dataset in tf_datasets:
        def crop_resize(im_array):
            # Crop images to remove zero padding and resize them
            new_images = []
            for image in im_array:
                nonzero_rows, nonzero_cols = np.nonzero(image)[0], np.nonzero(image)[1]
                first_row, last_row = min(nonzero_rows), max(nonzero_rows)
                first_col, last_col = min(nonzero_cols), max(nonzero_cols)
                image = image[first_row:last_row+1, first_col:last_col+1]
                image = Image.fromarray(image)
                reshaped_content = np.asarray(image.resize(resize_shape))
                new_images.append(reshaped_content)
            return np.array(new_images)


        if dataset in small_datasets:
            # Load data
            ds, info = tfds.load(dataset, batch_size=-1, with_info=True)
            ds = tfds.as_numpy(ds)
            x_train, y_train = ds['train']['image'], ds['train']['label']
            if "test" in ds.keys():
                x_test, y_test = ds['test']['image'], ds['test']['label']
            else:
                x_test, y_test = ds['validation']['image'], ds['validation']['label']
            if dataset in ["visual_domain_decathlon/aircraft"]:
                x_train, x_test = crop_resize(x_train), crop_resize(x_test)

        elif dataset in big_datasets:
            def fetch(im_array):
                # Fetch images from dataset and resize them
                new_images, labels = [], []
                for i, batch in enumerate(im_array):
                    if dataset in multilabel_datasets:
                        image, label = batch["image"][0], batch["labels"]
                    else:
                        image, label = batch["image"][0], batch["label"][0]
                    if dataset in bbox_datasets:
                        y1, x1, y2, x2 = batch["bbox"][0]
                        image = image[int(y1 * image.shape[0]):int(y2 * image.shape[0]),
                                int(x1 * image.shape[1]):int(x2 * image.shape[1])]
                    image = Image.fromarray(image.numpy())
                    reshaped_content = np.asarray(image.resize(resize_shape))
                    new_images.append(reshaped_content)
                    labels.append(label)
                if dataset not in ["voc"]:
                    labels = np.array(labels)
                return np.array(new_images), labels
            
            # Load data
            ds, info = tfds.load(dataset, batch_size=1, with_info=True)
            if "test" in ds.keys():
                train, test = ds["train"], ds["test"]
            else:
                train, test = ds["train"], ds["validation"]
            # Fetch and resize
            x_train, y_train = fetch(train)
            x_test, y_test = fetch(test)

        # Get info
        if dataset in ["voc"]:
            num_classes = info.features['labels'].num_classes
        else:
            num_classes = info.features['label'].num_classes
        input_shape = info.features['image'].shape
    
    # Target one-hot vectorization
    if dataset in ["voc"]:
        y_train = [keras.utils.to_categorical(label, num_classes)[0] for label in y_train]
        y_train = [label.sum(axis=0) if len(label.shape) == 2 else label for label in y_train]
        y_test = [keras.utils.to_categorical(label, num_classes)[0] for label in y_test]
        y_test = [label.sum(axis=0) if len(label.shape) == 2 else label for label in y_test]
        y_train, y_test = np.array(y_train), np.array(y_test)
    else:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    # Normalize data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    return (x_train, y_train), (x_test, y_test), num_classes


