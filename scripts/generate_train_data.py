import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Model
from codecarbon import EmissionsTracker
import os
from load_data import load_data

# Parameters
batch_size = 32
epochs = 30
step = 6
optimizer = 'adam'
input_shape = (32, 32, 3)

# Define datasets
multi_label_datasets = ['voc']
single_label_datasets = ['visual_domain_decathlon/aircraft', 'cifar10', 'cifar100', 'cars196', 'sun397', 'food101', 'dtd', 'oxford_iiit_pet', 'caltech101', 'oxford_flowers102', 'birdsnap']
all_datasets = multi_label_datasets + single_label_datasets
all_datasets = ["cifar10", "mnist"]

# Get experiment identifier by the last log file created
id_exp = [path for path in os.listdir() if path[:12] == "batch_train."]
id_exp = max([int(path.split('.')[1]) for path in id_exp])
# Create output directory
directory = f"data_{id_exp}"
os.makedirs(directory)
os.makedirs(os.path.join(directory, "models"))
with open(f"{directory}/history_{id_exp}.csv", "w") as f:
    f.write("dataset,mode,intervention,epoch,loss,accuracy,precision,recall,val_loss,val_accuracy,val_precision,val_recall\n")
print("Saving results in", directory)


def get_model(num_classes, multilabel=False, input_shape=input_shape, optimizer=optimizer):
    """Return a compiled VGG16 model pretrained on Imagenet with a new dense block"""
    
    # Import pre-trained layers and freeze them
    base_model = VGG16(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    # Establish new fully connected block
    x = base_model.output
    x = Flatten()(x)  # flatten from convolution tensor output
    x = Dense(4096, activation='relu')(x) # number of layers and units are hyperparameters, as usual
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Declare output layer and compile according to type of label
    if multilabel:
        predictions = Dense(num_classes, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    return model

def train(model, current, total, step, multilabel=False):
    """Create a training tree of a model with freezing and quantization as cloning options"""
    
    if current >= total:
        return
    print(f"Training from {current} to {total} with steps of {step}")
    loss = "binary_crossentropy" if multilabel else "categorical_crossentropy"

    # FREEZE
    
    # Clone and freeze model
    model_freeze = keras.models.clone_model(model)
    model_freeze.set_weights(model.get_weights())
    for layer in model_freeze.layers[:-1]:
        layer.trainable = False
    model_freeze.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall'])
    # Callback to save model weights
    cp_path = os.path.join(directory, "models", dataset, f"{dataset}_freeze_{current}_{id_exp}.ckpt")
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=cp_path, save_weights_only=True, verbose=1)

    # Train until end
    for i in range(current, total):
        #print(directory, id_exp, f"{dataset}_freeze_{current}_{i+1}", f"emissions_{id_exp}.csv", os.path.exists(directory))
        tracker = EmissionsTracker(
            #project_name=f"{dataset}_freeze_{current}_{i+1}",
            #output_dir=directory,
            #output_file=f"emissions_{id_exp}.csv"
        )
        tracker.start()
        history = model_freeze.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs = 1,
            verbose = 0,
            validation_data = (x_test, y_test),
            callbacks=[cp_callback]
        )
        tracker.stop()
        # Write results
        h = history.history
        history_data = [dataset,"freeze", str(current), str(i+1), *[str(metric[0]) for metric in h.values()]]
        with open(f"{directory}/history_{id_exp}.csv", "a") as f:
            f.write(','.join(history_data) + '\n')
    
    # QUANTIZATION
    
    # Change precision policy and clone model
    keras.mixed_precision.set_global_policy("float16")
    model_quant = keras.models.clone_model(model)
    model_quant.set_weights(model.get_weights())
    model_quant.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall'])
    # Callback to save model weights
    cp_path = os.path.join(directory, "models", dataset, f"{dataset}_quant_{current}_{id_exp}.ckpt")
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=cp_path, save_weights_only=True, verbose=1)

    # Train until the end
    for i in range(current, current+step):
        tracker = EmissionsTracker(
            #project_name=f"{dataset}_quant_{current}_{i+1}",
            #output_dir=directory,
            #output_file=f"emissions_{id_exp}.csv"
        )
        tracker.start()
        history = model_quant.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs = 1,
            verbose = 0,
            validation_data = (x_test, y_test),
            callbacks=[cp_callback]
        )
        tracker.stop()
        # Write results
        h = history.history
        history_data = [dataset,"quant", str(current), str(i+1), *[str(metric[0]) for metric in h.values()]]
        with open(f"{directory}/history_{id_exp}.csv", "a") as f:
            f.write(','.join(history_data) + '\n')
     
    # Reestablish precision policy
    keras.mixed_precision.set_global_policy("float32")

    # BASE
    
    # Callback to save model weights
    cp_path = os.path.join(directory, "models", dataset, f"{dataset}_base_{current+step}_{id_exp}.ckpt")
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=cp_path, save_weights_only=True, verbose=1)
    
    # Train for one step
    for i in range(current, current+step):
        tracker = EmissionsTracker(
            #project_name=f"{dataset}_base_{i+1}",
            #output_dir=directory,
            #output_file=f"emissions_{id_exp}.csv"
        )
        tracker.start()
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs = 1,
            verbose = 0,
            validation_data = (x_test, y_test),
            callbacks=[cp_callback]
        )
        tracker.stop()
        # Write results
        h = history.history
        history_data = [dataset,"base", "0", str(i+1),
            *[str(metric[0]) for metric in h.values()]]
        with open(f"{directory}/history_{id_exp}.csv", "a") as f:
            f.write(','.join(history_data) + '\n')

    # Advance one step and continue
    current += step
    train(model, current, total, step)


for dataset in all_datasets:
    print(f"Testing dataset {dataset}")
    os.makedirs(os.path.join(directory, "models", dataset))
    
    multilabel = dataset in multi_label_datasets
    (x_train, y_train), (x_test, y_test), num_classes = load_data(dataset, input_shape[:-1])
    model = get_model(num_classes, multilabel=multilabel)

    train(model, 0, epochs, step)

