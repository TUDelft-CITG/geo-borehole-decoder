# In[1]:

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import tensorflow as tf
from train_functions import load_image_resize, data_augment

#%%


## define the labels of the data


labels = ["bhole", "non_bhole"]

img_size = 224


# split the bhole and non_bhole images into train and test sets. Fetch train and test data,
train = load_image_resize("train", labels, img_size)
test = load_image_resize("test", labels, img_size)


x_train = data_augment(train, 255, img_size)[0]
y_train = data_augment(train, 255, img_size)[1]

x_test = data_augment(test, 255, img_size)[0]
y_test = data_augment(test, 255, img_size)[1]


## augument the train data by random rotations, zooming and flips
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
)


datagen.fit(x_train)

## define the model
## cnn with 1 convolutional layer followed by max pooling layers with a droput layer at the end
model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPool2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()


## compile the model using Adam optimizer and sparse categorical crossentropy as loss fn
opt = Adam(lr=0.000001)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

## train the model for 200 epochs
history = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))
# %%
## Model evaluations

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(200)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, accuracy, label="Training Accuracy")
plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

# %%
## predict on the test data
predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1, -1)[0]
print(
    classification_report(
        y_test, predictions, target_names=["bhole (Class 0)", "not_bhole (Class 1)"]
    )
)

# %%

# save model and architecture for the prediction of new data
model.save("model.h5")
print("Saved model to disk")
