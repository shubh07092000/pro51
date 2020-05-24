from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
model=Sequential()
model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       input_shape=(240,240,3)
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=16, 
                        kernel_size=(2,2), 
                        activation='relu',
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
from keras.optimizers import Adam
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train = train_datagen.flow_from_directory(
        '/root/final/pro/ML/train/',
        target_size=(240, 240),
        batch_size=32
        )
test = test_datagen.flow_from_directory(
        '/root/final/pro/ML/test/',
        target_size=(240, 240),
        batch_size=32
        )
model.fit(
        train,
        steps_per_epoch=200,
        epochs=5,
        validation_data=test,
        validation_steps=100)
