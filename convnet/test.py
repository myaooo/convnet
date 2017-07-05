from keras.preprocessing.image import ImageDataGenerator
from convnet.data.preprocess import prep_data

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

datasets = prep_data(0.2)
train_data_generator = datagen.flow(
        datasets[0],  # this is the target directory
        datasets[1],
        # target_size=(64, 64),  # all images will be resized to 150x150
        batch_size=32)

# i = 0
for i in range(10):
    a = train_data_generator.__next__()

print(train_data_generator)
