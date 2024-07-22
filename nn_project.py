import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import image_dataset_from_directory

#direktorijum sa 20 poddirektorijuma koji predstavljaju klase
parent_dir = './cats20/'


##### PRIKAZ ORIGINALNIH SLIKA I BROJA ODBIRAKA #####
#lista poddirektorijuma (imena su ista kao imena klasa)
subdirectories = os.listdir(parent_dir)
#inicijalizacija niza koji ce sadrzati brojeve odbiraka za sve klase
image_counts = []

N = 20

plt.figure(figsize=(20, 20))
for i in range(N):
    #ukupna putanja poddirektorijuma
    subdir_path = os.path.join(parent_dir, subdirectories[i])
    
    #lista imena svih slika u klasi i prebrojavanje koliko ih je
    image_files = os.listdir(subdir_path)
    image_counts.append(len(image_files))
    
    #ukupna putanja prve slike
    image_path = os.path.join(subdir_path, image_files[0])
    
    #prikaz na subplotu
    img = mpimg.imread(image_path)
    plt.subplot(5, int(N/5), i+1)
    plt.imshow(img)
    plt.title(subdirectories[i], fontsize = 30)
    plt.axis('off')
    
	
#prikaz histograma
plt.figure(figsize=(9, 4))
plt.bar(subdirectories, image_counts)
plt.xticks(rotation='vertical')
plt.title('Broj odbiraka svake klase')
plt.xlabel('Klasa')
plt.ylabel('Broj odbiraka')
plt.show()


main_path = './cats20/'
img_size = (128, 128)
batch_size = 128

from keras.utils import image_dataset_from_directory
Xtrain = image_dataset_from_directory(main_path, 
                                      subset='training', 
                                      validation_split=0.3,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)

Xv = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.3,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

val_batches = tf.data.experimental.cardinality(Xv)
Xval = Xv.take((val_batches) // 2)
Xtest = Xv.skip((val_batches) // 2)

classes = Xtrain.class_names
print(classes)


from keras import layers
from keras import Sequential

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_size[0], 
                                                 img_size[1], 3)),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
  ]
)

N = 10

plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')
plt.show()


from keras import Sequential
from keras import layers
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy  
from keras.callbacks import EarlyStopping

num_classes = len(classes)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(Adam(learning_rate=0.001), 
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
history = model.fit(Xtrain,
                    epochs=100,
                    validation_data=Xval,
                    callbacks=[es],
                    verbose=0)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()


labels = np.array([])
pred = np.array([])
for img, lab in Xtest:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))    


from sklearn.metrics import accuracy_score
print('Tačnost modela na test skupu je: ' + str(100*accuracy_score(labels, pred)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(100, 100)) 
cmDisplay.plot(ax=ax)
plt.show()


correctly_classified_imgs = []
incorrectly_classified_imgs = []

for img, lab in Xval:
    predictions = model.predict(img, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    for i in range(len(lab)):
        if predicted_labels[i] == lab[i]:
            correctly_classified_imgs.append((img[i], lab[i], predicted_labels[i]))
        else:
            incorrectly_classified_imgs.append((img[i], lab[i], predicted_labels[i]))

plt.figure(figsize=(10, 10))
plt.suptitle('Dobro i loše klasifikovane slike', fontsize=16)
for i in range(0,2):
    plt.subplot(2, 2, i+1)
    plt.imshow(correctly_classified_imgs[i][0].numpy().astype('uint8'))
    plt.title(f'True: {classes[correctly_classified_imgs[i][1]]}, Predicted: {classes[correctly_classified_imgs[i][2]]}')
    plt.axis('off')
    plt.subplot(2, 2, i+3)
    plt.imshow(incorrectly_classified_imgs[i][0].numpy().astype('uint8'))
    plt.title(f'True: {classes[incorrectly_classified_imgs[i][1]]}, Predicted: {classes[incorrectly_classified_imgs[i][2]]}')
    plt.axis('off')
plt.show()



