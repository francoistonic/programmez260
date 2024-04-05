#############################################################################
#                                                                           #
# RECONNAISSANCE DE CHATS & CHIENS PAR TRANSFERT LEARNING => Apprentissage  #
#       (dataset : https://www.kaggle.com/chetankv/dogs-cats-images)        #
#                                                                           #
#############################################################################

# Importation du module TensorFlow
import tensorflow as tf

# Références pour les fichiers images de 'training' et de 'validation'
path_train ='./training_set/'
path_valid ='./test_set/'

# Paramètres pour l'apprentissage
TAILLE_IMAGE = (299,299)
TAILLE_BATCH = 20
NB_CYCLES    = 10
# Réglage pour optimiser l'utilisation du CPU/GPU et de la mémoire
AUTOTUNE = tf.data.AUTOTUNE

# Chargement du réseau de base pour le transfert learning
InceptionV3_base = tf.keras.applications.InceptionV3(
  weights='imagenet',
  include_top=False,
  input_shape=TAILLE_IMAGE+(3,))
# Paramètres non-modifiables par apprentissage dans le réseau de base
InceptionV3_base.trainable=False

# Création du réseau complet avec les couches de classification
MonReseau = tf.keras.models.Sequential()
MonReseau.add(InceptionV3_base)
MonReseau.add(tf.keras.layers.GlobalAveragePooling2D())
MonReseau.add(tf.keras.layers.Dropout(0.25))
MonReseau.add(tf.keras.layers.Dense(128, activation='relu'))
MonReseau.add(tf.keras.layers.Dense(64, activation='relu'))
MonReseau.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Affichage de la description du réseau
MonReseau.summary()

# Création du Dataset pour l'apprentissage à partir des fichiers d'images
train_ds = tf.keras.utils.image_dataset_from_directory(
  directory=path_train,      # répertoire des fichiers
  label_mode='binary',       # classification en 2 catégories
  color_mode='rgb',          # images au format RGB
  batch_size=TAILLE_BATCH,   # taille des lots d'images
  image_size=TAILLE_IMAGE,   # taille pour redimensionner les images
  crop_to_aspect_ratio=False)# redimensionnement sans conservation du ratio

# Création du Dataset pour la validation à partir des fichiers d'images
valid_ds = tf.keras.utils.image_dataset_from_directory(
  directory=path_valid,      # répertoire des fichiers
  label_mode='binary',       # classification en 2 catégories
  color_mode='rgb',          # images au format RGB
  batch_size=TAILLE_BATCH,   # taille des lots d'images
  image_size=TAILLE_IMAGE,   # taille pour redimensionner les images
  crop_to_aspect_ratio=False)# redimensionnement sans conservation du ratio

# Normalisation des valeurs de pixels en réels sur [0,1] et optimisation
train_ds = train_ds.map(lambda x,y:(x/255,y),num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE).cache()
valid_ds = valid_ds.map(lambda x,y:(x/255,y),num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE).cache()

# Définition des paramètres pour l'apprentissage
MonReseau.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

# Apprentissage avec les données des Datasets
MonReseau.fit(x=train_ds, epochs=NB_CYCLES, validation_data=valid_ds)

# Sauvegarde du réseau après apprentissage
tf.keras.models.save_model(MonReseau, "MonReseau.h5")

