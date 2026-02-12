import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models

# 1. Carga de los datos del dataset Fashion MNIST
# Se utiliza keras.datasets en lugar de pd.read_csv

print("Iniciando carga de dataset Fashion MNIST...")
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

# 2. Preprocesamiento

x_train  = x_train.astype('float32')/255.
x_test   = x_test.astype('float32')/255.

# aplanado de imágenes de 28x28 a vector de 784 

x_train = x_train.reshape((len(x_train),784))
x_test = x_test.reshape((len(x_test),784))

# 3. Diseño del modelo

latent_dim = 32# dimension del espacio latente


# --- Encoder ---

encoder_input = layers.Input(shape=(784,), name='encoder_input')
x = layers.Dense(128, activation='relu', name = 'encoder_dense_1')(encoder_input)
x = layers.Dense(64, activation='relu', name = 'encoder_dense_2')(x)
latent = layers.Dense(latent_dim, activation='relu', name = 'espacio_latente')(x)

encoder = models.Model(encoder_input, latent, name='encoder')
encoder.summary()

#----Decoder----
decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
x = layers.Dense(64, activation='relu', name = 'decoder_dense_1')(decoder_input)
x = layers.Dense(128, activation='relu', name = 'decoder_dense_2')(x)
decoder_output = layers. Dense(784, activation='sigmoid', name = 'decoder_output')(x)

decoder = models.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

#--Autoencoder Completo--

autoencoder_input = layers.Input(shape=(784,), name='autoencoder_input')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)

autoencoder = models.Model(autoencoder_input, decoded_img, name= 'autoencoder')

# 4. Compilar y entrenar...

autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['mse'])

print("Iniciando entrenamiento de autoencoder")

autoencoder.fit(x_train, x_train,
                epochs =30,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

#5. MLOps: guardar solo el encoder para la etapa 2
encoder.save('encoder_final.h5')
print("Enconder guardado como 'encoder_final.h5'")