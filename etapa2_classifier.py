import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers, models

#1. Cargar datos
print("Cargando datos de dataset Fashion MNIST necesarios para la clasificación")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

#Normalizacion
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

#2. Carga del encoder
nombre_archivo = 'encoder_final.h5'

if not os.path.exists(nombre_archivo):
    print(f"ERROR: No se encuentra el archivo'{nombre_archivo}'. Favor verificar el nombre en carpeta.")
    exit()

print(f"Cargando encoder desde {nombre_archivo}")
pretrained_encoder = models.load_model(nombre_archivo)

# 3. Congelando el enconder
pretrained_encoder.trainable = False
print("Pesos del enconder CONGELADOS (a no modificar durante el entrenamiento)")

# 4. Construcción de nuevo modelo clasificador
inputs = layers.Input(shape=(784,), name = 'input_clasification')

# se pasa la entrada por enconder congelado

x = pretrained_encoder(inputs, training= False)

# nuevas capas, que serian como la head de clasificación

x = layers.Dense(64, activation='relu', name='capa_densa_nueva')(x)
x = layers.Dropout(0.2)(x)#evita memorizar
outputs = layers.Dense(10, activation='softmax', name='salida_10_classes')(x)

model = models.Model(inputs, outputs, name = 'modelo_final_TLearning')
model.summary

#Entrenamiento\

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Iniciando entrenamiento de la etapa 2...")
history = model.fit(x_train, y_train,
                    epochs = 30,
                    batch_size=128,
                    validation_data =(x_test, y_test))

print("Trabajo completado")
