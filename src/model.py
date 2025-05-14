import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(224, 224, 3)):
    """
    Cria um modelo CNN para classificação de imagens de raio-x
    """
    model = models.Sequential([
        # Primeira camada de convolução
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda camada de convolução
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Terceira camada de convolução
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Camadas densas
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Saída binária (NORMAL vs PNEUMONIA)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_model(model_path):
    """
    Carrega um modelo salvo
    """
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Pré-processa uma imagem para inferência
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=target_size,
        color_mode='rgb'
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0  # Normalização 