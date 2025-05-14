import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs):
        super(StreamlitCallback, self).__init__()
        self.num_epochs = num_epochs
        
        # Criar placeholders para métricas
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.metrics_container = st.empty()
        
        # Criar containers para os gráficos
        self.col1, self.col2 = st.columns(2)
        self.acc_chart = self.col1.empty()
        self.loss_chart = self.col2.empty()
        
        # Inicializar listas para armazenar métricas
        self.acc_history = []
        self.val_acc_history = []
        self.loss_history = []
        self.val_loss_history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.status_text.text(f"Época {epoch + 1}/{self.num_epochs}")
    
    def on_epoch_end(self, epoch, logs=None):
        # Atualizar progresso
        progress = (epoch + 1) / self.num_epochs
        self.progress_bar.progress(progress)
        
        # Atualizar métricas
        self.acc_history.append(logs.get('accuracy', 0))
        self.val_acc_history.append(logs.get('val_accuracy', 0))
        self.loss_history.append(logs.get('loss', 0))
        self.val_loss_history.append(logs.get('val_loss', 0))
        
        # Criar e atualizar gráficos
        fig_acc = plt.figure(figsize=(8, 4))
        plt.plot(self.acc_history, label='Treino')
        plt.plot(self.val_acc_history, label='Validação')
        plt.title('Acurácia do Modelo')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.grid(True)
        self.acc_chart.pyplot(fig_acc)
        plt.close(fig_acc)
        
        fig_loss = plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history, label='Treino')
        plt.plot(self.val_loss_history, label='Validação')
        plt.title('Perda do Modelo')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()
        plt.grid(True)
        self.loss_chart.pyplot(fig_loss)
        plt.close(fig_loss)
        
        # Mostrar métricas atuais
        metrics_text = f"""
        **Métricas da Época {epoch + 1}:**
        - Acurácia (Treino): {logs.get('accuracy', 0):.4f}
        - Acurácia (Validação): {logs.get('val_accuracy', 0):.4f}
        - Perda (Treino): {logs.get('loss', 0):.4f}
        - Perda (Validação): {logs.get('val_loss', 0):.4f}
        """
        self.metrics_container.markdown(metrics_text)
    
    def on_train_end(self, logs=None):
        self.status_text.text("Treinamento concluído!")
        st.success("Modelo treinado com sucesso!")

def create_data_generators(batch_size=32):
    """
    Cria geradores de dados para treinamento, validação e teste
    """
    # Aumentação de dados para o conjunto de treinamento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Apenas rescale para validação e teste
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    return train_datagen, val_datagen, test_datagen

def load_and_prepare_data(data_dir, batch_size=32, img_size=(224, 224)):
    """
    Carrega e prepara os dados para treinamento
    """
    train_datagen, val_datagen, test_datagen = create_data_generators()

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, val_generator, test_generator

def plot_training_history(history):
    """
    Plota o histórico de treinamento do modelo
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de acurácia
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Acurácia do Modelo')
    ax1.set_ylabel('Acurácia')
    ax1.set_xlabel('Época')
    ax1.legend(['Treino', 'Validação'])
    
    # Gráfico de perda
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Perda do Modelo')
    ax2.set_ylabel('Perda')
    ax2.set_xlabel('Época')
    ax2.legend(['Treino', 'Validação'])
    
    plt.tight_layout()
    return fig 