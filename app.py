import streamlit as st
import os
import numpy as np
from PIL import Image
from src.model import create_model, preprocess_image
from src.utils import load_and_prepare_data, StreamlitCallback
from src.validation import validate_xray_image

# Configuração da página
st.set_page_config(
    page_title="Classificador de Pneumonia - PUC Minas",
    page_icon="🫁",
    layout="wide"
)

# Título principal
st.title("Classificador de Pneumonia em Raios-X")
st.markdown("---")

# Sidebar para navegação
page = st.sidebar.selectbox(
    "Selecione uma página",
    ["Treinar Modelo", "Classificar Imagem"]
)

# Função para salvar imagem temporária
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("temp.jpg"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

if page == "Treinar Modelo":
    st.header("Treinamento do Modelo")
    
    # Parâmetros de treinamento com explicações
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input(
            "Número de épocas",
            min_value=1,
            value=10,
            help="Uma época representa uma passagem completa pelos dados de treinamento. "
                 "Mais épocas permitem que o modelo aprenda mais, mas podem causar overfitting se forem muitas."
        )
        batch_size = st.number_input(
            "Tamanho do batch",
            min_value=1,
            value=32,
            help="Número de imagens processadas antes de atualizar o modelo. "
                 "Batches maiores são mais eficientes mas consomem mais memória. "
                 "Valores comuns são 16, 32, 64."
        )
    
    with col2:
        validation_split = st.slider(
            "Divisão de validação",
            0.1,
            0.3,
            0.2,
            help="Porcentagem dos dados usada para validação durante o treinamento. "
                 "Ajuda a monitorar se o modelo está generalizando bem ou tendo overfitting."
        )
        
    if st.button("Iniciar Treinamento"):
        try:
            # Verificar se existem imagens para treinamento
            if not os.path.exists('data/train/NORMAL') or not os.path.exists('data/train/PNEUMONIA'):
                st.error("Dataset não encontrado. Execute 'make download-dataset' primeiro.")
                st.stop()
            
            # Carregar e preparar dados
            train_generator, val_generator, test_generator = load_and_prepare_data(
                'data',
                batch_size=batch_size
            )
            
            # Criar modelo
            model = create_model()
            
            # Criar callback do Streamlit
            streamlit_callback = StreamlitCallback(epochs)
            
            # Treinar modelo
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=[streamlit_callback]
            )
            
            # Salvar o modelo
            model.save('modelo_treinado.keras')
            
        except Exception as e:
            st.error(f"Erro durante o treinamento: {str(e)}")

elif page == "Classificar Imagem":
    st.header("Classificação de Imagens")
    
    uploaded_file = st.file_uploader("Escolha uma imagem de raio-x", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Exibir imagem
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem carregada', use_container_width=True)
        
        # Validar se é uma imagem de raio-X
        is_valid, error_message = validate_xray_image(image)
        
        if not is_valid:
            st.error(f"Imagem inválida: {error_message}")
            st.warning("Por favor, certifique-se de enviar uma imagem de raio-X válida.")
            st.stop()
        
        # Verificar se existe um modelo treinado
        if os.path.exists('modelo_treinado.keras'):
            if st.button("Classificar"):
                with st.spinner("Classificando..."):
                    # Salvar e preprocessar imagem
                    save_uploaded_file(uploaded_file)
                    img_array = preprocess_image("temp.jpg")
                    
                    # Carregar modelo e fazer predição
                    model = create_model()
                    model.load_weights('modelo_treinado.keras')
                    prediction = model.predict(img_array)
                    
                    # Exibir resultado
                    resultado = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
                    confianca = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
                    
                    # Criar colunas para resultado e confiança
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"Classificação: {resultado}")
                    
                    with col2:
                        st.info(f"Confiança: {confianca:.2%}")
                    
                    # Remover arquivo temporário
                    if os.path.exists("temp.jpg"):
                        os.remove("temp.jpg")
        else:
            st.warning("Nenhum modelo treinado encontrado. Por favor, treine o modelo primeiro.")

# Adicionar informações no rodapé
st.markdown("---")
st.markdown("Desenvolvido para classificação de imagens de raio-x pulmonar") 