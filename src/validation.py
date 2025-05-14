import numpy as np
from PIL import Image

def is_grayscale_like(img_array):
    """
    Verifica se a imagem tem características de tons de cinza,
    permitindo pequenas variações de cor
    """
    # Converte para array numpy se for imagem PIL
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # Se a imagem já é grayscale, retorna True
    if len(img_array.shape) == 2:
        return True
    
    # Para imagens RGB/RGBA
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Calcula a diferença máxima entre os canais
    max_diff = np.max([np.abs(r - g), np.abs(r - b), np.abs(g - b)])
    
    # Define um limiar de tolerância (ajuste conforme necessário)
    threshold = 30
    
    return max_diff <= threshold

def check_image_stats(img_array):
    """
    Verifica estatísticas básicas da imagem que são comuns em raios-X
    """
    # Converte para array numpy se for imagem PIL
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # Converte para grayscale se for RGB/RGBA
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    
    # Normaliza os valores
    img_array = img_array / 255.0
    
    # Calcula estatísticas
    mean_val = np.mean(img_array)
    std_val = np.std(img_array)
    
    # Valores típicos para raios-X (ajuste conforme necessário)
    return (0.2 <= mean_val <= 0.8) and (0.1 <= std_val <= 0.4)

def validate_xray_image(image):
    """
    Função principal de validação que combina diferentes verificações
    
    Args:
        image: PIL.Image ou caminho para a imagem
        
    Returns:
        tuple: (bool, str) - (é válido, mensagem de erro ou None)
    """
    try:
        # Se for caminho, carrega a imagem
        if isinstance(image, str):
            image = Image.open(image)
        
        # Converte para RGB se for outro modo
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Converte para array numpy
        img_array = np.array(image)
        
        # Verifica dimensões mínimas
        if img_array.shape[0] < 200 or img_array.shape[1] < 200:
            return False, "A imagem é muito pequena para ser um raio-X válido"
        
        # Verifica se tem características de tons de cinza
        if not is_grayscale_like(img_array):
            return False, "A imagem não possui características típicas de um raio-X (tons de cinza)"
        
        # Verifica estatísticas da imagem
        if not check_image_stats(img_array):
            return False, "A imagem não possui distribuição de intensidade típica de um raio-X"
        
        return True, None
        
    except Exception as e:
        return False, f"Erro ao validar imagem: {str(e)}" 