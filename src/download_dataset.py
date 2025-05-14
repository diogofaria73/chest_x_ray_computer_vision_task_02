import os
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """
    Configura as credenciais do Kaggle a partir do arquivo local
    """
    # Criar diretório .kaggle se não existir
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Copiar arquivo de credenciais
    cred_source = Path('kaggle/kaggle.json')
    cred_dest = kaggle_dir / 'kaggle.json'
    
    if not cred_source.exists():
        raise FileNotFoundError("Arquivo de credenciais não encontrado em kaggle/kaggle.json")
    
    shutil.copy2(cred_source, cred_dest)
    os.chmod(cred_dest, 0o600)  # Permissões corretas requeridas pelo Kaggle

def download_dataset():
    """
    Baixa o dataset Chest X-Ray Images (Pneumonia) do Kaggle
    e organiza nas pastas corretas.
    """
    try:
        setup_kaggle_credentials()
        # Importar kaggle após configurar as credenciais
        import kaggle
    except Exception as e:
        print(f"Erro ao configurar credenciais: {e}")
        return
    
    print("Baixando dataset do Kaggle...")
    
    # Configurar diretórios
    dataset_path = Path("chest_xray_data")
    data_path = Path("data")
    
    try:
        # Baixar dataset
        kaggle.api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path=dataset_path,
            unzip=True
        )
        
        # Organizar arquivos
        for split in ['train', 'val', 'test']:
            for label in ['NORMAL', 'PNEUMONIA']:
                # Criar diretórios de destino
                dest_dir = data_path / split / label
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Copiar arquivos
                src_dir = dataset_path / 'chest_xray' / split / label
                if src_dir.exists():
                    for img in src_dir.glob('*.jpeg'):
                        shutil.copy2(img, dest_dir)
        
        # Limpar arquivos temporários
        shutil.rmtree(dataset_path)
        print("Dataset baixado e organizado com sucesso!")
        
    except Exception as e:
        print(f"Erro ao baixar ou organizar o dataset: {e}")

if __name__ == "__main__":
    download_dataset() 