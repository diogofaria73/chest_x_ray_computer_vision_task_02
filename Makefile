.PHONY: setup install clean run help download-dataset

help:
	@echo "Comandos disponíveis:"
	@echo "  make setup           - Instala Poetry e dependências do projeto"
	@echo "  make install         - Atualiza dependências do projeto"
	@echo "  make clean           - Remove ambiente virtual e arquivos temporários"
	@echo "  make run            - Executa a aplicação"
	@echo "  make download-dataset - Baixa e organiza o dataset de raio-x"

setup:
	@echo "Verificando se o Poetry está instalado..."
	@if ! command -v poetry &> /dev/null; then \
		echo "Instalando Poetry..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
	fi
	@echo "Configurando Poetry..."
	poetry config virtualenvs.in-project true
	@echo "Removendo ambiente virtual existente (se houver)..."
	rm -rf .venv
	@echo "Criando novo ambiente virtual..."
	poetry env use python3
	@echo "Instalando dependências..."
	poetry install --no-root
	@echo "Criando estrutura de diretórios..."
	mkdir -p data/train/NORMAL data/train/PNEUMONIA \
		data/val/NORMAL data/val/PNEUMONIA \
		data/test/NORMAL data/test/PNEUMONIA
	@echo "\nSetup concluído! Use 'make download-dataset' para baixar o dataset e 'make run' para executar a aplicação."

install:
	@echo "Atualizando dependências..."
	poetry update

clean:
	@echo "Limpando ambiente..."
	rm -rf .venv
	rm -f poetry.lock
	rm -f temp.jpg
	rm -f modelo_treinado.keras
	rm -rf data/*/NORMAL/* data/*/PNEUMONIA/*
	@echo "Ambiente limpo!"

run:
	@if [ ! -d ".venv" ]; then \
		echo "Ambiente virtual não encontrado. Execute 'make setup' primeiro."; \
		exit 1; \
	fi
	@echo "Iniciando aplicação..."
	poetry run streamlit run app.py

download-dataset:
	@if [ ! -f "kaggle/kaggle.json" ]; then \
		echo "Arquivo de credenciais do Kaggle não encontrado em kaggle/kaggle.json"; \
		echo "Por favor, certifique-se de que:"; \
		echo "1. Você tem uma conta no Kaggle (https://www.kaggle.com)"; \
		echo "2. Você baixou seu arquivo kaggle.json do Kaggle"; \
		echo "3. O arquivo está no diretório kaggle/kaggle.json do projeto"; \
		exit 1; \
	fi
	@echo "Baixando dataset..."
	poetry run python src/download_dataset.py

# Regra padrão
.DEFAULT_GOAL := help 