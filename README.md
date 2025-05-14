# Classificador de Pneumonia em Raios-X - PUC Minas

Este projeto utiliza Deep Learning (CNN) para classificar imagens de raio-x de pulmão, identificando casos de pneumonia. A aplicação foi desenvolvida usando TensorFlow e Streamlit, oferecendo uma interface gráfica intuitiva para treinamento e classificação.

## Estrutura do Projeto
```
.
├── data/               # Diretório para armazenar os dados
│   ├── train/         # Imagens de treinamento
│   ├── val/           # Imagens de validação
│   └── test/          # Imagens de teste
├── src/               # Código fonte
│   ├── model.py       # Definição do modelo CNN
│   └── utils.py       # Funções auxiliares
├── kaggle/            # Diretório para credenciais do Kaggle
│   └── kaggle.json    # Arquivo de credenciais do Kaggle
├── app.py             # Aplicação Streamlit
├── pyproject.toml     # Configuração do Poetry
└── Makefile          # Comandos de automação
```

## Pré-requisitos

- Python 3.9 ou superior
- Poetry (será instalado automaticamente se não estiver presente)
- Conta no Kaggle para download do dataset

## Configuração do Kaggle

1. Crie uma conta no [Kaggle](https://www.kaggle.com) se ainda não tiver
2. Vá para sua conta: `Seu perfil -> Account -> Create New API Token`
3. Isso irá baixar um arquivo `kaggle.json`
4. Crie um diretório `kaggle` no projeto e mova o arquivo para lá:
```bash
mkdir kaggle
mv /caminho/do/download/kaggle.json kaggle/
```

## Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd chest-xray-classifier
```

2. Execute o setup inicial:
```bash
make setup
```
Este comando irá:
- Instalar o Poetry se necessário
- Criar um ambiente virtual
- Instalar todas as dependências
- Criar a estrutura de diretórios necessária

3. Baixe o dataset:
```bash
make download-dataset
```
Este comando irá baixar e organizar automaticamente o dataset de raios-X do Kaggle.

## Executando a Aplicação

Para iniciar a aplicação Streamlit:
```bash
make run
```

## Comandos Disponíveis

- `make setup` - Instala Poetry e dependências do projeto
- `make install` - Atualiza dependências do projeto
- `make clean` - Remove ambiente virtual e arquivos temporários
- `make run` - Executa a aplicação
- `make download-dataset` - Baixa e organiza o dataset de raio-x

## Funcionalidades

### Página de Treinamento
- Interface para configurar parâmetros de treinamento
- Visualização em tempo real do progresso
- Gráficos de acurácia e perda atualizados durante o treinamento
- Métricas detalhadas por época

### Página de Classificação
- Upload de imagens de raio-x
- Visualização da imagem carregada
- Classificação com indicador de confiança
- Resultado em formato amigável

## Dataset

O projeto utiliza o dataset "Chest X-Ray Images (Pneumonia)" do Kaggle, que contém:
- Imagens de raio-x de pulmão classificadas como normais e com pneumonia
- Aproximadamente 5,856 imagens no total
- Divisão em conjuntos de treino (5,216 imagens), validação (16 imagens) e teste (624 imagens)

## Contribuindo

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request 