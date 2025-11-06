# 1. Imagem Base
# Começamos com uma imagem Python 3.10 leve.
FROM python:3.10-slim

# 2. Instalação de Dependências do Sistema
# Precisamos do Tesseract (para OCR) e suas dependências.
# 'tesseract-ocr-por' é o pacote de idioma Português que seu código usa.
# 'libgl1-mesa-glx' é necessário para o OpenCV (cv2) funcionar.
# Linha CORRIGIDA
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Define o Diretório de Trabalho
# Informa ao Docker que todo o resto acontecerá dentro da pasta /app
WORKDIR /app

# 4. Copia e Instala Dependências Python
COPY requirements.txt .

# Atualiza o PIP primeiro e DEPOIS instala os pacotes
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia os Arquivos da Aplicação
# Copia o resto dos seus arquivos (main.py, best.pt) para dentro da imagem
COPY . .

# 6. Expõe a Porta
# Informa ao Docker que o container vai escutar na porta 8501 (padrão do Streamlit)
EXPOSE 8501

# 7. Comando de Execução
# O comando para iniciar seu app quando o container rodar
CMD ["streamlit", "run", "main.py"]
