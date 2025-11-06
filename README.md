# Sistema Inteligente de Análise Forense de Documentos

**Autor:** Artur Luiz Rizzato Toru Soda

**Instituição:** Universidade Federal de Santa Catarina (UFSC)

**Disciplina:** INE5448 — Inteligência Artificial e Segurança

**Orientação:** Wellington Fernandes Silvano

---

Este projeto é uma aplicação web construída com Streamlit que implementa um sistema de análise forense de documentos com uma arquitetura híbrida, focada em segurança e privacidade.

A análise é dividida em duas etapas:
1. IA Local (Proteção de PII): Um modelo YOLOv8 (best.pt) é executado localmente na máquina do usuário para detectar e redigir (censurar) Informações de Identificação Pessoal (PII) da imagem do documento.
2. IA em Nuvem (Análise Forense): Somente a imagem "segura" (redigida) e metadados não-sensíveis são enviados para a API do Google Gemini. A IA em nuvem então realiza uma análise forense detalhada para detectar sinais de fraude, adulteração, inconsistências de compressão, etc.

---

## 🚀 Funcionalidades Principais

- **Arquitetura Híbrida:** Garante que dados sensíveis (fotos, nomes, CPFs) nunca saiam da máquina do usuário.
- **Detecção Local de PII:** Usa um modelo YOLOv8 customizado para identificar PII em documentos.
- **Revisão Humana (Human-in-the-Loop):** O usuário deve revisar e aprovar as detecções de PII antes que qualquer censura seja aplicada.
- **Análise Forense Avançada:** Utiliza o poder do Google Gemini para analisar a imagem redigida em busca de anomalias de pixels, padrões de compressão, clonagem de texturas e outros artefatos de adulteração.
- **Extração de Features:** O sistema extrai metadados técnicos (nível de foco, variância de cor, etc.) da imagem original e os envia para a nuvem como contexto adicional para a análise forense.
- **Portabilidade com Docker:** O projeto é 100% "dockerizado", permitindo que qualquer pessoa o execute com dois comandos, sem se preocupar com a instalação de Python, Tesseract, OpenCV ou outras dependências.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **IA Local (PII):** YOLOv8 (ultralytics)
- **IA em Nuvem (Forense):** Google Gemini
- **Processamento de Imagem:** OpenCV, Pillow
- **OCR (Para Revisão de PII):** Tesseract
- **Portabilidade:** Docker & Docker Compose

---

## ▶️ Como Executar (Plug and Play)

Este projeto foi desenhado para ser executado em qualquer máquina que tenha o Docker instalado, resolvendo todos os problemas de dependência.

- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (geralmente incluído no Docker Desktop)


### Passo 1: Clone o Repositório

```bash
git clone [https://github.com/ArturRSoda/Analise_Forense_de_Documento_Inteligente.git](https://github.com/ArturRSoda/Analise_Forense_de_Documento_Inteligente.git)
cd Analise_Forense_de_Documento_Inteligente
```


### Passo 2: Configure sua Chave de API

O aplicativo precisa de uma chave da API do Google Gemini para funcionar.
1. Crie um arquivo chamado .env na raiz do projeto.
2. Abra este arquivo e adicione sua chave de API da seguinte forma:

```bash
GOOGLE_API_KEY=SUA_CHAVE_API_VEM_AQUI
```

O arquivo .gitignore já está configurado para nunca enviar seu arquivo .env para o GitHub.


### Passo 3: Construa e Execute o Container

Abra um terminal na raiz do projeto e execute o seguinte comando:

```bash
docker-compose up --build
```
- ```--build```: Força o Docker a construir a imagem do zero (só é realmente necessário na primeira vez ou se você mudar o Dockerfile).
- O Docker irá ler seu ```Dockerfile```, instalar o Python, Tesseract, OpenCV e todas as dependências do requirements.txt em um ambiente isolado.
- O ```docker-compose``` irá ler seu ```.env``` e injetar a ```GOOGLE_API_KEY``` com segurança no container.


### Passo 4: Acesse a Aplicação

Abra seu navegador e acesse:
[http://localhost:8501](http://localhost:8501)


---

## 📂 Estrutura do Projeto
```
.
├── Dockerfile           # Receita para construir a imagem Docker
├── docker-compose.yml   # Orquestra o build e a injeção da API key
├── main.py              # O código principal da aplicação Streamlit
├── requirements.txt     # Dependências Python
├── best.pt              # (Local) Modelo YOLOv8 para detecção de PII
├── .env                 # (Local) Onde você coloca sua API key (ignorado pelo Git)
├── .gitignore           # Ignora arquivos do Git (como .env e .pt)
├── .dockerignore        # Ignora arquivos do build do Docker
└── README.md            # Esta documentação
```
