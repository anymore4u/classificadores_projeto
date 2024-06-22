# Use uma imagem base com Python
FROM python:3.12.4

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie os arquivos de requisitos e o script para o diretório de trabalho
COPY requirements.txt requirements.txt
COPY classificadores.py classificadores.py
COPY LLM.csv LLM.csv

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Comando para rodar o script
CMD ["python", "classificadores.py"]
