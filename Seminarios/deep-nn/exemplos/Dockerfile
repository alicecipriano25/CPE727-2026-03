# Ambiente para rodar o cap6_deepnn_demo.ipynb (NumPy + Jupyter; PyTorch em CPU só para a seção 10)
#
#   docker build -t deepnn-exemplos .
#   docker run --rm -p 8888:8888 deepnn-exemplos
#
# e abrir o link http://127.0.0.1:8888/... que aparece no terminal.

FROM python:3.12-slim

# BLAS em uma thread: os resultados do notebook ficam iguais de uma execução para outra
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

WORKDIR /exemplos

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY cap6_deepnn_demo.ipynb ./

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
