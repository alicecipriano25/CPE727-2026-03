FROM python:3.12-slim

WORKDIR /app

ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

COPY requirements.txt ./

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "python grupos.py && python transferencia.py && python zero_shot.py"]
