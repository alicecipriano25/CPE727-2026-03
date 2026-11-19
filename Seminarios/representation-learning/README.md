# Representation Learning

**Vinícius Ferreira Neves**  
CPE727 · Aprendizado Profundo, PEE/COPPE/UFRJ

O seminário segue as seis seções do capítulo 15 de *Deep Learning*, de Goodfellow, Bengio e Courville. Parte da definição de representação aprendida e passa por pré-treinamento, transferência, fatores de variação, códigos distribuídos, profundidade e hipóteses sobre os dados. O bloco final reúne aplicações práticas, dois experimentos e uma demonstração de similaridade imagem–texto.

## Materiais

- [Slides](representation_learning.pdf) e [fonte Beamer](main.tex).
- [Notebook](exemplos/codigos.ipynb), [protocolo dos experimentos](exemplos/PROTOCOLO.md) e [resultados por seed](exemplos/results/).

As ilustrações que remetem às figuras do livro são esquemas próprios. As imagens de dígitos são exemplos do conjunto público Digits, com fonte no protocolo.

## Executar os exemplos

Com Python 3.11 ou 3.12, a partir desta pasta:

```bash
cd exemplos
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python grupos.py
python transferencia.py
python zero_shot.py
```

Com Docker, a partir da pasta do seminário:

```bash
docker build -t representation-learning exemplos
mkdir -p exemplos/results
docker run --rm --mount "type=bind,source=${PWD}/exemplos/results,target=/app/results" representation-learning
```
