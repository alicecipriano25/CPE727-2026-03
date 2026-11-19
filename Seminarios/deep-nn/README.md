# Seminário: Redes Feedforward Profundas (DeepNN)

CPE727 - Aprendizado Profundo, PEE/COPPE/UFRJ
Gabriel H. B. Lisboa, 20/10/2026

Seminário sobre o capítulo 6 de *Deep Learning* (Goodfellow, Bengio e Courville): o exemplo do XOR, aprendizado baseado em gradiente, unidades de saída e funções de custo, funções de ativação, projeto de arquitetura e back-propagation.

```
deep-nn/
├── README.md
├── seminario_cap6_DeepNN.pdf         slides do seminário
├── seminario_cap6_DeepNN_latex.zip   fonte dos slides (Beamer)
└── exemplos/
    ├── cap6_deepnn_demo.ipynb        MLP em NumPy com back-propagation escrito à mão
    ├── requirements.txt              dependências do notebook
    └── Dockerfile                    ambiente para rodar o notebook
```

## Slides

O PDF é a versão apresentada. O fonte LaTeX (Beamer) está no `.zip`: basta descompactar e compilar o `main.tex` da raiz, ou enviar o `.zip` para o Overleaf.

## Exemplos

A rede é escrita em NumPy com o back-propagation explícito, sem biblioteca de diferenciação automática. O PyTorch aparece só na última seção, para conferir os gradientes.

| Seção | O que mostra |
|---|---|
| 1. Biblioteca | camadas `Linear`, `ReLU` e `Sigmoid` com `forward` e `backward`; classes `Sequencial` e `MLP`; custos e laço de treino |
| 2. XOR e o modelo linear | o modelo linear responde 0,5 nos quatro pontos; os pesos do livro resolvem exatamente |
| 3. Treino por gradiente | um MLP 2-8-1 aprende o XOR a partir de pesos aleatórios (custo final 0,0101) |
| 4. Verificação de gradientes | back-prop comparado com diferenças finitas em cinco redes |
| 5. Representar ≠ aprender | 100 seeds por largura: 46, 71, 84, 98 e 100 resolvem o XOR com 2, 3, 4, 8 e 16 unidades ocultas; ReLUs mortas |
| 6. Entropia cruzada × MSE | por que o custo importa quando a saída é sigmoide |
| 7. Saturação | o gradiente que chega às primeiras camadas: sigmoide × ReLU |
| 8. Softmax | três espirais: 54% com modelo linear, 99% com um MLP 2-64-64-3 |
| 9. Profundidade | as dobras de Telgarsky: 2^L pedaços lineares com L camadas |
| 10. Conferência com o PyTorch | os mesmos gradientes pelo `autograd` |

Com Docker:

```bash
cd exemplos
docker build -t deepnn-exemplos .
docker run --rm -p 8888:8888 deepnn-exemplos
```

e abrir o link `http://127.0.0.1:8888/...` que aparece no terminal.

Sem Docker (recomendo usar venv), com Python 3.10 ou mais recente:

```bash
cd exemplos
pip install -r requirements.txt
jupyter notebook cap6_deepnn_demo.ipynb
```

O PyTorch é opcional: sem ele, a seção 10 avisa que foi pulada e o resto roda normalmente. Para rodá-la também:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Todas as seeds são fixas. O treino das espirais (seção 8) é sensível ao arredondamento do BLAS: com `OPENBLAS_NUM_THREADS=1`, como no Dockerfile, a acurácia do MLP é 99,3%; com várias threads pode sair 99,0% (um ponto de diferença em 300). Os demais números não mudam.

## Referência principal

Goodfellow, Bengio e Courville. Deep Learning, cap. 6 (Deep Feedforward Networks). MIT Press, 2016.
https://www.deeplearningbook.org/contents/mlp.html
