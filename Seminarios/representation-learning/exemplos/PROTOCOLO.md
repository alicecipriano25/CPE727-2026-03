# Experimentos do capítulo 15

## Grupos e rótulos (§15.3)

Três nuvens gaussianas geram 600 pontos de treino e 900 de teste. As mesmas coordenadas sustentam duas tarefas: identificar a nuvem central ou identificar o semiplano superior. Em cada seed, selecionamos três pontos por nuvem geradora iguais entre tarefas e métodos.

O primeiro método é um classificador de um vizinho sobre os nove exemplos rotulados. O segundo ajusta K-means às 600 coordenadas, com três grupos, dez inicializações e seed 727. Cada grupo recebe a classe majoritária entre os pontos rotulados que lhe pertencem. Em caso de empate, escolhe-se a classe 0. Se um grupo aprendido ficar sem rótulos, usa-se a classe majoritária global entre os nove. A acurácia é calculada nos 900 pontos de teste.

## Transferência (§15.2)

Digits contém imagens de 8 × 8 pixels. O script divide as intensidades por 16 e verifica os índices do protocolo anterior. Usa o treino e o teste fixos. A partição de validação fica reservada, pois os hiperparâmetros deste exemplo são fixos.

- **Origem:** dígitos 0–4, 541 exemplos no treino. O MLP 64 → 16 → 5 usa ReLU, Adam, taxa inicial 0,003, lotes de 64, 100 épocas e L2 igual a zero.
- **Destino:** dígitos 5–9, 537 exemplos disponíveis no treino e 180 no teste. Cada repetição usa 5 ou 20 rótulos por classe no treino de destino. O orçamento menor está contido no maior.
- **Representações:** pixels (64 valores), PCA ajustada na origem (16), projeção aleatória com ReLU e inicialização Glorot (16) e camada oculta do MLP de origem (16).
- **Avaliação:** congelar as representações e ajustar um StandardScaler e uma regressão logística para cada uma, usando apenas os exemplos rotulados selecionados no treino de destino. Os rótulos de teste entram apenas na avaliação.

O encoder aleatório controla a contribuição de uma projeção com a mesma dimensão e ativação. Pixels e PCA são referências adicionais. O teste mede o que uma sonda linear lê do código congelado.

Fonte dos dados: [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), via `sklearn.datasets.load_digits`.

## Leitura dos resultados

Os CSVs em `results/` contêm cinco seeds (0–4). Os gráficos mostram a média e o desvio padrão amostral de acurácia entre essas execuções, com teste fixo. No Digits, as seeds variam os rótulos selecionados e a inicialização da rede. No exemplo, variam somente os rótulos selecionados. Para estender a comparação, mantenha as mesmas amostras, as mesmas seeds e a mesma sonda, declare mudanças de dimensão ou orçamento e gere os resultados de todas as repetições.

## Demonstração de similaridade imagem–texto

`zero_shot.py` usa vetores sintéticos para ilustrar a regra de decisão de um classificador imagem–texto. O vetor da imagem fica fixo. Altere a lista de descrições candidatas para verificar como muda a classe escolhida. Os escores são cossenos. A demonstração é didática, com vetores definidos no código. O artigo de referência descreve o treino de encoders a partir de pares reais de imagem e texto. [Radford et al. (2021)](https://proceedings.mlr.press/v139/radford21a.html).
