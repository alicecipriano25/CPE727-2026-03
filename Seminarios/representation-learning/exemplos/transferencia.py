"""Transferência entre classes disjuntas do Digits, com sonda linear congelada."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
METHODS = ("pixels", "pca_origem", "encoder_aleatorio", "encoder_pre_treinado")


def dataset():
    digits = load_digits()
    x, y = digits.data / 16.0, digits.target
    split = json.loads((ROOT / "splits.json").read_text())
    return x, y, np.array(split["train"]), np.array(split["test"])


def run():
    x, y, train, test = dataset()
    source = train[y[train] < 5]
    target = train[y[train] >= 5]
    target_test = test[y[test] >= 5]
    pca = PCA(n_components=16, svd_solver="full").fit(x[source])
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        shuffled = {c: rng.permutation(target[y[target] == c]) for c in range(5, 10)}
        net = MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            learning_rate_init=0.003,
            batch_size=64,
            alpha=0,
            random_state=seed,
        )
        # A primeira chamada inicializa e atualiza a rede.
        init = np.random.RandomState(seed)
        radius = np.sqrt(6 / (64 + 16))
        w0 = init.uniform(-radius, radius, (64, 16))
        b0 = init.uniform(-radius, radius, 16)
        for _ in range(100):
            net.partial_fit(x[source], y[source], classes=np.arange(5))
        codes = {
            "pixels": x,
            "pca_origem": pca.transform(x),
            "encoder_aleatorio": np.maximum(0, x @ w0 + b0),
            "encoder_pre_treinado": np.maximum(
                0, x @ net.coefs_[0] + net.intercepts_[0]
            ),
        }
        for budget in (5, 20):
            labeled = np.concatenate(
                [indices[:budget] for indices in shuffled.values()]
            )
            for method, z in codes.items():
                probe = make_pipeline(
                    StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)
                )
                probe.fit(z[labeled], y[labeled])
                rows.append(
                    {
                        "seed": seed,
                        "rotulos_por_classe": budget,
                        "metodo": method,
                        "acuracia": probe.score(z[target_test], y[target_test]),
                    }
                )
    return rows, (len(source), len(target), len(target_test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results_local")
    args = parser.parse_args()
    rows, counts = run()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "transferencia.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Origem treino, destino treino e destino teste:", counts)
    for n in (5, 20):
        for method in METHODS:
            values = [
                r["acuracia"]
                for r in rows
                if r["rotulos_por_classe"] == n and r["metodo"] == method
            ]
            print(
                f"{n:2d} por classe · {method:22s} {np.mean(values):.2%} ± {np.std(values, ddof=1):.2%}"
            )


if __name__ == "__main__":
    main()
