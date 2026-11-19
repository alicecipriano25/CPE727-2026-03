"""Duas tarefas sobre a mesma distribuição de entradas, com poucos rótulos."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent


def amostras(seed, n):
    rng = np.random.default_rng(seed)
    grupo = np.repeat([0, 1, 2], n)
    centros = np.array([[-4, 0], [0, 0], [4, 0]])
    x = centros[grupo] + rng.normal(size=(len(grupo), 2)) * [0.63, 0.85]
    return x, grupo


def run():
    x, grupo = amostras(727, 200)
    xt, gt = amostras(728, 300)
    cluster = KMeans(n_clusters=3, n_init=10, random_state=727).fit(x)
    train_clusters, test_clusters = cluster.labels_, cluster.predict(xt)
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        labeled = np.concatenate(
            [rng.choice(np.flatnonzero(grupo == k), 3, replace=False) for k in range(3)]
        )
        tasks = {
            "alinhado": (grupo == 1, gt == 1),
            "cruzado": (x[:, 1] > 0, xt[:, 1] > 0),
        }
        for name, (y, yt) in tasks.items():
            knn = KNeighborsClassifier(n_neighbors=1).fit(x[labeled], y[labeled])
            labels = {}
            for k in range(3):
                members = y[labeled[train_clusters[labeled] == k]]
                labels[k] = (
                    members.mean() > 0.5 if len(members) else y[labeled].mean() > 0.5
                )
            pred = np.array([labels[k] for k in test_clusters])
            for method, acc in [
                ("1nn", knn.score(xt, yt)),
                ("grupos", np.mean(pred == yt)),
            ]:
                rows.append(
                    {
                        "seed": seed,
                        "tarefa": name,
                        "metodo": method,
                        "acuracia": float(acc),
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results_local")
    args = parser.parse_args()
    rows = run()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "grupos.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for task in ("alinhado", "cruzado"):
        for method in ("1nn", "grupos"):
            values = [
                r["acuracia"]
                for r in rows
                if r["tarefa"] == task and r["metodo"] == method
            ]
            print(
                f"{task:9s} · {method:6s} {np.mean(values):.2%} ± {np.std(values, ddof=1):.2%}"
            )


if __name__ == "__main__":
    main()
