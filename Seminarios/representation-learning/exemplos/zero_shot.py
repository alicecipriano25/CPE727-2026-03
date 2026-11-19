"""Ilustração de classificação zero-shot com vetores de duas dimensões."""

from math import hypot

IMAGE = (0.95, 0.25)
DESCRIPTIONS = {
    "um quadrado": (1.0, 0.2),
    "um triângulo": (0.6, 0.8),
    "um círculo": (0.1, 1.0),
}


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (hypot(*a) * hypot(*b))


def rank(descriptions):
    """Ordena as descrições pela similaridade com a imagem fixa."""
    scores = [(label, cosine(IMAGE, DESCRIPTIONS[label])) for label in descriptions]
    return sorted(scores, key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    for candidates in (tuple(DESCRIPTIONS), ("um triângulo", "um círculo")):
        print("Candidatas:", ", ".join(candidates))
        for label, score in rank(candidates):
            print(f"  {label:14s} {score:.3f}")
