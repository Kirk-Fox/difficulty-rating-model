import numpy as np
import csv
from numpy.polynomial.polynomial import Polynomial
from statistics import mean


MAX_POLY_DEG: int = 1


def calculate_biases(
    difficulties: list[float], ratings: np.ndarray[tuple[int, int]]
) -> list[tuple[Polynomial, float]]:
    biases = []
    for s_ratings in ratings.T:
        data = np.array(
            [
                [difficulty, difficulty - rating]
                for (difficulty, rating) in zip(difficulties, s_ratings)
                if rating
            ]
        ).T
        polynomial = Polynomial.fit(data[0], data[1], MAX_POLY_DEG)

        mean_bias = mean(data[1])
        deter_coeff = 1 - sum(
            (bias - polynomial(difficulty)) ** 2 for [difficulty, bias] in data.T
        ) / sum((bias - mean_bias) ** 2 for bias in data[1])
        biases.append((polynomial, float(deter_coeff)))
    return biases


if __name__ == "__main__":
    rating_list: list[list[int | None]] = []
    with open("ratings.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rating_list.append(
                [int(string) if string.isdigit() else None for string in row]
            )

    ratings: np.ndarray[tuple[int, int]] = np.array(rating_list)

    difficulties: list[float] = [mean(filter(None, p_ratings)) for p_ratings in ratings]
    print("Averages")
    for d in difficulties:
        print(d)
    for _ in range(100):
        biases: list[tuple[Polynomial, float]] = calculate_biases(difficulties, ratings)
        difficulties = [
            mean(
                rating + biases[i][0](difficulties[j])  # * biases[i][1]
                for (i, rating) in enumerate(p_ratings)
                if rating
            )
            for (j, p_ratings) in enumerate(ratings)
        ]
    print("Adjusted Averages")
    for d in difficulties:
        print(d)
    for b in biases:
        print(b[0])
        print("R^2 =", b[1])
