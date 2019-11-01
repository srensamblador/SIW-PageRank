def coef_dice(bag1, bag2):
    # Coeficiente de Dice
    # 2 * len(intersection(X, Y)) / (len(X) + len(Y))
    if len(bag1) == 0 or len(bag2) == 0:
        return 0
    return 2 * len(bag1.intersection(bag2))/(len(bag1) + len(bag2))


def coef_jaccard(bag1, bag2):
    # Coeficiente de Jaccard
    # len(intersection(X, Y)) / len(union(X, Y))
    if len(bag1) == 0 or len(bag2) == 0:
        return 0
    return len(bag1.intersection(bag2))/len(bag1.union(bag2))


def coef_cosine(bag1, bag2):
    # Coeficiente del coseno
    # len(intersection(X, Y)) / (len(X) * len(Y))
    if len(bag1) == 0 or len(bag2) == 0:
        return 0
    return len(bag1.intersection(bag2))/(len(bag1)*len(bag2))


def coef_overlapping(bag1, bag2):
    # Coeficiente de solapamiento
    # len(intersection(X, Y)) / min(len(X), len(Y))
    if len(bag1) == 0 or len(bag2) == 0:
        return 0
    return len(bag1.intersection(bag2))/min(len(bag1), len(bag2))
