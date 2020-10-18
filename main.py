import math as mt
import pandas as pd
import numpy as np
import random as rd


# Task 1
print("Task 1")
critiques = pd.read_csv("movies.csv", index_col=0)
print(critiques)
print()


# Task 2a (calculating similarity):


def get_distance(user_1, user_2, data, criteria):
    distance = None
    for item in data.index.to_list():
        rating_1, rating_2 = data[user_1][item], data[user_2][item]
        if not np.isnan(rating_1) and not np.isnan(rating_2) and criteria in ["manhattan", "euclidean"]:
            if distance is None: distance = 0
            distance += abs(rating_1 - rating_2) if criteria == "manhattan" else mt.pow(rating_1 - rating_2, 2)
    return distance if criteria == "manhattan" else mt.sqrt(distance) if criteria == "euclidean" else None


print("Task 2a")
print("Lisa Rose - Gene Seymour (manhattan): ", get_distance("Lisa Rose", "Gene Seymour", critiques, "manhattan"))
print("Lisa Rose - Gene Seymour (euclidean): ", get_distance("Lisa Rose", "Gene Seymour", critiques, "euclidean"))
print()


# Task 2b (making recommendations):


def get_distances(user_1, data, criteria):
    distances, users = [], data.columns.to_list()
    for user_2 in users:
        if user_2 != user_1 and criteria in ["manhattan", "euclidean"]:
            distances.append((get_distance(user_1, user_2, data, criteria), user_2))
    distances.sort()
    return distances if distances else None


def get_movies(user_1, data, criteria):
    nearest_neighbor, movies = get_distances(user_1, data, criteria)[0][1], []
    for item in data.index.to_list():
        rating_1, rating_2 = data[user_1][item], data[nearest_neighbor][item]
        if np.isnan(rating_1) and not np.isnan(rating_2):
            movies.append((rating_2, item))
    movies.sort(reverse=True)
    return movies


print("Task 2b")
print("Toby (manhattan): ", get_movies("Toby", critiques, "manhattan"))
print("Toby (euclidean): ", get_movies("Toby", critiques, "euclidean"))
print()


# Task 2c, 2d, 2e (exp, per, cos):


def pearson(user_1, user_2, data):
    n, sum_1, sum_2, sum_12, sum_1_pow, sum_2_pow = 0, 0, 0, 0, 0, 0
    for item in data.index.to_list():
        rating_1, rating_2 = data[user_1][item], data[user_2][item]
        if not np.isnan(rating_1) and not np.isnan(rating_2):
            n += 1
            sum_1 += rating_1
            sum_2 += rating_2
            sum_12 += rating_1 * rating_2
            sum_1_pow += pow(rating_1, 2)
            sum_2_pow += pow(rating_2, 2)
    denominator = mt.sqrt(sum_1_pow - pow(sum_1, 2) / n) * mt.sqrt(sum_2_pow - pow(sum_2, 2) / n)
    return 0 if denominator == 0 else (sum_12 - sum_1 * sum_2 / n) / denominator


def cosine(user_1, user_2, data):
    sum_12, sum_1_pow, sum_2_pow = 0, 0, 0
    for item in data.index.to_list():
        rating_1, rating_2 = data[user_1][item], data[user_2][item]
        if not np.isnan(rating_1) and not np.isnan(rating_2):
            sum_12 += rating_1 * rating_2
            sum_1_pow += pow(rating_1, 2)
            sum_2_pow += pow(rating_2, 2)
    denominator = mt.sqrt(sum_1_pow) * mt.sqrt(sum_2_pow)
    return 0 if denominator == 0 else sum_12 / denominator


def get_movies_2(user_1, data, criteria, criteria_2="def"):
    movies, distances = [], get_distances(user_1, data, criteria)
    for item in data.index.to_list():
        if np.isnan(data[user_1][item]):
            total, s = 0, 0
            for user_2 in data.columns.to_list():
                rating = data[user_2][item]
                if not np.isnan(rating):
                    distance = None
                    if distances is not None:
                        for pair in distances:
                            if pair[1] == user_2: distance = pair[0]
                    if criteria == "prs":
                        c = pearson(user_1, user_2, data)
                        s += c
                        total += rating * c
                    elif criteria == "cos":
                        c = cosine(user_1, user_2, data)
                        s += c
                        total += rating * c
                    elif criteria_2 == "exp":
                        s += mt.exp(-distance)
                        total += rating * mt.exp(-distance)
                    else:
                        s += 1 / (1 + distance)
                        total += rating / (1 + distance)
            movies.append((round(total/s, 1), item))
            movies.sort(reverse=True)
    return movies


def get_best_recommendation(user_1, data, criteria_1, criteria_2="def"):
    return get_movies_2(user_1, data, criteria_1, criteria_2)[0][1]


print("Task 2cde")
print("Toby (denominator): ", get_best_recommendation("Toby", critiques, "euclidean"))
print("Toby (exponencial): ", get_best_recommendation("Toby", critiques, "euclidean", "exp"))
print("Toby (pearson): ", get_best_recommendation("Toby", critiques, "euclidean", "prs"))
print("Toby (cosine): ", get_best_recommendation("Toby", critiques, "euclidean", "cos"))
print()


# Task 3
critiques3 = pd.read_csv("movies3.csv", index_col=0)
print("Task 3")
print("Veronica (cosine): ", get_best_recommendation("Veronica", critiques3, "euclidean", "cos"))
print("Hailey (cosine): ", get_best_recommendation("Hailey", critiques3, "euclidean", "cos"))
print()


# Task 4
# Elaborate your own example with n critics (8 ≥ n ≥ 5) and m movies (8 ≥ m ≥ 5) where we have the same
# recommendations, given to the one of these critics, when two similarity measures based on distances are used. These
# recommendations should be different with those given by Pearson and/or Cosine similarity measures.


def create_data(i_min, i_max, u_min, u_max):
    users, items, ratings = [], [], []
    n_users, n_items = rd.randint(u_min, u_max), rd.randint(i_min, i_max)
    for user_index in range(n_users):
        users.append("critique_"+str(user_index+1))
        for item_index in range(n_items):
            if len(users) == 1:
                items.append("film_"+str(item_index+1))
                ratings.append([rd.randint(0, 5)])
            else:
                ratings[item_index].append(rd.randint(0, 5))
    ratings[0][0] = None
    ratings[1][0] = None
    ratings[2][0] = None
    ratings[4][0] = None
    return pd.DataFrame(ratings, columns=users, index=items)


def generate_dataset_1():
    condition, data = False, None
    while not condition:
        data = create_data(5, 8, 5, 8)
        mnh = get_best_recommendation("critique_1", data, "manhattan")
        euc = get_best_recommendation("critique_1", data, "euclidean")
        prs = get_best_recommendation("critique_1", data, "prs")
        cos = get_best_recommendation("critique_1", data, "cos")
        condition = mnh == euc and prs == cos and prs != mnh
        if condition: print("mnh:", mnh, "/ euc:", euc, "/ prs:", prs, "/ cos:", cos)
    print(data)


print("Task 4 dataset (automatically generated):")
generate_dataset_1()
print()


# Task 5
# Elaborate your own example with n critics (n ≥ 10) and m movies (m ≥ 10) where the recommendations given to
# the one of these critics, based on at least 4 similarity measures, are completely different.
def generate_dataset_2():
    condition, data = False, None
    while not condition:
        data = create_data(10, 15, 10, 15)
        mnh = get_best_recommendation("critique_1", data, "manhattan")
        euc = get_best_recommendation("critique_1", data, "euclidean")
        prs = get_best_recommendation("critique_1", data, "prs")
        cos = get_best_recommendation("critique_1", data, "cos")
        condition = mnh != euc and euc != prs and prs != cos and cos != mnh and mnh != prs and euc != cos
        if condition: print("mnh:", mnh, "/ euc:", euc, "/ prs:", prs, "/ cos:", cos)
    print(data)


# (may take few minutes)
print("Task 5 dataset (automatically generated, MAY TAKE FEW MINUTES):")
generate_dataset_2()
