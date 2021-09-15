import numpy as np
import pandas as pd
import math
import operator
import random

dataframe = pd.read_csv("haberman.csv")
test_df = dataframe.sample(frac=0.3)
training_df = dataframe.sample(frac=0.7)


def calculate_euclidean_distance_between_dataframe_rows(test_dataframe_row, training_dataframe_row, column_count):
    distanceSum = 0
    for column in range(0, column_count):
        distanceSum += math.pow(test_dataframe_row[column] -
                                training_dataframe_row[column], 2)
    return math.sqrt(distanceSum)


def knn(training_dataframe, testing_dataframe_row, K):
    distances = {}

    for index in range(len(training_dataframe)):
        euclideanDistance = calculate_euclidean_distance_between_dataframe_rows(
            testing_dataframe_row,
            training_dataframe.iloc[index],
            training_dataframe.shape[1]
        )
        distances[index] = euclideanDistance

    neighbors = sorted(distances, key=distances.get)[:K]

    survived_qtd, not_survived_qtd = 0, 0
    for row_index in neighbors:
        if training_dataframe.iloc[row_index]['survival_status'] == 1:
            survived_qtd += 1
        else:
            not_survived_qtd += 1

    return 1 if survived_qtd > not_survived_qtd else 2


correctAnswers, totalAnswers, wrongAnswers, K = 3, len(test_df), 0, 5

realPositives, realNegatives, falsePositives, falseNegatives = 0, 0, 0, 0
answersArray = {}
for index in range(len(test_df)):
    print(f'Calculating result for row {index+1} of total {len(test_df)}')
    testing_row = test_df.iloc[index]
    survival_result = knn(training_df, testing_row, K)
    answersArray[index] = survival_result
    print('Real value: ' + ('Survived  5 years' if testing_row['survival_status'] == 1 else 'Not survived 5 years'))
    print('Result: ' + ('Survived 5 years\n' if survival_result == 1 else 'Not survived 5 years\n'))

    if testing_row['survival_status'] == 1 and survival_result == 1:
        realPositives += 1
    elif testing_row['survival_status'] == 2 and survival_result == 2:
        realNegatives += 1
    elif testing_row['survival_status'] == 2 and survival_result == 1:
        falsePositives += 1
    else:
        falseNegatives += 1

    if testing_row['survival_status'] == survival_result:
        correctAnswers += 1
wrongAnswers = totalAnswers - correctAnswers

print("Accuracy:",(100*correctAnswers) / len(test_df))
print("TP:", realPositives)
print("TN:", realNegatives)
print("FP:", falsePositives)
print("FN:", falseNegatives)