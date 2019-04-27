from operator import itemgetter
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats.stats import pearsonr
from numpy import zeros
from myFunctions import cv2NN
from myFunctions import cv2NM

#region GetData

with open('i.csv') as f:
    lines = f.readlines()

formattedList = []
for line in lines:
    formattedList.append(line.split(';'))

converted = []
counter = 0
for fl in formattedList:
    converted.append(
        [float(fl[0]), fl[1] == 'yes', fl[2] == 'yes', fl[3] == 'yes', fl[4] == 'yes', fl[5] == 'yes', fl[6] == 'yes',
         fl[7] == 'yes', int(fl[8])])

npArray = np.array(converted)

#endregion

#region GetBestFeatures
correlations = []
correlations2 = []

for i in range(6):
    corr, _ = pearsonr(npArray[:, i], npArray[:, 6])
    correlations.append([i, corr.__abs__()])

    corr, _ = pearsonr(npArray[:, i], npArray[:, 7])
    correlations2.append([i, corr.__abs__()])

npCorrelations = np.array(correlations)
sortedCorrelations = sorted(npCorrelations, key=itemgetter(1), reverse=True)
print("\nKorelacja cech z decyzja zapelenie pecherza moczowego: ")
print(np.array(sortedCorrelations))

npCorrelations2 = np.array(correlations2)
sortedCorrelations2 = sorted(npCorrelations2, key=itemgetter(1), reverse=True)
print("\nKorelacja cech z decyzja zapalenie nerek pochodzenia miedniczek nerkowych: ")
print(np.array(sortedCorrelations2))

bestAttributes1 = np.ravel(np.array(sortedCorrelations)[:, 0]).astype(int)
bestAttributes2 = np.ravel(np.array(sortedCorrelations2)[:, 0]).astype(int)
#endregion


###################KNN
print("\n###ALGORYTM KNN###\n")
###################klasa 1

howmanytimes = 100

#array dimemnsions : test, k-nn value, featureCount, metrics
results = zeros([howmanytimes, 11, len(bestAttributes1), 2])

#                #testCount, featureCount, metrics
resultsNM = zeros([howmanytimes, len(bestAttributes1), 2])
metrics = ['euclidean', 'manhattan']

X = npArray[:, 0:6]
y = npArray[:, 8]
print("Zapalenie pecherza moczowego poprawnosc klasyfikacji:")


kValues = [1, 5, 10]

for testCount in range(howmanytimes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    for metricIndex in [0,1]:
        for atrIndex in range(0, len(bestAttributes1)):

            score = cv2NM(X_train[:, bestAttributes1[0:atrIndex + 1]], X_test[:, bestAttributes1[0:atrIndex + 1]], y_train, y_test, metric=metrics[metricIndex])
            resultsNM[testCount][atrIndex][metricIndex] = score

            for k in kValues:
                score = cv2NN(X_train[:, bestAttributes1[0:atrIndex + 1]], X_test[:, bestAttributes1[0:atrIndex + 1]], y_train, y_test, kneighbors=k, metric=metrics[metricIndex])
                results[testCount][k][atrIndex][metricIndex] = score

#print(resultsNM) #macierz wynikow


for metricIndex in [0,1]:
    print("\nMetryka: {}".format(metrics[metricIndex]))
    for k in kValues:
        print("\nK: {}".format(k))

        for atrIndex in range(0, len(bestAttributes1)):
            print("{} atrybutow: {}".format(atrIndex+1, results[:,k, atrIndex, metricIndex].mean()))

print("\n Algorytm NM")
for metricIndex in [0,1]:
    print("\nMetryka: {}".format(metrics[metricIndex]))
    for atrIndex in range(0, len(bestAttributes1)):
        print("{} atrybutow: {}".format(atrIndex+1, resultsNM[:, atrIndex, metricIndex].mean()))


#mean = results[:, 1, 1]
#print(mean)




#
# print("Dane nieznormalizowane")
# #region NonNormalized
# print("\nMetryka euclidean")
# for k in [1, 5, 10]:
#     print("\n{}-NN".format(k))
#     for i in range(0, 6):
#         X = npArray[:, bestAttributes1[0:i + 1]]
#         print("Liczba cech: {}. Jakosc klasyfikacji: {}".format(i + 1, crossValidation(X, y, count=5, kneigbors=k)))
#
# print("\nMetryka manhattan")
# for k in [1, 5, 10]:
#     print("\n{}-NN".format(k))
#     for i in range(0, 6):
#         X = npArray[:, bestAttributes1[0:i + 1]]
#         print("Liczba cech: {}. Jakosc klasyfikacji: {}".format(i + 1, crossValidation(X, y, count=5, kneigbors=k, metric='manhattan')))
# #endregion
#
# print("Dane znormalizowane:")
# #region Normalized
# print("\nMetryka euclidean")
# for k in [1, 5, 10]:
#     print("\n{}-NN".format(k))
#     for i in range(0, 6):
#         X = npArray[:, bestAttributes1[0:i + 1]]
#         print("Liczba cech: {}. Jakosc klasyfikacji: {}".format(i + 1, crossValidationWithScaling(X, y, count=5, kneigbors=k)))
#
# print("\nMetryka manhattan")
# for k in [1, 5, 10]:
#     print("\n{}-NN".format(k))
#     for i in range(0, 6):
#         X = npArray[:, bestAttributes1[0:i + 1]]
#         print("Liczba cech: {}. Jakosc klasyfikacji: {}".format(i + 1, crossValidationWithScaling(X, y, count=5, kneigbors=k, metric='manhattan')))
# #endregion
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# ###################klasa 2
# y = npArray[:, 7]
#
# print("\n")
# print("Zapalenie nerek pochodzenia miedniczek nerkowych poprawnosc klasyfikacji:\n")
#
# print("\nMetryka euclidean")
#
# for k in [1, 5, 10]:
#     print("\n{}-NN".format(k))
#     for i in range(0, 6):
#         X = npArray[:, bestAttributes2[0:i + 1]]
#         print("Liczba cech: {}. Jakosc klasyfikacji: {}".format(i + 1, crossValidation(X, y, count=5, kneigbors=k)))
#
# print("\nMetryka manhattan")
#
# for k in [1, 5, 10]:
#     print("\n{}-NN".format(k))
#     for i in range(0, 6):
#         X = npArray[:, bestAttributes2[0:i + 1]]
#         print("Liczba cech: {}. Jakosc klasyfikacji: {}".format(i + 1, crossValidation(X, y, count=5, kneigbors=k, metric='manhattan')))
#
# ###################NM
# print("\n###ALGORYTM NM###\n")
