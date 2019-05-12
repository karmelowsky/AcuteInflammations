from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import numpy as np
from scipy.stats.stats import pearsonr
from numpy import zeros
from myFunctions import cv2NN
from myFunctions import cv2NM
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

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

for i in range(6):
    corr, _ = pearsonr(npArray[:, i], npArray[:, 8])
    correlations.append([i, corr.__abs__()])

npCorrelations = np.array(correlations)
sortedCorrelations = sorted(npCorrelations, key=itemgetter(1), reverse=True)
print("\nKorelacja cech z decyzja: ")
print(np.array(sortedCorrelations))

bestAttributes1 = np.ravel(np.array(sortedCorrelations)[:, 0]).astype(int)

chi2Features = SelectKBest(chi2, k=6)
x_kbest_features = chi2Features.fit_transform(npArray[:,[0,1,2,3,4,5]],npArray[:, 8])

print("\nPunkty cech w Chi2 test")
print(chi2Features.scores_)

bestAttributes1 = [3, 2, 4, 1, 5, 0]
#endregion


###################KNN
print("\n###ALGORYTM KNN###\n")
###################klasa 1

howmanytimes = 5

#array dimemnsions : test, k-nn value, featureCount, metrics, non/normalized
results = zeros([howmanytimes, 11, len(bestAttributes1), 2, 2])

#                #testCount, featureCount, metrics
resultsNM = zeros([howmanytimes, len(bestAttributes1), 2, 2])
metrics = ['euclidean', 'manhattan']

X = npArray[:, 0:6]
y = npArray[:, 8]
print("Ostre zapalenie drog moczowych, poprawnosc klasyfikacji:")


kValues = [1, 5, 10]
nonNormalizedIndex = 0
normalizedIndex = 1

for testCount in range(howmanytimes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    for metricIndex in [0,1]:
        for atrIndex in range(0, len(bestAttributes1)):

            score = cv2NM(X_train[:, bestAttributes1[0:atrIndex + 1]], X_test[:, bestAttributes1[0:atrIndex + 1]], y_train, y_test, metric=metrics[metricIndex])
            resultsNM[testCount][atrIndex][metricIndex][nonNormalizedIndex] = score

            normalizedDataScore = cv2NM(X_train[:, bestAttributes1[0:atrIndex + 1]], X_test[:, bestAttributes1[0:atrIndex + 1]], y_train, y_test, metric=metrics[metricIndex], scalling=True)
            resultsNM[testCount][atrIndex][metricIndex][normalizedIndex] = normalizedDataScore

            for k in kValues:
                score = cv2NN(X_train[:, bestAttributes1[0:atrIndex + 1]], X_test[:, bestAttributes1[0:atrIndex + 1]], y_train, y_test, kneighbors=k, metric=metrics[metricIndex])
                results[testCount][k][atrIndex][metricIndex][nonNormalizedIndex] = score

                normalizedDataScore = cv2NN(X_train[:, bestAttributes1[0:atrIndex + 1]], X_test[:, bestAttributes1[0:atrIndex + 1]], y_train, y_test, kneighbors=k, metric=metrics[metricIndex], scalling= True)
                results[testCount][k][atrIndex][metricIndex][normalizedIndex] = normalizedDataScore

#print(resultsNM) #macierz wynikow


for metricIndex in [0,1]:
    print("\nMetryka: {}".format(metrics[metricIndex]))
    for k in kValues:
        print("\nK: {}".format(k))

        for atrIndex in range(0, len(bestAttributes1)):
            print("{} atrybutow: {}".format(atrIndex+1, results[:,k, atrIndex, metricIndex, nonNormalizedIndex].mean()))

print("\n Algorytm NM")
for metricIndex in [0,1]:
    print("\nMetryka: {}".format(metrics[metricIndex]))
    for atrIndex in range(0, len(bestAttributes1)):
        print("{} atrybutow: {}".format(atrIndex+1, resultsNM[:, atrIndex, metricIndex, nonNormalizedIndex].mean()))

