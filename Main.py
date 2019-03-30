from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import numpy as np
from myFunctions import crossValidation
from scipy.stats.stats import pearsonr
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

###########################
##GETTING DATA#####
with open('i.txt') as f:
    lines = f.readlines()

formattedList = []
for line in lines:
    formattedList.append(line.split())

converted = []
counter = 0
for fl in formattedList:
    converted.append(
        [float(fl[0]), fl[1] == 'yes', fl[2] == 'yes', fl[3] == 'yes', fl[4] == 'yes', fl[5] == 'yes', fl[6] == 'yes',
         fl[7] == 'yes'])
    if fl[7] == 'yes':
        counter = counter + 1

npArray = np.array(converted)

#######################################
#

correlations = []
correlations2 = []

for i in range(6):
    corr, ncorr = pearsonr(npArray[:, i], npArray[:, 6])
    correlations.append([i, corr.__abs__()])

    corr, ncorr = pearsonr(npArray[:, i], npArray[:, 7])
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

###################KNN
print("\n###ALGORYTM KNN###\n")
###################klasa 1
y = npArray[:, 6]

print("Zapalenie pecherza moczowego poprawnosc klasyfikacji:")
for k in [1, 5, 10]:
    print("\n{}-NN".format(k))
    for i in range(0, 6):
        X = npArray[:, bestAttributes1[0:i + 1]]
        print("Liczba cech: {}. Jakosc klasyfikacji: {}".format(i + 1, crossValidation(X, y, count=5, kneigbors=k)))




###################klasa 2
y = npArray[:, 7]

print("\n")
print("Zapalenie nerek pochodzenia miedniczek nerkowych poprawnosc klasyfikacji:\n")
for k in [1, 5, 10]:
    print("\n{}-NN".format(k))
    for i in range(0, 6):
        X = npArray[:, bestAttributes2[0:i + 1]]
        print("Liczba cech: {}. Jakosc klasyfikacji: {}".format(i + 1, crossValidation(X, y, count=5, kneigbors=k)))

###################NM
print("\n###ALGORYTM NM###\n")
