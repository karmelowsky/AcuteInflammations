from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def cv2(X_train, X_test, y_train, y_test, kneighbors, metric = 'euclidean'):

    knn = KNeighborsClassifier(n_neighbors=kneighbors, metric=metric)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    score1 = accuracy_score(y_test, pred)

    knn = KNeighborsClassifier(n_neighbors=kneighbors, metric=metric)
    knn.fit(X_test, y_test)
    pred = knn.predict(X_train)
    score2 = accuracy_score(y_train, pred)

    return (score2+score1)/2

def crossValidation(X_train, X_test, y_train, y_test, count, kneigbors=3, metric='euclidean'):
    scores = []
    for i in range(0, count):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        knn = KNeighborsClassifier(n_neighbors=kneigbors, metric=metric)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        score1 = accuracy_score(y_test, pred)
        scores.append(score1)

        knn = KNeighborsClassifier(n_neighbors=kneigbors, metric=metric)
        knn.fit(X_test, y_test)
        pred = knn.predict(X_train)
        score2 = accuracy_score(y_train, pred)
        scores.append(score2)

    return sum(scores) / len(scores)


def crossValidationWithScaling(X, y, count, kneigbors=3, metric='euclidean'):
    scores = []
    for i in range(0, 100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        scaledX = preprocessing.scale(X_train)
        knn = KNeighborsClassifier(n_neighbors=kneigbors, metric=metric)
        knn.fit(scaledX, y_train)
        pred = knn.predict(X_test)
        score1 = accuracy_score(y_test, pred)
        scores.append(score1)

        scaledX = preprocessing.scale(X_test)
        knn = KNeighborsClassifier(n_neighbors=kneigbors, metric=metric)
        knn.fit(scaledX, y_test)
        pred = knn.predict(X_train)
        score2 = accuracy_score(y_train, pred)
        scores.append(score2)

    return sum(scores) / len(scores)
