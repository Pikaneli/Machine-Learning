import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
from pandas.core.common import SettingWithCopyWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarnings)
warnings.filterwarnings('ignore')

# ucitavanje podataka
dataset = pd.read_csv('smoke_detection_iot.csv')

# info o podacima
print("Info: ")
dataset.info()
print("-----------------------------------")
print("Opis: ")
print(dataset.describe())
print("-----------------------------------")
print("Shape: ")
print(dataset.shape)
print("-----------------------------------")

# proveravanje nepostojecih podataka
print("Broj null vrednosti: ")
print(dataset.isnull().sum())
print("-----------------------------------")

# izbacivanje nepotrebnih kolona
dataset.drop(['Unnamed: 0', 'CNT', 'UTC'], axis=1, inplace=True)
print("Nakon izbacivanja nepotrebnih kolona: ")
print(dataset.head(10))
print("-----------------------------------")

# matrica korelacija
plt.figure(figsize=(8, 6))
matrica = dataset.corr()
maska = np.triu(np.ones_like(matrica, dtype=bool))
sns.heatmap(matrica, mask=maska, annot=True, fmt=".1f", cmap="crest")
plt.title('Matrica korelacija')
plt.show()

# plot korelacija
korelacija = dataset.drop('Fire Alarm', axis=1)
korelacija.corr(method='spearman')
korelacija.corrwith(dataset['Fire Alarm']).plot(kind='barh', title="Zavisnost alarma od parametara")
plt.show()

# prikaz u kojem opsegu se nalaze parametri iz kolone
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
sns.histplot(data=dataset, x="Temperature[C]", hue="Fire Alarm", kde=True, fill=True, edgecolor="black")
plt.subplot(2, 2, 2)
sns.histplot(data=dataset, x="Humidity[%]", hue="Fire Alarm", kde=True, fill=True, edgecolor="black")
plt.subplot(2, 2, 3)
sns.histplot(data=dataset, x="Raw H2", hue="Fire Alarm", kde=True, fill=True, edgecolor="black")
plt.subplot(2, 2, 4)
sns.histplot(data=dataset, x="Raw Ethanol", hue="Fire Alarm", kde=True, fill=True, edgecolor="black")
plt.show()

# pronalazak anomalija(granica(0.5%,99.5%))
columns = dataset.columns
for x in columns:
	q995, q05 = np.percentile(dataset.loc[:, x], [99.5, 0.5])
	intr_qr = q995 - q05

	maximum = q995 + (1.5 * intr_qr)
	minimum = q05 - (1.5 * intr_qr)

	dataset.loc[dataset[x] < minimum, x] = np.nan
	dataset.loc[dataset[x] > maximum, x] = np.nan

print("Broj anomalija: ")
print(dataset.isnull().sum())
print("-----------------------------------")

# izbacivanje anomalija
dataset = dataset.dropna()
print("Nakon izbacivanja anomalija: ")
print(dataset.isnull().sum())
print("-----------------------------------")

# normalizacija podataka(MinMaxScaler)
min_max = MinMaxScaler()
dataset_skaliran = pd.DataFrame(min_max.fit_transform(dataset), columns=dataset.columns)
print("Skalirani podaci: ")
print(dataset_skaliran.head())
print("-----------------------------------")

# obucavajuci skupovi
X = dataset_skaliran.copy()
X.drop('Fire Alarm', axis=1, inplace=True)
y = dataset_skaliran['Fire Alarm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# KNN sa hiperparametrima
knn = KNeighborsClassifier()

parametriknn = {
	"n_neighbors": [3, 5, 7, 8, 10, 20, 50],
	"algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
}

knncv = GridSearchCV(knn, parametriknn, scoring='precision', cv=5)
knncv.fit(X_train, y_train)
print("K najblizih suseda:")
print("Najbolji parametri:", knncv.best_params_)

knn.n_neighbors = knncv.best_params_['n_neighbors']
knn.algorithm = knncv.best_params_['algorithm']
pocetak = time.time()
knn.fit(X_train, y_train)
# unakrsna validacija
kfold = KFold(n_splits=5)
scores = cross_val_score(knn, X_train, y_train, cv=kfold)
y_predknn = knn.predict(X_test)
kraj = time.time()

knntest = accuracy_score(y_test, knn.predict(X_test))

print("Tacnost tokom treniranja :", "%0.2f" % (accuracy_score(y_train, knn.predict(X_train))*100))
print("Tacnost tokom testiranja :", "%0.2f" % (accuracy_score(y_test, knn.predict(X_test))*100))
print("Preciznost tokom treniranja :", "%0.2f" % (precision_score(y_train, knn.predict(X_train))*100))
print("Preciznost tokom testiranja :", "%0.2f" % (precision_score(y_test, knn.predict(X_test))*100))
print("Odziv tokok treniranja :", "%0.2f" % (recall_score(y_train, knn.predict(X_train))*100))
print("Odziv tokom testiranja :", "%0.2f" % (recall_score(y_test, knn.predict(X_test))*100))
print("F1 tokom treniranja :", "%0.2f" % (f1_score(y_train, knn.predict(X_train))*100))
print("F1 tokom testiranja :", "%0.2f" % (f1_score(y_test, knn.predict(X_test))*100))
print("Rezultati unakrsne validacije :", scores)
print("Srednja tačnost: %0.2f" % scores.mean())
print("Vreme za koje je izvrseno :", "%0.2f" % ((kraj-pocetak)*100), 'sekundi')
print("-----------------------------------")

knn_matrica = confusion_matrix(y_test, y_predknn)
sns.heatmap(knn_matrica, annot=True, fmt=".1f", cmap="crest")
plt.title("Confusion Matrix")
plt.show()

# plot stabla
stablo = DecisionTreeClassifier()
stablo.fit(X_train, y_train)
predikcija = stablo.predict(X_test)

imena = columns[0:12]

fig = plt.figure(figsize=(20, 20))
_ = tree.plot_tree(stablo, feature_names=imena, class_names=['Alarm OFF', 'Alarm ON'], filled=True)
plt.show()

# Stablo sa hiperparametrima
parametristablo = {
	'criterion': ["gini", "entropy"],
	'ccp_alpha': [0, 0.01, 0.02, 0.05, 0.2, 0.4, 0.5, 1.0]
}

stablocv = GridSearchCV(stablo, parametristablo, scoring='precision', cv=5)
stablocv.fit(X_train, y_train)
print("Stablo:")
print("Najbolji parametri:", stablocv.best_params_)

stablo.criterion = stablocv.best_params_['criterion']
stablo.ccp_alpha = stablocv.best_params_['ccp_alpha']
pocetak = time.time()
stablo.fit(X_train, y_train)
# unakrsna validacija
kfold = KFold(n_splits=5)
scores = cross_val_score(stablo, X_train, y_train, cv=kfold)
y_predstablocv = stablo.predict(X_test)
kraj = time.time()

dttest = accuracy_score(y_test, stablo.predict(X_test))

print("Tacnost tokom treniranja :", "%0.2f" % (accuracy_score(y_train, stablo.predict(X_train))*100))
print("Tacnost tokom testiranja :", "%0.2f" % (accuracy_score(y_test, stablo.predict(X_test))*100))
print("Preciznost tokom treniranja :", "%0.2f" % (precision_score(y_train, stablo.predict(X_train))*100))
print("Preciznost tokom testiranja :", "%0.2f" % (precision_score(y_test, stablo.predict(X_test))*100))
print("Odziv tokok treniranja :", "%0.2f" % (recall_score(y_train, stablo.predict(X_train))*100))
print("Odziv tokom testiranja :", "%0.2f" % (recall_score(y_test, stablo.predict(X_test))*100))
print("F1 tokom treniranja :", "%0.2f" % (f1_score(y_train, stablo.predict(X_train))*100))
print("F1 tokom testiranja :", "%0.2f" % (f1_score(y_test, stablo.predict(X_test))*100))
print("Rezultati unakrsne validacije :", scores)
print("Srednja tačnost: %0.2f" % scores.mean())
print("Vreme za koje je izvrseno :", "%0.2f" % ((kraj-pocetak)*100), 'sekundi')
print("-----------------------------------")

# logisticka regresija
log = LogisticRegression()

parametri = {
	'solver': ['liblinear', 'newton-cg', 'sag'],
	'max_iter': [50, 60, 70, 80, 90, 100]
}

logcv = GridSearchCV(log, parametri, scoring='precision', cv=5)
logcv.fit(X_train, y_train)
print("Logisticka regresija:")
print("Najbolji parametri:", logcv.best_params_)

log.solver = logcv.best_params_['solver']
log.max_iter = logcv.best_params_['max_iter']
pocetak = time.time()
log.fit(X_train, y_train)
# unakrsna validacija
kfold = KFold(n_splits=5)
scores = cross_val_score(log, X_train, y_train, cv=kfold)
y_predlog = log.predict(X_test)
kraj = time.time()

logtest = accuracy_score(y_test, log.predict(X_test))

print("Tacnost tokom treniranja :", "%0.2f" % (accuracy_score(y_train, log.predict(X_train))*100))
print("Tacnost tokom testiranja :", "%0.2f" % (accuracy_score(y_test, log.predict(X_test))*100))
print("Preciznost tokom treniranja :", "%0.2f" % (precision_score(y_train, log.predict(X_train))*100))
print("Preciznost tokom testiranja :", "%0.2f" % (precision_score(y_test, log.predict(X_test))*100))
print("Odziv tokok treniranja :", "%0.2f" % (recall_score(y_train, log.predict(X_train))*100))
print("Odziv tokom testiranja :", "%0.2f" % (recall_score(y_test, log.predict(X_test))*100))
print("F1 tokom treniranja :", "%0.2f" % (f1_score(y_train, log.predict(X_train))*100))
print("F1 tokom testiranja :", "%0.2f" % (f1_score(y_test, log.predict(X_test))*100))
print("Rezultati unakrsne validacije :", scores)
print("Srednja tačnost: %0.2f" % scores.mean())
print("Vreme za koje je izvrseno :", "%0.2f" % ((kraj-pocetak)*100), 'sekundi')
print("-----------------------------------")

# Chi-Squared
# X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=5)
chi_selector.fit(X, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:, chi_support].columns.tolist()
print("5 najbitnijih kolona :", chi_feature)
print("-----------------------------------")

# logisticka regresija sa 5 najbitnijih kolona
X_selected = chi_selector.fit_transform(X, y)
y = dataset_skaliran['Fire Alarm']
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, stratify=y)

log5 = LogisticRegression()

logcv5 = GridSearchCV(log5, parametri, scoring='precision', cv=5)
logcv5.fit(X_train, y_train)
print("Logisticka regresija sa 5 najvaznijih kolona:")
print("Najbolji parametri:", logcv5.best_params_)

log5.solver = logcv5.best_params_['solver']
log5.max_iter = logcv5.best_params_['max_iter']
pocetak = time.time()
log5.fit(X_train, y_train)
# unakrsna validacija
kfold = KFold(n_splits=5)
scores = cross_val_score(log5, X_train, y_train, cv=kfold)
y_predlog5 = log5.predict(X_test)
kraj = time.time()

logtest5 = accuracy_score(y_test, log5.predict(X_test))

print("Tacnost tokom treniranja :", "%0.2f" % (accuracy_score(y_train, log5.predict(X_train))*100))
print("Tacnost tokom testiranja :", "%0.2f" % (accuracy_score(y_test, log5.predict(X_test))*100))
print("Preciznost tokom treniranja :", "%0.2f" % (precision_score(y_train, log5.predict(X_train))*100))
print("Preciznost tokom testiranja :", "%0.2f" % (precision_score(y_test, log5.predict(X_test))*100))
print("Odziv tokok treniranja :", "%0.2f" % (recall_score(y_train, log5.predict(X_train))*100))
print("Odziv tokom testiranja :", "%0.2f" % (recall_score(y_test, log5.predict(X_test))*100))
print("F1 tokom treniranja :", "%0.2f" % (f1_score(y_train, log5.predict(X_train))*100))
print("F1 tokom testiranja :", "%0.2f" % (f1_score(y_test, log5.predict(X_test))*100))
print("Rezultati unakrsne validacije :", scores)
print("Srednja tačnost: %0.2f" % scores.mean())
print("Vreme za koje je izvrseno :", "%0.2f" % ((kraj-pocetak)*100), 'sekundi')
print("-----------------------------------")

plt.figure(figsize=(15, 5))
plt.title('Tacnost modela')
models = ['LogisticRegression(chi-squared)', 'LogisticRegression', 'KNN Classifier', 'Decision Tree Classifier']
test_accuracy = [logtest5, logtest, knntest, dttest]
plt.plot(models, test_accuracy, marker='o', color='teal')
plt.ylim(0.65, 1.05)
plt.ylabel("Tacnost")
plt.grid()
plt.show()
