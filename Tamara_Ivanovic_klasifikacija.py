# Kod je napisan na osnovu materijala sa vezbi i dokumentacije za pandas biblioteku

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prep
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as met
# ----------------------------------------------------
# podesavanja da u rezultatu mogu da se vide sve kolone
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 17)
# -----------------------------------------------------

# ucitavanje podataka, cuvanje atributa, racunanje broja instanci(=45211) i provera null vrednosti
data = pd.read_csv('bank_customer_survey.csv')
attributes = data.columns
num_of_rows = data.count()[0]
print("Broj null vrednosti\n", data.isnull().sum())

print("Statistike podataka\n", data.describe())

# -------- Korelacije atributa sa trazenim --------
print("Korelacije ostalih atributa sa ciljnim y: ")
print(data[['job', 'y']].groupby("job").mean().sort_values("y", ascending=False))
print(data[['marital', 'y']].groupby("marital").mean().sort_values("y", ascending=False))
print(data[['education', 'y']].groupby("education").mean().sort_values("y", ascending=False))
print(data[['default', 'y']].groupby("default").mean().sort_values("y", ascending=False))
print(data[['housing', 'y']].groupby("housing").mean().sort_values("y", ascending=False))
print(data[['loan', 'y']].groupby("loan").mean().sort_values("y", ascending=False))
print(data[['contact', 'y']].groupby("contact").mean().sort_values("y", ascending=False))

yes_values = data.groupby("y")
print("Procenat klijenata koji su uzeli oroceni depozit\n", yes_values.mean())

print(pd.DataFrame(abs(data.corr()['y']).reset_index().sort_values('y', ascending=False)))
corr = data.corr()
print(corr)

# -------- Menjanje kategorije unknown ------------
data['job'].replace('unknown', 'self', inplace=True)
data['education'].replace('unknown', 'tertiary', inplace=True)

# --------- Izdvajanje potrebnih atributa ----------
new_data = data[['duration', 'pdays', 'previous', 'campaign', 'balance', 'job', 'education', 'y']].copy()
new_data.to_csv(r'D:\Fakultet\2018-19\Seminarski\ _pretprocesirani_podaci.csv', index=None, header=True)
new_data = data[['duration', 'pdays', 'previous', 'campaign', 'balance', 'y']].copy()

# ------------- Normalizacija i podela na trening i test skup ---------------
features = new_data.columns[:5].tolist()
x_original = new_data[features]
x = pd.DataFrame(prep.MinMaxScaler().fit_transform(x_original))
x.columns = features
new_data['y'].replace('0', 'no', inplace=True)
new_data['y'].replace('1', 'yes', inplace=True)
y = new_data["y"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, stratify=y)

# ------------------- Naivan Bajesov algoritam ----------------------
print("\n")
print("Naivan Bajesov algoritam ")
clf_gnb = GaussianNB()
clf_gnb.fit(x_train, y_train)
y_pred = clf_gnb.predict(x_test)

cnf_matrix = met.confusion_matrix(y_test, y_pred)
print("Matrica konfuzije", cnf_matrix, sep="\n")

accuracy = met.accuracy_score(y_test, y_pred)
print("Preciznost", accuracy)
print("\n")

class_report = met.classification_report(y_test, y_pred, target_names=['yes', 'no'])
print("Izvestaj klasifikacije", class_report, sep="\n")
