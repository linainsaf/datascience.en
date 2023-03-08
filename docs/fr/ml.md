## On fait quoi avec ?

Le machine learning peut être utilisé pour différentes tâches, selon le type de données que vous avez et le problème que vous essayez de résoudre. Voici un aperçu de quelques-unes des tâches les plus courantes :

- **Régression** : la régression consiste à prédire une valeur numérique en fonction d'un ou plusieurs paramètres. Par exemple, vous pouvez utiliser la régression pour prédire le prix d'une maison en fonction de son emplacement, de sa taille et de ses caractéristiques. 

- **Classification** : la classification consiste à prédire une étiquette discrète en fonction d'un ensemble de caractéristiques. Par exemple, vous pouvez utiliser la classification pour prédire si un email est du spam ou non, en fonction de son contenu et de ses caractéristiques.

- **Clustering** : le clustering consiste à regrouper des points de données similaires dans des groupes, sans étiquettes de classe prédéfinies. Par exemple, vous pouvez utiliser le clustering pour regrouper les clients de votre boutique en fonction de leurs habitudes d'achat.

- **Réduction de dimension** : la réduction de dimension consiste à réduire la complexité des données en réduisant le nombre de variables ou de caractéristiques. Par exemple, vous pouvez utiliser la réduction de dimension pour visualiser des données à haute dimension en deux ou trois dimensions.

## Types de M L:

Il existe deux types principaux :

### L'apprentissage supervisé : 

Dans ce type d'apprentissage, le modèle apprend à prédire une sortie à partir d'une entrée, en utilisant un ensemble de données d'apprentissage qui contient des exemples d'entrées et de sorties connues. Le modèle ajuste ensuite ses paramètres pour minimiser l'erreur de prédiction. 

<br />

Voici quelques modèles de machine learning supervisés :

#### Régression linéaire  :

La régression linéaire est une méthode de prédiction qui permet de modéliser la relation entre une variable cible continue et une ou plusieurs variables prédictives continues ou catégorielles. Le but de la régression linéaire est de trouver la meilleure droite (ou hyperplan en dimension supérieure) qui représente la relation entre les variables. 

<br />

Le modèle peut être utilisé pour la prédiction ou pour comprendre la relation entre les variables. Par exemple, dans le cas de la prédiction de prix de vente d'une maison, les variables prédictives pourraient être la superficie, le nombre de chambres, la localisation, etc.

<br />


Exemple de code :

```
from sklearn.linear_model import LinearRegression

# Création d'un modèle de régression linéaire
regression_model = LinearRegression()

# Entraînement du modèle sur les données d'entraînement
regression_model.fit(X_train, y_train)

# Prédiction des valeurs pour les données de test
y_pred = regression_model.predict(X_test)
```

#### Régression logistique :

La régression logistique est une méthode de classification qui permet de prédire la probabilité qu'une observation appartienne à une classe particulière (binaire ou multi-classes). La sortie est une probabilité, donc la valeur prédite est toujours comprise entre 0 et 1. 

<br />

Le modèle estime la probabilité de l'appartenance à chaque classe en fonction des variables prédictives. Le modèle est souvent utilisé pour la classification de clients potentiels, de mails comme spam ou non, etc.

<br />


Exemple de code :

```
from sklearn.linear_model import LogisticRegression

# Création d'un modèle de régression logistique
logistic_model = LogisticRegression()

# Entraînement du modèle sur les données d'entraînement
logistic_model.fit(X_train, y_train)

# Prédiction des classes pour les données de test
y_pred = logistic_model.predict(X_test)
```
#### Arbre de décision :

<br />

Les arbres de décision sont une méthode de classification et de régression qui permettent de créer un modèle à partir des données en formes d'arbre. Les noeuds internes de l'arbre représentent une condition sur les variables prédictives, les feuilles de l'arbre représentent les prédictions de la variable cible. Le modèle est facilement interprétable et permet de comprendre comment les variables sont utilisées pour la prédiction.

<br />

Exemple de code :

```
from sklearn.tree import DecisionTreeClassifier

# Création d'un modèle d'arbre de décision
tree_model = DecisionTreeClassifier()

# Entraînement du modèle sur les données d'entraînement
tree_model.fit(X_train, y_train)

# Prédiction des classes pour les données de test
y_pred = tree_model.predict(X_test)
```

#### Forêt d'arbres décisionnels :

Les forêts d'arbres décisionnels sont une méthode de classification et de régression qui combine les résultats de plusieurs arbres de décision. Chaque arbre dans la forêt est entraîné sur un échantillon aléatoire des données d'entraînement. Le modèle agrège les prédictions des différents arbres pour donner une prédiction finale. Les forêts d'arbres décisionnels sont souvent plus précises que les arbres de décision.

<br />

Voici un exemple de code utilisant Scikit-learn pour entraîner un modèle de forêt aléatoire pour la classification :

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_features=4, random_state=0)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
```

#### Les Machines à Vecteurs de Support (SVM):

<br />

Les SVM sont un algorithme d'apprentissage supervisé utilisé pour la classification et la régression. Les SVM trouvent une frontière de décision qui maximise la marge entre les deux classes. Cette frontière de décision est appelée un hyperplan. Les SVM peuvent être utilisées pour des tâches de classification avec deux classes ou plus.

<br />

Les SVM ont une grande précision et sont populaires dans les tâches de classification. Ils peuvent également être utilisés pour la régression.

<br />

Voici un exemple de code utilisant Scikit-learn pour entraîner un modèle SVM pour la classification :


```
from sklearn import svm
from sklearn.datasets import make_classification

X, y = make_classification(n_features=4, random_state=0)
clf = svm.SVC(kernel='linear', C=1, random_state=0)
clf.fit(X, y)
```

### L'apprentissage non supervisé : 

Les modèles non supervisés sont des algorithmes de machine learning qui n'ont pas besoin de données étiquetées pour fonctionner. Ils sont utilisés pour explorer les données et trouver des structures cachées, des modèles et des corrélations. Voici quelques exemples de modèles non supervisés:


#### K-means clustering :

Le K-means clustering est une technique de partitionnement de données qui permet de diviser un ensemble de données en K groupes distincts (clusters). Chaque point de données est affecté à un cluster en fonction de sa proximité avec le centre de ce cluster.

<br />

Voici un exemple de code en Python pour appliquer le K-means clustering :

```
from sklearn.cluster import KMeans
import numpy as np

# Générer des données aléatoires
X = np.random.rand(100, 2)

# Instancier le modèle de clustering
kmeans = KMeans(n_clusters=3)

# Fitter le modèle aux données
kmeans.fit(X)

# Prédire les clusters pour de nouvelles données
labels = kmeans.predict(X)
```

#### Analyse en composantes principales (PCA):

<br />

L'analyse en composantes principales est une méthode de réduction de dimension qui permet de projeter des données à haute dimension sur un espace de dimension inférieure tout en conservant autant que possible les informations contenues dans les données originales.

<br />

Voici un exemple de code en Python pour appliquer l'analyse en composantes principales :


```
from sklearn.decomposition import PCA
import numpy as np

# Générer des données aléatoires
X = np.random.rand(100, 10)

# Instancier le modèle de PCA
pca = PCA(n_components=3)

# Fitter le modèle aux données
pca.fit(X)

# Transformer les données pour obtenir les nouvelles dimensions
X_pca = pca.transform(X)
```

#### Réduction de dimension avec t-SNE: 

<br />

t-SNE est une méthode de réduction de dimension qui permet de projeter des données à haute dimension sur un espace de dimension inférieure en conservant autant que possible les relations de proximité entre les données.

<br />

Voici un exemple de code en Python pour appliquer la réduction de dimension avec t-SNE :


```
from sklearn.manifold import TSNE
import numpy as np

# Générer des données aléatoires
X = np.random.rand(100, 10)

# Instancier le modèle de t-SNE
tsne = TSNE(n_components=2)

# Fitter le modèle aux données
X_tsne = tsne.fit_transform(X)
```