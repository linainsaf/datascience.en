# La Régression linéaire simple 


1. Introduction 
	2. Rappel sur le machine learning supervisé ?
	3. Le principe de régression : approche intuitive  
2. La régression linéaire simple en théorie
	3. Variables cible et variables explicative
	4. Explication des coefficient, constante et résidus 
	5. Les hypothèses derrière une régression linéaire : linéarité, homoscedasticité et normalité des variables
	7. Estimation par la méthode des moindres carrés
	8. Analyse et validation du modèle : R2 et MSE  

8. La régression linéaire simple en pratique avec la librairie scikit-learn


______________________

# Partie 1. Introduction  

Le machine learning une science multidisciplinaire qui a pour ambition de permettre à un ordinateur de résoudre des problèmes complexes qui ne peuvent être appréhendés par des algorithmes simples. On appelle algorithme simple une suite d'instruction ou conditions afin de calculer un résultat peu complexe.

Selon Wikipedia, en mathématiques, les régressions recouvrent plusieurs méthodes d’analyse statistique permettant d’approcher une variable à partir d’autres qui lui sont corrélées.

Dans ce cours on s'intéressera aux différentes techniques de régressions et plus particulièrement à la régression linéaire simple. 
 
L'objectif de la régression linéaire simple est de modéliser la relation entre une caractéristique unique/une variable explicative `x` et une réponse/une variable cible `y`. 
 

## 1.1. Rappel sur le machine learning supervisé ?

Le machine learning supervisé est une branche du machine learning qui vise à résoudre des problèmes pour lesquels on dispose d’exemples déjà résolus. 

Par exemple on rassemble des données sur un échantillons de logements à Paris qui décrivent leur localisation, divers caractéristiques ainsi que le montant du loyer. 

Si notre problème est d’estimer le montant du loyer d’un logement qui n’est pas dans notre base, on construira à partir de nos données un modèle qui estime le montant du loyer en fonction des caractéristiques du logement et on appliquera ce modèle (ou estimateur) au logement inconnu pour en estimer le loyer. Ce problème relève de l’apprentissage supervisé car au moment de construire notre modèle on connaissait les valeurs prises par la variable que l’on souhaite estimer, qu’on appelle la variable cible.


### 1.1.1. Pourquoi faire du machine learning supervisé ? 

Une fois le problème bien posé, c'est à dire que la variable cible à été choisi et que nous avons rassemblé un certain nombre de variables explicatives, on peut résumer les objectifs de l'apprentissage supervisée en trois grandes catégories :


* **Décrire** : On peut chercher à comprendre les relations qui peuvent exister entre la variable cible ```Y``` et le(s) variable(s) explicative(s) `x` dans le cas d'une régression simple et `X1, ..., Xp` dans le cas multiple afin par exemple de sélectionner celle qui sont le plus pertinentes, où obtenir une visualisation des comportements dans la population observée (attention ici population est employé comme un terme statistique et peut aussi bien désigner des personnes, des pays, des transactions financières etc…)
*   **Expliquer** : Lorsqu’on a une connaissance à priori du sujet traité, comme c’est souvent le cas en économie ou en biologie par exemple, l’objectif est de construire un test qui permet de vérifier ou confirmer des résultats théorique dans des situations pratiques.
*   **Prédire** : Ici on met l’accent sur la qualité d'estimation, on cherche à construire un modèle qui permette de produire des prédictions fiables pour de futures observations.



## 1.2. Le principe de régression : approche intuitive  

On rappel que en statistique un modèle de régression linéaire est un modèle qui cherche à établir une relation linéaire entre une variable `x` et une variable `y`. 

**L'objectif étant de pouvoir ensuite avec ce modèle faire des prévisions/prédictions sur la variable `y` à partir d'une variable `x` inconnue du modèle**

On peut prendre les exemples suivants : 

- Existe t'il une relation (linéaire) entre le nombre d'année d'expérience comme développeur noté `x` et un certain salaire noté `y`. 
- Existe t'il une relation (linéaire) entre la surface d'un appartement noté `x` et un certain prix de vente noté `y`. 

Développons notre première exemple avec les données abstaites ci-dessous :  


| expérience | 0  | 3  | 6  | 8  |
|------------|----|----|----|----|
| salaire    | 35 | 45 | 65 | 80 |


Commençons par tracer un nuage de point avec nos données. 

![Nuage de points](./intuitive_scatter.png) 


Après avoir représenté ces points on va essayer de voir si il existe finalement une relation linéaire entre les années d'expérience en tant que développeur et le salaire proposé. On peut reformuler cet énoncé comme "est ce que plus j'ai d'années d'expérience plus le salaire proposé augmente". Pour répondre à cette questions nous allons suivre les étapes suivantes : 

1. Tracer une droite qui passe "au mieux" par ces points
2. Formaliser un critère de sélection de "meilleure" droite
3. Analyser le résultat de la droite 


Commençons par tracer intuitivement la droite de régression (en rouge sur le schéma) qui passe "au mieux" par ces points. Ce qu'on entend par "au mieux" c'est un critère de réussite : la somme des écarts mesurés (en vert sur le schéma) entre tous les points et la droite doit être le plus petit possible.  

![Nuage de points avec une droite](./intuitive_scatter_reg.pdf) 

Par ailleurs, on appelle cette droite, une courbe afine et nous avons en mathématique une manière spécifique de la noter, nous y reviendrons dans la suite. 

Maintenant pour analyser cette droite il nous faut comme dit ci-dessus un certain critère qui va nous servir à mesurer la qualité de cette droite de régression : que la somme des écarts mesurés soit la plus petite possible on va donc pour cela aussi formaliser une mesure mathématique que nous verrons dans la suite du chapitre. 


__________________________________________________


# Partie 2. La régression linéaire simple en théorie


## 2.1. Variable cible et variable explicative

Selon Wikipédia, dans les mathématiques supérieures et en logique, une variable est un symbole représentant, a priori, un objet indéterminé. On peut cependant ajouter des conditions sur cet objet, tel que l'ensemble ou la collection le contenant. On peut alors utiliser une variable pour marquer un rôle dans un prédicat, une formule ou un algorithme, ou bien résoudre des équations et d'autres problèmes. 

Ce qui va nous intéresser dans le cadre du machine learning supervisé, comme énoncé ci-dessus c'est de trouver un algorithme, une fonction, un modele (dans ce contexte on peut considérer ces termes comme des synonymes) qui peut faire des prévisions/prédictions d'une variable cible noté `y` à partir d'une variable `x` inconnue. 

On va donc distinguer deux type de variables : 

- **la variable cible ou target** souvent notée `Y` est la variable dont on souhaite estimer la valeur. Par exemple, si nous essayons de prédire le salaire de quelqu’un en fonction de son nombre d’années d’expérience. La variable target `Y` correspond au salaire.
- **la variable explicative** souvent représentée par `X` est la variable que nous avons à disposition ou que l'on a choisi qui va nous permettre de déterminer la valeur d'une variables cible `Y`. Par exemple, si nous essayons de prédire le salaire de quelqu’un en fonction de son nombre d’années d’expérience. La variable explicative `X` correspond au nombre d’années d’expérience.


### 2.1.1 Les types de variables 

Avant de passer à la suite il est important de comprendre les différents types de variables que vous pouvez rencontrer dans le cadre de nos différents cours. 

#### 2.1.1.1 Les variables quantitatives 

Selon Wikipedia, en statistique, une variable quantitative ou critère qualitatif est une variable Ce lien renvoie vers une page d'homonymie ou grandeur qui peut être représentée par des nombres sur lesquels les opérations arithmétiques de base ont un sens. 

On distingue deux types de variables quantitatives : 

- **Les variables discrètes** : elles prennent des valeurs entières (exemple 1, 2, 3, 7). Elle ne peuvent pas etre des nombres à virgule, on peut prendre l'exemple du nombre de frères et soeurs, il est impossible d'avoir 1,5 soeur. 
- **Les variables continues** : elles prennent n'importe quelle valeur numérique, nombre entier ou bien à virgule. On peut prendre l'exemple d'un relevé de taille d'une population. 

#### 2.1.1.2 Les variables qualitatives

Selon Wikipedia, en statistique, une variable qualitative, une variable catégorielle, ou bien un facteur est une variable qui prend pour valeur des modalités, des catégories ou bien des niveaux, par opposition aux variables quantitatives qui mesurent sur chaque individu une quantité.

On distingue trois types de variables qualitatives :

- **Les variables binaires** : elles ne prennent uniquement que deux modalités comme les variable booléenne. Suivant les situations il est possible des les classer. 
- **Les variables ordinales** : elles prennent plusieurs modalités et peuvent être hiérarchisées entre elles. 
- **Les variables nominales** : elles prennent plusieurs modalités et il est impossible de les classer. Elles sont très courantes dans le domaine du machine  learning 


## 2.2. Explication des coefficient, constante et résidus 

Comme énoncé dans la partie précedente, considérons la cas classique d'une fonction affine :

$$y=ax+b$$

Ici, `a` et `b` sont des nombres réels. Ces deux nombres définissent entièrement la courbe et permet donc d'obtenir une relation affine entre `x` et `y`. 

On rappel que `a` le coefficient directeur de la droite `y` et `b` représente l'ordonnée à l'origine (nommée intercept). 

En statistique, cette relation est à la base des modèles dit linéaires, où une variable cible se définit comme une somme de variables explicatives où chacune de ces dernières sont multipliés par un coefficient. Dans ce chapitre nous verrons uniquement le cas de la régression avec une variable explicative. 


Dans ce modèle simple (à une seule variable explicative), on suppose que la variable réponse suit le modèle suivant :

$$y_i=\beta_0 + \beta_1 x_i + \varepsilon_i$$

On remarque la ressemblance avec la fonction affine présentée ci-dessus. La différence réside dans la notation (nous avons utilisé des lettre grec) ainsi que dans l'existence d'un terme aléatoire appelé bruit $\varepsilon_i$ . 

Afin de considérer le modèle, il est nécessaire de se placer sous certaines hypothèses dont nous parlerons par la suite. 

Si nous prenons l'exemple d'un échantillons de logements à Paris qui décrivent leur localisation, diverses caractéristiques ainsi que le montant du loyer. On pourrait imaginer une liste de prix `Y` ainsi qu'une liste de caractéristiques (observations) `X`. 

Les différents éléments qui interviennent dans la formule ci-dessus seraient donc :

- $ \beta_0$ : l'ordonnée à l'origine (c’est à dire le niveau 0 de `y` lorsque `x` vaut 0)
- $ \beta_1$ : le coefficient directeur, c'est le paramètre du modèle qui mesure l’influence de `x` sur `y`, si `x` augmente de 1, `y` augmentera de $ \beta_1$
- $ x_i$ : l'observation $i$
- $ y_i$ : le $i$-ème prix
- $ \varepsilon_i$ : le bruit aléatoire liée ou résidu à la $i$-ème observation

En effet, l’équation ci-dessus est la représentation d’un modèle statistique, il n’a pas la prétention d’être exact, il est vrai en moyenne, voilà qui explique la présence du résidu.

C'est donc avec ces paramètres et en fonction des individus (ou nuage de point), que le modèle va trouver la droite qui se rapproche le plus possible tous les individus à la fois. 


Quelques mots sur les résidus, souvent noté $\varepsilon$, ils correspondent à l’erreur commise lors de la modélisation. Cette erreur correspond à toute l’information qui n’est pas expliquée par le modèle, on suppose souvent que l’erreur suit une loi de probabilité particulière. Nous reparlerons de ce concept d'erreur dans la partie suivante 🤓

Dans ce cours nous ne verrons pas en détail comment calculer les coefficients du modèle $\beta_1$ et $\beta_0$ mais sachez qu'ils peuvent se calculer avec les expressions suivantes :  

$\beta_1 = \frac{Cov(X,Y)}{V(X)}$

et

$\beta_0 = \bar{Y} - a.\bar{X}$

ou : 

- $Cov(X,Y)$ désigne la covariance entre les variables $X$ et $Y$ 
- $\bar{Y}$ et $\bar{X}$ désignent la moyenne des variables $Y$ et $X$ 


## 2.3. Les hypothèses derrière une régression linéaire : linéarité, homoscedasticité et normalité des variables

Quand vous construisez un modèle de machine learning, vous devrez être conscient des hypothèses que vous devez respecter pour que votre modèle fonctionne bien. Dans le cas inverse, vous aurez des performances déplorables. 

Voici donc les hypothèses d’un modèle de régression linéaire simple :


### 2.3.1. Linéarité

La première hypothèse est simple. Il faut que vos points suivent à peu près une droite. En d’autres termes, vous devez vous assurer que votre variable dépendante suive une croissance linéaire à mesure que vos variables indépendantes augmentent.



### 2.3.2. Homoscedasticité

Au delà de la complexité du modèle en lui même, cela veut dire que la variance de vos points doit être relativement la même. Si vous avez une variance énorme, cela veut dire que vous avez des points très éloignés les uns des autres et que donc il sera difficile d’avoir une ligne qui soit représentative de votre dataset.



### 2.3.3. Normalité des variables

Les points doivent avoir une distribution normale (ou du moins à peu de choses près). Vous n’aurez cependant rarement une distribution normale de vos points. Le tout est d’avoir une moyenne, une médiane et un mode qui ne soit pas trop éloignés.  



## 2.4 Estimation par la méthode des moindres carrés 


Dans le langage statistique, un estimateur est une fonction permettant d'évaluer un paramètre inconnu (relatif à une loi de probabilité) à partir d'autre(s) paramètre(s). **Dans notre cas notre estimateur va etre notre modele / notre fonctions scikit-learn** donc notre droite de régression. 

Vous vous demandez sûrement comment on sait que la droite de notre modèle est celle qui se rapproche “Le plus” de chacun des points de notre dataset. Et bien, c’est grâce à _la méthode des moindres carrés_. 
Cette méthode va donc nous servir à calculer à mesurer les erreurs entre notre droite de prédiction et nos données.  
Nous n’allons pas aller trop loin dans la démonstration de la formule. Ce qu’il y a à comprendre est que l’algorithme va chercher la distance minimum possible entre chaque point dans votre graphique via cette formule :


$$min\sum_{i=0}^{n} (y_i - \hat{y_i})^2 = min\sum_{i=0}^{n} (y_i - \beta_0 - \beta_1x_i)^2$$

Dans cette équation, $y_i$ représente chaque individu (ou point) de votre jeu de données alors que $\hat{y_i}$ représente la prédiction de votre modèle. En reprenant les mêmes notations, $ \beta_0$ est l'ordonnée à l'origine, $ \beta_1$ le coefficient directeur et $ x_i$ l'observation de la ligne $i$. 

Le fait de soustraire pour tout les individus $\hat{y_i}$ à $y_i$ en mettant cette différence au carré revient à calculer la distance entre la prédiction de notre modèle en un point et le point lui même. Cette distance peut être considéré comme une mesure de la distance entre les données expérimentales et le modèle théorique qui prédit ces données (l'erreur de prédiction de notre estimateur). Si on cherche la distance minimum pour tous les points on obtient donc la "meilleure" droite, ou du moins celle qui minimise l'erreur. 

Après plusieurs itérations, l'algorithme est capable de "trouver la meilleure formule de la meilleure droite possible" qui décrit votre jeu de données.


## 2.5. Analyse et validation du modèle : $R^2$ et $MSE$   

### 2.5.1. Le coefficient $R^2$ 

Maintenant que nous avons compris comment fonctionne le modele linéaire voyons **comment analyser notre résultat et apprécier la qualité de notre prédiction**. 

Dans l'idée, nous aimerions obtenir une valeur numérique (une statistique "dans le jargon") qui nous indique **à quel point la régression linéaire a un sens sur nos données**. Pour cela, nous allons introduire quelques notations mathématiques que nous détaillerons bien par la suite donc pas de panique. 

- La somme des carrés résiduels : $SCR=\sum_{i=0}^{n}(y_i-\hat{y_i})^2=\varepsilon^2 $ est un indicateur qui quantifie l’erreur commise par le modèle, ou en d’autres termes la portion de la dispersion de la variable cible qui n’est pas expliquée par le modèle, d’où l’idée de résidu.
- La somme des carrés totaux : $SCT=\sum_{i=0}^{n}(y_i-\bar{y})^2$ est un indicateur de la dispersion des valeurs de la variable cible $Y$ (dont les valeurs sont notées $(y_1, ... y_n)$ sur la population considérée. 
- La somme des carrés expliqués : $SCE=\sum_{i=0}^{n}(\hat{y_i} - \bar{y})^2 $ est un indicateur qui représente la quantité de dispersion de la variable cible qui est expliquée par le modèle.  

Il est essentiel de bien comprendre ces valeurs car elles vont nous permettre de construire des métriques d’évaluations des modèles de régression linéaire simple et multiple. 

L'idée est de décomposer la somme des carrés totaux comme la somme des carrés que le modèle explique, en plus de la somme des carrés qui sont liés aux résidus (et donc que le modèle ne peut pas expliquer). On voit donc ici l'intérêt de calculer un coefficient à partir du $SCE$. Puisque l'on a la relation suivante :

$$SCT=SCE+SCR \text{ alors } 1=\frac{SCE}{SCT}+\frac{SCR}{SCT}$$


Plus les résidus sont petits (et donc la régression est "bonne"), plus $SCR$ devient petit et donc $SCE$ devient grand. Le schéma inverse s'opère de la même façon. Dans le meilleur des cas, on obtient $SCR=0$ et donc $SCE=SCT$ d'où le premier membre vaut $1$. Dans le cas contraite, $SCE=0$ et automatiquement, le premier membre est nul. C'est ainsi que l'on définit le coefficient de détermination $R^2$ comme 
$$R^2=\frac{SCE}{SCT}=1-\frac{SCR}{SCT}$$

Pour résumer, $SCT$ est la variance totale de la variable cible, qui peut se décomposer en deux composantes : $SCE$ la variance expliquée par le modèle, qui correspond à la quantité de variance de nos estimation par rapport à la moyenne réelle de la population observée et $SCR$ qui est la somme des carrés des écarts entre nos estimations est les valeurs réelles de la variables cible. En d’autres termes $SCT$ est la quantité d’information totale, $SCE$ est l’information expliquée par le modèle et $SCR$ l’information qui reste à expliquer, ou l’erreur commise.

Ainsi, $R^2 \in [0,1]$, et plus $R^2$ est proche de $1$, plus la régression linéaire a du sens. Au contraire, si $R^2$ est proche de $0$, le modèle linéaire possède un faible pouvoir explicatif.

On peut le calculer avec la fonction `r2_score` de `sklearn.metrics`. 


### 2.5.2. Le critère $MSE$ 

Selon Wikipedia En statistiques, l’erreur quadratique moyenne d’un estimateur $\hat{\theta}$ d’un paramètre $\theta$ ou mean squared error $MSE$
en anglais) est une mesure caractérisant la « précision » de cet estimateur. Elle est plus souvent appelée « erreur quadratique » (« moyenne » étant sous-entendu) ; elle est parfois appelée aussi « risque quadratique ».

Dans notre cas, l'estimateur sera notre algorithme `LinearRegression()` de `sklearn.linear_model`, l'objectif sera donc de mesurer sa précision et donc sa qualité de prédiction. On va pour cela faire la moyenne des erreurs au carrée avec la formule suivante : 

$$MSE = \frac{1}{n} \sum_{i=0}^{n} (y_i - \hat{y_i})^2 $$

Attention toutefois à l'interprétation de ce critère, suivant votre jeu de données il sera plus convenable de prendre la racine carrée appelé $RMSE$ pour root mean squared error. 

$$RMSE = [\frac{1}{n} \sum_{i=0}^{n} (y_i - \hat{y_i})^2 ]^\frac{1}{2} $$

On rappel que $x^\frac{1}{2}=\sqrt{x}$

Vous savez maintenant comment fonctionne une régression linéaire simple et analyser ses résultats. Passons maintenant à la pratique avec la librairie scikit-learn. 


__________________________________________________ 

# Partie 3. La régression linéaire simple en pratique avec la librairie scikit-learn 

Pour cette partie nous aurons besoin de plus de données que dans la partie précédente. On va donc pour cela générer des données linéaires avec la librairie numpy et les fonctions :  

- `numpy.arange()`: https://numpy.org/doc/stable/reference/generated/numpy.arange.html 
- `numpy.random.uniform()` : https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html 


```python
import numpy as np 
x=np.arange(75)
delta = np.random.uniform(-10,10, size=(75,))
y = 0.4 * x +3 + delta
```

On peut visualiser rapidement nos données avec la fonction `plot()` de  la librairie matplotlib

```python
import matplotlib.pyplot as plt 
plt.plot(x,y,"*")
plt.xlabel("Variable explicative 'x' ")
plt.ylabel("Variable cible 'y'")
plt.title("Nuage de point")
plt.savefig("./intuitive_scatter_bis.png")
```

![Nuage de points exemple](./intuitive_scatter_bis.png) 


On importe ensuite la librairie scikit-learn afin d'aller chercher l'algorithme de régression sous forme d'une fonction. 

Scikit-learn est un package python qui contient la plupart des modèles statistiques que nous allons utiliser dans les cours. On l'importe grâce à la commande suivante :


```python
from sklearn.linear_model import LinearRegression
```

Vous remarquerez que nous n'importons que le modèle qui nous intéresse (le modèle linéaire) car c'est un package assez lourd 🤓

On va ensuite créer notre fonction de regression de la façon suivante : 

```python
linear_model = LinearRegression()
```

Pour optimiser les paramètres du modèle (utiliser ce modèle générique et le faire coller à nos données) en utilisant la méthode des moindres carrés :


```python
linear_model.fit(x.reshape(-1, 1),y)
```

Le parametre `x.reshape(-1, 1)` de la fonction `fit()` est présent car la fonction `LinearRegression()` de scikit-learn est faite pour la régression multiple que nous verrons dans le prochain chapitre.  

On peut donc maintenant faire des prédictions à partir de données avec la fonction `predict()` tel que :


```
predictions = linear_model.predict(x)
```

Pour visualiser notre le modèle (droite de prédiction) et les données on peut utiliser la fonction suivante : 

```
def reg_plot(x,y,m):
    plt.scatter(x,y,c='blue',label="les données")
    plt.plot(x, m.predict(x.reshape(-1, 1)), color='red',label="droite de prédiction")
    plt.xlabel("Variable explicative 'x' ")
    plt.ylabel("Variable cible 'y'")
    plt.legend()
    return None 
```

et ensuite l'appliquer sur nos données tel que : 

```
reg_plot(x,y,linear_model)
```
![Nuage de points](./prediction.png) 

On peut aussi afficher les coefficients $R^2$ et $MSE$ tel que : 

```
y_pred = regr.predict(x)
print("Mean squared error: %.2f"
      % mean_squared_error(y, y_pred))
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y, y_pred)))
print("R square: %.2f"
      % r2_score(y, y_pred))
```