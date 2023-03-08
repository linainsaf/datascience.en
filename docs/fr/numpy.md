# NumPy 

NumPy est l'abréviation de Numerical Python. C'est une bibliothèque/package Python utilisée pour travailler avec des tableaux qui contiennent des classes, des fonctions, des variables, une grande bibliothèque de fonctions mathématiques, etc. pour travailler avec des calculs scientifiques. Il peut être utilisé pour créer un tableau "n" dimensionnel où "n" est un entier quelconque.

Pourquoi NumPy ?
En Python, nous avons des listes qui servent de tableaux, mais elles sont lentes. NumPy vise à fournir un objet de tableau qui est jusqu'à 50 fois plus rapide qu'une liste Python traditionnelle.

<br />
L'objet de tableau dans NumPy s'appelle ndarray, il fournit de nombreuses fonctions de soutien qui rendent le travail avec ndarray très facile. Les tableaux sont très fréquemment utilisés en science des données, où la vitesse et les ressources sont très importantes.

<br />
Ce qui rend les tableaux NumPy plus rapides que les listes : les tableaux NumPy sont stockés à un endroit continu en mémoire contrairement aux listes, de sorte que les processus peuvent y accéder et les manipuler très efficacement. Ce comportement est appelé la localité de référence. C'est la principale raison pour laquelle NumPy est plus rapide que les listes. Il est également optimisé pour fonctionner avec les dernières architectures de processeur.

Installation de NumPy
Pour installer NumPy, vous pouvez utiliser pip, l'installateur de paquets Python. Ouvrez votre terminal ou votre invite de commande et entrez la commande suivante :

Copy code
pip3 install numpy
Importation de NumPy
Il existe deux façons d'importer NumPy. Exemple de code :

python
Copy code
# cela importera l'ensemble du module NumPy.
import numpy as np
# cela importera toutes les classes, les objets, les variables, etc. du package NumPy   
from numpy import*
NumPy est généralement importé sous l'alias np.

<br />
alias : en Python, les alias sont un nom alternatif pour faire référence à la même chose.

<br />
Création de tableaux NumPy
L'objet de tableau dans NumPy s'appelle ndarray. Nous pouvons créer un objet ndarray NumPy en utilisant la fonction array(). Les tableaux NumPy peuvent être créés de plusieurs manières. Voici quelques-unes des méthodes les plus courantes :

Utilisation de la fonction numpy.array() pour créer un tableau à partir d'une liste/d'un tuple :
css
Copy code
a = np.array([1, 2, 3])
Utilisation de la fonction numpy.zeros() pour créer un tableau rempli de zéros :
bash
Copy code
b = np.zeros((2, 3))
Utilisation de la fonction numpy.ones() pour créer un tableau rempli de uns :
bash
Copy code
c = np.ones((2, 3))
Utilisation de la fonction numpy.random.randint() : renvoie un tableau d'entiers aléatoires entre les deux nombres donnés.
lua
Copy code
d = np.random.randint(0, 10)
Utilisation de la fonction numpy.random.rand() pour créer un tableau de valeurs aléatoires :
lua
Copy code
e = np.random.rand(2, 3)
Dimensions des tableaux NumPy