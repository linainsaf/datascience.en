
# Régression linéaire simple 

## Un exemple simple 


```python
#géneration des données de l'exemple
import numpy as np
X=np.array([0,3,6,8])
Y=np.array([35,45,65,80])
```


```python
import matplotlib.pyplot as plt 
plt.plot(X,Y,'*')
plt.xlabel("Année d'expérience comme développeur | Variable explicative 'x' ")
plt.ylabel("Salaire | Variable cible 'y'")
plt.title("Nuage de point")
plt.savefig("./intuitive_scatter.png")
```


```python
def reg_plot(x,y,m):
    plt.scatter(x,y,c='blue',label="les données")
    plt.plot(x, m.predict(x.reshape(-1, 1)), color='red',label="droite de prédiction")
    plt.xlabel("Variable explicative 'x' ")
    plt.ylabel("Variable cible 'y'")
    plt.legend()
    return None 
```


```python
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X.reshape(-1, 1),Y)
reg_plot(X,Y,linear_model)
plt.savefig("./approche_intuitive.png")
```


![png](output_4_0.png)


## Un exemple avec plus de données 

On génère des données linéaires pour notre exemple avec la librairie numpy et les fonctions :  
- `arange()`: https://numpy.org/doc/stable/reference/generated/numpy.arange.html 
- `random.uniform()` : https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html 


```python
import numpy as np 
x=np.arange(75)
delta = np.random.uniform(-10,10, size=(75,))
y = 0.4 * x +3 + delta
```

visualisation rapide des données avec la fonction `plot()` de  la librairie matplotlib


```python
plt.plot(x,y,"*")
plt.xlabel("Variable explicative 'x' ")
plt.ylabel("Variable cible 'y'")
plt.title("Nuage de point")
plt.savefig("./intuitive_scatter_bis.png")
```


![png](output_9_0.png)


## Utilisation de la librairie `scikit-learn`
On importe la librairie scikit-learn afin d'aller chercher l'algorithme de régression sous forme d'une fonction. 


```python
from sklearn.linear_model import LinearRegression
```


```python
linear_model = LinearRegression()
```


```python
linear_model.fit(x.reshape(-1, 1),y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
reg_plot(x,y,linear_model)
plt.savefig("./prediction.png")
```


![png](output_14_0.png)


# Cas pratique : prediction du salaire en fonction des années d'expérience

Importer la librairie pandas, les data et afficher le début du Dataframe : 


```python
import pandas as pd 
df = pd.read_csv("https://data.princeton.edu/wws509/datasets/salary.dat",delim_whitespace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sx</th>
      <th>rk</th>
      <th>yr</th>
      <th>dg</th>
      <th>yd</th>
      <th>sl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>full</td>
      <td>25</td>
      <td>doctorate</td>
      <td>35</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>full</td>
      <td>13</td>
      <td>doctorate</td>
      <td>22</td>
      <td>35350</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>full</td>
      <td>10</td>
      <td>doctorate</td>
      <td>23</td>
      <td>28200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>full</td>
      <td>7</td>
      <td>doctorate</td>
      <td>27</td>
      <td>26775</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>full</td>
      <td>19</td>
      <td>masters</td>
      <td>30</td>
      <td>33696</td>
    </tr>
  </tbody>
</table>
</div>



On selectionne la variable a predire et de la variable explicative avec un masque Pandas de la façon suivante : 


```python
df=df[["yr","sl"]]
```

On va maintenant afficher le nuage de point afin de voir si cela peut avoir du sens de corréler ces deux variables 


```python
X=df.yr
Y=df.sl
plt.plot(X,Y,'*')
plt.xlabel("Année d'expérience | Variable explicative 'x' ")
plt.ylabel("Salaire | Variable cible 'y'")
plt.title("Salaire en fonction des années d'expérience")
```




    Text(0.5, 1.0, "Salaire en fonction des années d'expérience")




![png](output_21_1.png)


On importe maintenant l'estimateur LinearRegression pour faire un fit sur nos données. 

🚧 Attention à ne pas oublier la méthode .reshape(-1, 1) car votre variable explicative est en une dimension 🚧


```python
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
```


```python
linear_model.fit(np.array(X).reshape(-1, 1),np.array(Y))
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



Enfin on utilise la fonction reg_plot pour afficher notre droite de prédiction tel que : 


```python
reg_plot(np.array(X),np.array(Y),linear_model)
```


![png](output_26_0.png)


Puis les différents score : 


```python
from sklearn.metrics import mean_squared_error, r2_score
Y_pred = linear_model.predict(np.array(X).reshape(-1, 1))
print("Mean squared error: %.2f"
      % mean_squared_error(Y, Y_pred))
print("Root mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y, Y_pred)))
print("R square: %.2f"% r2_score(Y, Y_pred))
```

    Mean squared error: 17481710.59
    Root mean squared error: 4181.11
    R square: 0.49



```python

```
