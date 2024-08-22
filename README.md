# Machine Learning e fenotipagem multiespectral 

## Introdução

### Fenotipagem

A fenotipagem é o processo de avaliação de características mensuráveis,
como peso, formato dos frutos e estruturas vegetais. O uso de imagens
capturadas em diferentes regiões do espectro eletromagnético permite
estimar medidas essenciais, como altura, largura e número de folhas,
para monitorar o crescimento vegetativo.

### Requeima

A requeima, ou *mela*, é uma doença causada pelo oomiceto *Phytophthora
infestans*, que afeta culturas de tomate. Embora possa causar a perda
total da cultura, existem cultivares resistentes à doença. Experimentos
com diferentes variedades podem avaliar a severidade da requeima em
diversas cultivares, utilizando imagens de drone.

### Cultivares

De acordo com o Código Internacional de Nomenclatura de Plantas
Cultivadas (**ICNCP**), uma cultivar é definida como um \"*conjunto de
plantas selecionado por atributos específicos ou combinação de
atributos*\". Dada a variedade de cultivares disponíveis, um experimento
foi conduzido com o uso de imagens de drone, que foram processadas e
analisadas com ferramentas de *machine learning* para economizar tempo e
recursos.

### Índices de vegetação

Uma câmera multiespectral **MicaSense**, equipada com cinco bandas
espectrais (RGB, infravermelho próximo (NIR) e Red Edge), foi utilizada
para capturar as imagens. Os índices de vegetação, calculados a partir
dessas bandas espectrais, foram processados e resultaram no *dataset*
utilizado para prever a severidade da requeima.

## Análise preditiva

### Importamos as bibliotecas

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.svm import SVR 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.feature_selection import RFE 
```

### Carregamos o dataset

``` python
df = pd.read_csv('dataset_sintetico.csv')

X = df.drop(['id', 'Severidade'], axis = 1)
Y = df['Severidade']
```

### Extraímos uma amostra dos dados

``` python
df.head()
```
|   | id | NDVI_d22 | SAVI_d22 | GNDVI_d22 | MCARI1_d22  | SR_d22    | NDVI_d01 | SAVI_d01 | GNDVI_d01 | MCARI1_d01  | ... | SAVI_d15 | GNDVI_d15 | MCARI1_d15  | SR_d15   | NDVI_d08 | SAVI_d08 | GNDVI_d08 | MCARI1_d08  | SR_d08   | Severidade |
|---|----|----------|----------|-----------|-------------|-----------|----------|----------|-----------|-------------|-----|----------|-----------|-------------|----------|----------|----------|-----------|-------------|----------|------------|
| 0 | 2  | 0.806955 | 1.210392 | 0.705323  | 19513.85630 | 10.557411 | 0.774359 | 1.161514 | 0.717367  | 30041.91766 | ... | 1.166604 | 0.680269  | 22731.09954 | 8.380453 | 0.725521 | 1.088240 | 0.669113  | 16190.82038 | 6.491860 | 37.83      |
| 1 | 10 | 0.789403 | 1.184062 | 0.688353  | 17859.92417 | 9.527958  | 0.757559 | 1.136312 | 0.706716  | 26529.38283 | ... | 1.106409 | 0.653475  | 17483.16681 | 7.039555 | 0.701538 | 1.052262 | 0.651048  | 13607.31327 | 5.889840 | 46.07      |
| 2 | 12 | 0.806006 | 1.208966 | 0.698522  | 18325.86796 | 10.450621 | 0.776762 | 1.165121 | 0.712256  | 35348.89095 | ... | 1.136947 | 0.663469  | 18970.25945 | 7.691801 | 0.721819 | 1.082688 | 0.654506  | 16312.51025 | 6.385158 | 38.38      |
| 3 | 20 | 0.778408 | 1.167566 | 0.674224  | 16130.88569 | 9.135821  | 0.755313 | 1.132947 | 0.697300  | 30860.23050 | ... | 1.097415 | 0.648051  | 15867.27075 | 6.888080 | 0.723709 | 1.085521 | 0.659776  | 15024.84460 | 6.503054 | 38.15      |
| 4 | 46 | 0.777937 | 1.166863 | 0.670604  | 17315.57673 | 8.797762  | 0.765304 | 1.147933 | 0.699105  | 30874.50847 | ... | 1.165921 | 0.678815  | 23387.23253 | 8.498031 | 0.782553 | 1.173796 | 0.707183  | 22071.41814 | 8.726001 | 2.91       |


### Funções para ajuste dos modelos

``` python
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data, X, y
```

``` python
def knn_regressor():
    neigh_score = []
    for i in range(1, 30):
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(X_train, Y_train)
        pred = knn.predict(X_test)
        score = (mean_squared_error(Y_test, pred)**0.5)
        neigh_score.append((i, score))
    
    # Seleciona o k com o menor erro
    k = min(neigh_score, key=lambda x: x[1])[0]
    knn = KNeighborsRegressor(n_neighbors=k)
    return knn
```

``` python
def svr_regressor():
    svr_scores = []
    for c in [0.01, 0.1, 1, 10, 100]:
        svr = SVR(C=c)
        svr.fit(X_train, Y_train)
        pred = svr.predict(X_test)
        score = mean_squared_error(Y_test, pred)
        svr_scores.append((c, score))
    
    # Seleciona o C com o menor erro
    best_c = min(svr_scores, key=lambda x: x[1])[0]
    svr = SVR(C=best_c)
    return svr
```

``` python
def decision_tree_regressor():
    tree_scores = []
    for depth in range(1, 30):
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(X_train, Y_train)
        pred = tree.predict(X_test)
        score = mean_squared_error(Y_test, pred)
        tree_scores.append((depth, score))
    
    # Seleciona a profundidade com o menor erro
    best_depth = min(tree_scores, key=lambda x: x[1])[0]
    tree = DecisionTreeRegressor(max_depth=best_depth)
    return tree
```

``` python
def random_forest_regressor():
    forest_scores = []
    for n in range(10, 200, 10):
        forest = RandomForestRegressor(n_estimators=n)
        forest.fit(X_train, Y_train)
        pred = forest.predict(X_test)
        score = mean_squared_error(Y_test, pred)
        forest_scores.append((n, score))
    
    # Seleciona o número de árvores com o menor erro
    best_n = min(forest_scores, key=lambda x: x[1])[0]
    forest = RandomForestRegressor(n_estimators=best_n)
    return forest
```

``` python
def hist_gradient_boosting_regressor():
    hgb_scores = []
    for l_rate in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
        hgb = HistGradientBoostingRegressor(learning_rate=l_rate)
        hgb.fit(X_train, Y_train)
        pred = hgb.predict(X_test)
        score = mean_squared_error(Y_test, pred)
        hgb_scores.append((l_rate, score))
    
    # Seleciona a taxa de aprendizado com o menor erro
    best_rate = min(hgb_scores, key=lambda x: x[1])[0]
    hgb = HistGradientBoostingRegressor(learning_rate=best_rate)
    return hgb
```

``` python
# modelos = {"Linear": LinearRegression, "KNN": KNeighborsRegressor, "SVR": svr_regressor, "Decision Tree": DecisionTreeRegressor, "Random Forest": RandomForestRegressor, "HGB": HistGradientBoostingRegressor }
tunning = {"Linear": LinearRegression, "KNN": knn_regressor, "SVR": svr_regressor, "Decision Tree": decision_tree_regressor, "Random Forest": random_forest_regressor, "HGB": hist_gradient_boosting_regressor }
```

### Separamos os dados de treinamento e teste

``` python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```

### Ajuste dos modelos sem pre-processamento ou seleção de \"features\"

``` python
# modelo = LinearRegression()
for name in tunning:
    model = tunning[name]()
    model.fit(X_train, Y_train).predict(X_train)
    
    print(name)
    score = cross_val_score(model, X_train, Y_train, cv = 10, scoring='r2')
    print(np.mean(score))
    print("=======================================")
    
    # plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (name, roc_auc))
    # print(X.shape)()
```

    Modelo Linear
    0.8417215780173016
    =======================================
    KNN
    0.7544301598879973
    =======================================
    SVR
    0.7283238256537258
    =======================================
    Decision Tree
    0.6043537834888687
    =======================================
    Random Forest
    0.8442266652601512
    =======================================
    HGB
    0.7665020843799575
    =======================================

### Função para seleção de \"features\"

``` python

def best_k(X, Y):
    err = float('inf')
    best_val = 0
    
    for i in range(1, X.shape[1] + 1):
        model = LinearRegression()
        selector = RFE(model, n_features_to_select=i)
        X_new = selector.fit_transform(X, Y)
        
        scores = cross_val_score(model, X_new, Y, cv=10, scoring='neg_mean_squared_error')
        mse = -scores.mean()
        
        if mse < err:
            err = mse
            best_val = i
          
    return best_val
```

### Ajustes dos modelos com pré-processamento e seleção de \"features\"

``` python
# Padronizando os dados
X = StandardScaler().fit_transform(X)

# Seleção de features usando a função best_k
K = best_k(X, Y)
selector = RFE(LinearRegression(), n_features_to_select=K)
X_new = selector.fit_transform(X, Y)

# Divisão os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, train_size=0.7)

# Treinamento e validação cruzada para seleção do melhor modelo
best_score = -float('inf')
best_model_name = None
best_model = None

for name in tunning:
    model = tunning[name]()
    scores = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2')
    mean_score = scores.mean()
    
    print(f'Modelo: {name}, Média R2: {mean_score}')
    
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_model = model

print("=======================================")
print(f"Melhor modelo: {best_model_name} apresentou R2: {best_score}")
```

    Model: Modelo Linear, Mean R2 Score: 0.7850270627975064
    Model: KNN, Mean R2 Score: 0.8590771136164085
    Model: SVR, Mean R2 Score: 0.7869646771349614
    Model: Decision Tree, Mean R2 Score: 0.7055605805910499
    Model: Random Forest, Mean R2 Score: 0.8320267115668851
    Model: HGB, Mean R2 Score: 0.3188519070274789
    =======================================
    Melhor modelo: KNN apresentou R2: 0.8590771136164085

# Avaliação do melhor modelo no conjunto de teste

``` python
best_model.fit(X_train, Y_train)
y_pred = best_model.predict(X_test)

r2 = r2_score(Y_test, y_pred)
mse = (mean_squared_error(Y_test, y_pred))**0.5
mae = mean_absolute_error(Y_test, y_pred)

print("Avaliação do modelo final com os dados de teste")
print('r2:', r2)
print('mse:', mse)
print('mae:', mae)
```

    Avaliação do modelo final com os dados de teste
    r2: 0.9623229358546539
    mse: 4.757336912601418
    mae: 3.79435

