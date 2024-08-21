---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.12
  nbformat: 4
  nbformat_minor: 5
---

::: {#b6159270-7138-4192-9d84-225125bb3941 .cell .markdown}
# Fenotipagem de tomateiro resistente à requeima
:::

::: {#fb80969a-b54e-4389-af2b-872f0da7ea14 .cell .markdown}
## Introdução
:::

::: {#5c6aac25-689f-4d24-b357-3f25c70a2010 .cell .markdown}
A fenotipagem é o processo de avaliação de características mensuráveis,
como peso, formato dos frutos e estruturas vegetais. O uso de imagens
capturadas em diferentes regiões do espectro eletromagnético permite
estimar medidas essenciais, como altura, largura e número de folhas,
para monitorar o crescimento vegetativo.

A requeima, ou *mela*, é uma doença causada pelo oomiceto *Phytophthora
infestans*, que afeta culturas de tomate. Embora possa causar a perda
total da cultura, existem cultivares resistentes à doença. Experimentos
com diferentes variedades podem avaliar a severidade da requeima em
diversas cultivares, utilizando imagens de drone.

De acordo com o Código Internacional de Nomenclatura de Plantas
Cultivadas (**ICNCP**), uma cultivar é definida como um \"*conjunto de
plantas selecionado por atributos específicos ou combinação de
atributos*\". Dada a variedade de cultivares disponíveis, um experimento
foi conduzido com o uso de imagens de drone, que foram processadas e
analisadas com ferramentas de *machine learning* para economizar tempo e
recursos.

Uma câmera multiespectral **MicaSense**, equipada com cinco bandas
espectrais (RGB, infravermelho próximo (NIR) e Red Edge), foi utilizada
para capturar as imagens. Os índices de vegetação, calculados a partir
dessas bandas espectrais, foram processados e resultaram no *dataset*
utilizado para prever a severidade da requeima.
:::

::: {#619887d0-cf79-4919-b201-c7082393373d .cell .markdown}
## Análise preditiva
:::

::: {#56a78b9b-4b69-4944-b100-e5f1bd8401bc .cell .markdown}
### Importamos as bibliotecas
:::

::: {#84493c54-2d74-4c97-8e56-131220b09e40 .cell .code execution_count="113"}
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
:::

::: {#481c7009-2cbb-401a-aef0-2e0f8e5c2f14 .cell .markdown}
### Carregamos o dataset
:::

::: {#b572afef-bcbd-4df6-9b98-509061642a78 .cell .code execution_count="87"}
``` python
df = pd.read_csv('dataset_sintetico.csv')

X = df.drop(['id', 'Severidade'], axis = 1)
Y = df['Severidade']
```
:::

::: {#fa455e88-3409-45db-9d52-b070f0a02319 .cell .markdown}
### Extraímos uma amostra dos dados
:::

::: {#53105cc1-b171-42a6-8598-a63bf4d908b2 .cell .code execution_count="88"}
``` python
df.head()
```

::: {.output .execute_result execution_count="88"}
```{=html}
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
      <th>id</th>
      <th>NDVI_d22</th>
      <th>SAVI_d22</th>
      <th>GNDVI_d22</th>
      <th>MCARI1_d22</th>
      <th>SR_d22</th>
      <th>NDVI_d01</th>
      <th>SAVI_d01</th>
      <th>GNDVI_d01</th>
      <th>MCARI1_d01</th>
      <th>...</th>
      <th>SAVI_d15</th>
      <th>GNDVI_d15</th>
      <th>MCARI1_d15</th>
      <th>SR_d15</th>
      <th>NDVI_d08</th>
      <th>SAVI_d08</th>
      <th>GNDVI_d08</th>
      <th>MCARI1_d08</th>
      <th>SR_d08</th>
      <th>Severidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.806955</td>
      <td>1.210392</td>
      <td>0.705323</td>
      <td>19513.85630</td>
      <td>10.557411</td>
      <td>0.774359</td>
      <td>1.161514</td>
      <td>0.717367</td>
      <td>30041.91766</td>
      <td>...</td>
      <td>1.166604</td>
      <td>0.680269</td>
      <td>22731.09954</td>
      <td>8.380453</td>
      <td>0.725521</td>
      <td>1.088240</td>
      <td>0.669113</td>
      <td>16190.82038</td>
      <td>6.491860</td>
      <td>37.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>0.789403</td>
      <td>1.184062</td>
      <td>0.688353</td>
      <td>17859.92417</td>
      <td>9.527958</td>
      <td>0.757559</td>
      <td>1.136312</td>
      <td>0.706716</td>
      <td>26529.38283</td>
      <td>...</td>
      <td>1.106409</td>
      <td>0.653475</td>
      <td>17483.16681</td>
      <td>7.039555</td>
      <td>0.701538</td>
      <td>1.052262</td>
      <td>0.651048</td>
      <td>13607.31327</td>
      <td>5.889840</td>
      <td>46.07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>0.806006</td>
      <td>1.208966</td>
      <td>0.698522</td>
      <td>18325.86796</td>
      <td>10.450621</td>
      <td>0.776762</td>
      <td>1.165121</td>
      <td>0.712256</td>
      <td>35348.89095</td>
      <td>...</td>
      <td>1.136947</td>
      <td>0.663469</td>
      <td>18970.25945</td>
      <td>7.691801</td>
      <td>0.721819</td>
      <td>1.082688</td>
      <td>0.654506</td>
      <td>16312.51025</td>
      <td>6.385158</td>
      <td>38.38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>0.778408</td>
      <td>1.167566</td>
      <td>0.674224</td>
      <td>16130.88569</td>
      <td>9.135821</td>
      <td>0.755313</td>
      <td>1.132947</td>
      <td>0.697300</td>
      <td>30860.23050</td>
      <td>...</td>
      <td>1.097415</td>
      <td>0.648051</td>
      <td>15867.27075</td>
      <td>6.888080</td>
      <td>0.723709</td>
      <td>1.085521</td>
      <td>0.659776</td>
      <td>15024.84460</td>
      <td>6.503054</td>
      <td>38.15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46</td>
      <td>0.777937</td>
      <td>1.166863</td>
      <td>0.670604</td>
      <td>17315.57673</td>
      <td>8.797762</td>
      <td>0.765304</td>
      <td>1.147933</td>
      <td>0.699105</td>
      <td>30874.50847</td>
      <td>...</td>
      <td>1.165921</td>
      <td>0.678815</td>
      <td>23387.23253</td>
      <td>8.498031</td>
      <td>0.782553</td>
      <td>1.173796</td>
      <td>0.707183</td>
      <td>22071.41814</td>
      <td>8.726001</td>
      <td>2.91</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>
```
:::
:::

::: {#aadfd512-5e18-415a-8d56-33137cd7803a .cell .markdown}
### Funções para ajuste dos modelos
:::

::: {#937a1407-f344-40b0-8398-ae59b0ebab0f .cell .code execution_count="89"}
``` python
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data, X, y
```
:::

::: {#839c7e36-8f65-4002-a10f-8555762031e5 .cell .code execution_count="90"}
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
:::

::: {#bc67d650-e4de-4c9f-8a26-366150be69c0 .cell .code execution_count="91"}
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
:::

::: {#140b8a07-8b93-415f-b6b1-fd6743776bf9 .cell .code execution_count="92"}
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
:::

::: {#4dc5d0b0-e5d0-4eaa-ab59-86baad4c3294 .cell .code execution_count="93"}
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
:::

::: {#5bff90a1-7b95-4f84-9e4d-d80cd500e050 .cell .code execution_count="94"}
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
:::

::: {#86ba3a4e-46f1-429a-a577-68ecfce3928e .cell .code execution_count="116"}
``` python
# modelos = {"Linear": LinearRegression, "KNN": KNeighborsRegressor, "SVR": svr_regressor, "Decision Tree": DecisionTreeRegressor, "Random Forest": RandomForestRegressor, "HGB": HistGradientBoostingRegressor }
tunning = {"Linear": LinearRegression, "KNN": knn_regressor, "SVR": svr_regressor, "Decision Tree": decision_tree_regressor, "Random Forest": random_forest_regressor, "HGB": hist_gradient_boosting_regressor }
```
:::

::: {#e58be991-eb9e-43e6-80b4-737a1ad2b86c .cell .markdown}
### Separamos os dados de treinamento e teste
:::

::: {#1e870cc1-40b6-4f31-b32f-80105e223cf0 .cell .code execution_count="96"}
``` python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```
:::

::: {#4039a6a2-c2d4-40ce-b31f-602579fdbdfb .cell .markdown}
### Ajuste dos modelos sem pre-processamento ou seleção de \"features\"
:::

::: {#fc6c1054-1280-4129-a5ab-9426c9a80071 .cell .code execution_count="103"}
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

::: {.output .stream .stdout}
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
:::
:::

::: {#d2d531db-9158-47a8-b94b-01b1ba0aeb8f .cell .markdown}
### Função para seleção de \"features\"
:::

::: {#88f47847-f2e7-4bea-b27f-9a22ba18d8f0 .cell .code execution_count="111"}
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
:::

::: {#0e0d822d-788f-43ee-bbeb-2aa81d681445 .cell .markdown}
### Ajustes dos modelos com pré-processamento e seleção de \"features\"
:::

::: {#1c079ae3-6cb4-4bf2-a1da-c56911160fb7 .cell .code execution_count="115"}
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

::: {.output .stream .stdout}
    Model: Modelo Linear, Mean R2 Score: 0.7850270627975064
    Model: KNN, Mean R2 Score: 0.8590771136164085
    Model: SVR, Mean R2 Score: 0.7869646771349614
    Model: Decision Tree, Mean R2 Score: 0.7055605805910499
    Model: Random Forest, Mean R2 Score: 0.8320267115668851
    Model: HGB, Mean R2 Score: 0.3188519070274789
    =======================================
    Melhor modelo: KNN apresentou R2: 0.8590771136164085
:::
:::

::: {#33b6c940-4ea8-4222-92da-a87acde2de94 .cell .markdown}
# Avaliação do melhor modelo no conjunto de teste
:::

::: {#19f04f8d-d07a-485e-b3fc-a878cafa37b2 .cell .code execution_count="118"}
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

::: {.output .stream .stdout}
    Avaliação do modelo final com os dados de teste
    r2: 0.9623229358546539
    mse: 4.757336912601418
    mae: 3.79435
:::
:::
