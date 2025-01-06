# **Projeto de Predição de Digitos**
## **Descrição do Projeto**
Este projeto tem como objetivo a criação de um modelo preditivo para classificar dígitos manuscritos, utilizando aprendizado de máquina e algoritmos de classificação. O modelo foi desenvolvido para analisar imagens de dígito.

### Dataset
O conjunto de dados utilizado contém imagens de dígitos manuscritos, representadas em formato vetorial. Cada linha no dataset representa uma imagem transformada em um vetor de características, enquanto a coluna Classification indica a classe do dígito correspondente (de 0 a 9).

### Estrutura do Dataset
- Features (Colunas): Cada coluna representa um pixel da imagem em escala de cinza, transformado em um valor numérico. Esse formato permite que o modelo capture a estrutura visual dos dígitos.
- Classification: Coluna-alvo que contém os rótulos das classes (0-9) indicando qual dígito está representado.
Essa estrutura foi projetada para facilitar o treinamento de modelos de aprendizado supervisionado, permitindo a extração de padrões relevantes para a classificação.

## Tecnologias Utilizadas
Python: Linguagem de programação principal do projeto.
Pandas: Para manipulação e análise de dados.
Seaborn e Matplotlib: Para visualização de dados e análise gráfica.
Scikit-learn: Para treinamento e avaliação do modelo de aprendizado de máquina.
RandomForestClassifier: Algoritmo de aprendizado supervisionado utilizado para classificação.

## Metodologia
**Pré-processamento de Dados:**
- Carregamento e análise inicial do dataset.
- Separação em dados de treino e teste.

**Treinamento do Modelo:**
- Utilização do algoritmo Random Forest para criar um modelo preditivo.
- Ajuste de hiperparâmetros para maximizar a performance.

**Avaliação do Modelo:**
- Métricas de avaliação como acurácia, precisão, recall, F1-score e matriz de confusão.
- Visualização de resultados e análise de erros.

**Otimização:**
- Aplicação de GridSearchCV para encontrar os melhores parâmetros do modelo.

## Script Executável
python:

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar o dataset
df = pd.read_csv("digit_prediction_project.csv")

# Separar as features e o alvo
X = df.drop("Classification", axis=1)
y = df["Classification"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = RandomForestClassifier(n_estimators=150, max_depth=50, max_features="log2", random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
print("Acc:", accuracy_score(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))
```

## Resultados
O modelo alcançou uma alta acurácia na classificação dos dígitos manuscritos.
Métricas adicionais como Sensibilidade, Especificidade e F-Score foram calculadas para avaliar o desempenho geral.
Gráficos e matrizes de confusão foram utilizados para interpretar os resultados de forma visual.
---

## Contribuição:
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

--- 
Este projeto demonstra a aplicação prática de algoritmos de aprendizado de máquina em problemas de classificação, sendo uma excelente introdução ao uso de Random Forest para tarefas de reconhecimento de padrões.
