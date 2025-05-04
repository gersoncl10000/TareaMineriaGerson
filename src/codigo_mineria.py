import os
import sys
# Establezco el directorio de trabajo
os.chdir(r"C:")
sys.path.append(os.getcwd())
import pandas as pd
from FuncionesMineria import cuentaDistintos, analizar_variables_categoricas,atipicosAmissing, patron_perdidos,lm_stepwise, lm_forward, lm_backward,crear_data_modelo, validacion_cruzada_lm, glm_stepwise, glm_backward, glm_forward
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import statsmodels.api as sm
from sklearn.metrics import r2_score,accuracy_score, roc_auc_score,roc_curve
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Cargo el conjunto de datos desde el archivo Excel
datos = pd.read_excel(r"data\DatosEleccionesEspaña.xlsx")

# Elimino las variables objetivo que no han sido seleccionadas para el análisis
variables_no_utilizadas = ['Izda_Pct', 'Dcha_Pct', 'Otros_Pct', 'Izquierda', 'Derecha']
datos = datos.drop(columns=variables_no_utilizadas)

# Compruebo el tipo de variable asignado al importar los datos
datos.dtypes

# Variables numéricas que en realidad son cualitativas
numericasAcategoricas = ['CodigoProvincia']

# Se transforman a tipo string
for var in numericasAcategoricas:
    datos[var] = datos[var].astype(str)

# Verificación del cambio de tipo
datos[numericasAcategoricas].dtypes

# Número de observaciones y variables
print("Número de observaciones:", datos.shape[0])
print("Número de variables:", datos.shape[1])

# Número de valores distintos y tipos por variable
cuentaDistintos(datos)

# Distribución de frecuencias para variables cualitativas
analizar_variables_categoricas(datos)

# Cálculo de descriptivos extendidos para variables numéricas
numericas = datos.select_dtypes(include='number').columns.tolist()
descriptivos_num = datos[numericas].describe().T

# Añadimos más descriptivos a los anteriores
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)



# --- Corrección de errores numéricos y cualitativos ---

# Códigos numéricos erróneos a reemplazar
codigos_erroneos_numericos = [-99, 9999, 99999]

# Reemplazo en todas las variables numéricas
for col in datos.select_dtypes(include=['float64', 'int64']).columns:
    datos[col] = datos[col].replace(codigos_erroneos_numericos, np.nan)

# Reemplazo de todos los valores negativos en proporciones
variables_con_negativos = ['ForeignersPtge', 'Age_over65_pct']
for var in variables_con_negativos:
    datos.loc[datos[var] < 0, var] = np.nan

# Reemplazo de errores cualitativos
datos['Densidad'] = datos['Densidad'].replace('?', np.nan)

# --- Tratamiento de valores atípicos ---

variables_numericas = datos.select_dtypes(include=['float64', 'int64']).columns.tolist()

for var in variables_numericas:
    datos[var] = atipicosAmissing(datos[var])[0]

# --- Análisis y tratamiento de valores perdidos ---

# Visualización del patrón de missing
patron_perdidos(datos)

# Creación de la variable de proporción de missing por observación
datos['prop_missings'] = datos.isna().mean(axis=1)

# Eliminación de observaciones con más del 50% de datos perdidos
eliminar_observaciones = datos['prop_missings'].astype(float) > 0.5
datos = datos[~eliminar_observaciones]

# Eliminación de la variable auxiliar prop_missings
datos = datos.drop(columns=['prop_missings'])
datos = datos.dropna()
# --- Recodificación de categorías de baja frecuencia en ActividadPpal ---

# Agrupo 'Construccion' e 'Industria' como 'Otras'
datos['ActividadPpal'] = datos['ActividadPpal'].replace({'Construccion': 'Otras', 'Industria': 'Otras'})

# --- Verificación del porcentaje de missing final por variable ---

porcentaje_missing_final = datos.isna().mean() * 100
print(porcentaje_missing_final[porcentaje_missing_final > 0])

#-- MODELO DE REGRESION LINEAL ---

# Definición de variable objetivo y variables predictoras
y = datos['AbstentionPtge']
X = datos.drop(columns=['Name', 'AbstentionPtge'])  # Excluye identificador único

# Identificación de variables continuas y categóricas
var_cont = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
var_categ = X.select_dtypes(include='object').columns.tolist()

# División en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234567)

# Construcción de modelos clásicos
modeloStepAIC = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'AIC')
modeloStepBIC = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'BIC')
modeloBackAIC = lm_backward(y_train, x_train, var_cont, var_categ, [], 'AIC')
modeloBackBIC = lm_backward(y_train, x_train, var_cont, var_categ, [], 'BIC')
modeloForwAIC = lm_forward(y_train, x_train, var_cont, var_categ, [], 'AIC')
modeloForwBIC = lm_forward(y_train, x_train, var_cont, var_categ, [], 'BIC')

def evaluar_modelo(modelo, x_train, y_train, x_test, y_test):
    vars_usadas = modelo.model.exog_names

    datos_train = crear_data_modelo(x_train, var_cont, var_categ)
    datos_test = crear_data_modelo(x_test, var_cont, var_categ)

    datos_train = sm.add_constant(datos_train, has_constant='add')
    datos_test = sm.add_constant(datos_test, has_constant='add')

    datos_train = datos_train[vars_usadas]
    datos_test = datos_test[vars_usadas]

    pred_train = modelo.predict(datos_train)
    pred_test = modelo.predict(datos_test)

    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    n_parametros = len(vars_usadas)

    return r2_train, r2_test, n_parametros

# Evaluación de todos los modelos
resultados_modelos = {
    'Backward AIC': evaluar_modelo(modeloBackAIC['Modelo'], x_train, y_train, x_test, y_test),
    'Backward BIC': evaluar_modelo(modeloBackBIC['Modelo'], x_train, y_train, x_test, y_test),
    'Forward AIC': evaluar_modelo(modeloForwAIC['Modelo'], x_train, y_train, x_test, y_test),
    'Forward BIC': evaluar_modelo(modeloForwBIC['Modelo'], x_train, y_train, x_test, y_test),
    'Stepwise AIC': evaluar_modelo(modeloStepAIC['Modelo'], x_train, y_train, x_test, y_test),
    'Stepwise BIC': evaluar_modelo(modeloStepBIC['Modelo'], x_train, y_train, x_test, y_test),
}


tabla_resultados = pd.DataFrame([
    [metodo, *valores] for metodo, valores in resultados_modelos.items()
], columns=["Método", "R² Train", "R² Test", "Nº Parámetros"])

print(tabla_resultados)



#Selección Aleatoria de Variables



# Diccionario para almacenar fórmulas y variables seleccionadas
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}




# Número de iteraciones de selección aleatoria
n_iteraciones = 30

# Iterar creando subconjuntos aleatorios y aplicando selección stepwise BIC
for i in range(n_iteraciones):
    print(f"Iteración {i + 1}")
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size=0.3, random_state=1234567 + i)
    modelo = lm_stepwise(y_train2, x_train2, var_cont, var_categ, [], 'BIC')
    
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    formula = '+'.join(sorted(modelo['Modelo'].model.exog_names))
    variables_seleccionadas['Formula'].append(formula)

# Calcular la frecuencia de aparición de cada fórmula
frecuencias = Counter(variables_seleccionadas['Formula'])

# Convertir a DataFrame ordenado por frecuencia
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns=['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending=False).reset_index(drop=True)

# Mostrar las 5 fórmulas más frecuentes
print(frec_ordenada.head())


#d)	Selección del Modelo Ganador y Validación Cruzada



# Extraer las dos combinaciones más frecuentes del proceso aleatorio
formula1 = frec_ordenada['Formula'][0]
formula2 = frec_ordenada['Formula'][1]

# Extraer las variables correspondientes
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(formula1)]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(formula2)]

# Separar variables continuas y categóricas de cada fórmula
def separar_vars(diccionario):
    return diccionario['cont'], diccionario['categ'], diccionario['inter'] if 'inter' in diccionario else []

var_cont1, var_categ1, _ = separar_vars(modeloStepBIC['Variables'])
var_cont2, var_categ2, _ = separar_vars(var_1)
var_cont3, var_categ3, _ = separar_vars(var_2)

# Validación cruzada con 5 bloques y 20 repeticiones
results = pd.DataFrame(columns=['Rsquared', 'Resample', 'Modelo'])

for rep in range(20):
    r1 = validacion_cruzada_lm(5, x_train, y_train, var_cont1, var_categ1)
    r2 = validacion_cruzada_lm(5, x_train, y_train, var_cont2, var_categ2)
    r3 = validacion_cruzada_lm(5, x_train, y_train, var_cont3, var_categ3)

    rep_df = pd.DataFrame({
        'Rsquared': r1 + r2 + r3,
        'Resample': ['Rep' + str(rep + 1)] * 15,
        'Modelo': [1]*5 + [2]*5 + [3]*5
    })

    results = pd.concat([results, rep_df], ignore_index=True)

# Gráfico boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x='Modelo', y='Rsquared', data=results, palette="Set2")
plt.title("Comparación de R² por modelo (Validación cruzada)")
plt.xlabel("Modelo")
plt.ylabel("R²")
plt.xticks(ticks=[0, 1, 2], labels=["Stepwise BIC", "Aleatorio 1", "Aleatorio 2"])
plt.grid(True)
plt.tight_layout()
plt.show()

# R² promedio
print(results.groupby("Modelo")["Rsquared"].mean())


# Accedemos al modelo ganador
modelo_ganador = modeloStepBIC['Modelo']

# Mostrar todos los coeficientes del modelo
coeficientes = modelo_ganador.params.sort_values()
print(coeficientes)


# CONSTRUCCIÓN DEL MODELO DE REGRESIÓN LOGÍSTICA

# Crear la variable binaria objetivo (por ejemplo, alta abstención si supera la mediana)
umbral = datos['AbstentionPtge'].median()
datos['AbstencionAlta'] = (datos['AbstentionPtge'] > umbral).astype(int)

# Redefinir variables predictoras y objetivo para el modelo logístico
y = datos['AbstencionAlta']
X = datos.drop(columns=['Name', 'AbstentionPtge', 'AbstencionAlta'])

# Reidentificar variables continuas y categóricas
var_cont = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
var_categ = X.select_dtypes(include='object').columns.tolist()

# División de datos
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234567)


# Construcción de modelos logísticos con selección clásica
modeloLogStepAIC = glm_stepwise(y_train, x_train, var_cont, var_categ, [], 'AIC')
modeloLogStepBIC = glm_stepwise(y_train, x_train, var_cont, var_categ, [], 'BIC')
modeloLogBackAIC = glm_backward(y_train, x_train, var_cont, var_categ, [], 'AIC')
modeloLogBackBIC = glm_backward(y_train, x_train, var_cont, var_categ, [], 'BIC')
modeloLogForwAIC = glm_forward(y_train, x_train, var_cont, var_categ, [], 'AIC')
modeloLogForwBIC = glm_forward(y_train, x_train, var_cont, var_categ, [], 'BIC')


# Función para evaluar modelos logísticos clásicos
def evaluar_modelo_logistico(modelo_dict, x_train, y_train, x_test, y_test):
    modelo = modelo_dict['Modelo']
    
    # Crear matrices de diseño codificadas
    datos_train = crear_data_modelo(x_train, var_cont, var_categ)
    datos_test = crear_data_modelo(x_test, var_cont, var_categ)

    # Añadir constante
    datos_train = sm.add_constant(datos_train, has_constant='add')
    datos_test = sm.add_constant(datos_test, has_constant='add')

    # Extraer las columnas usadas por el modelo en el entrenamiento
    columnas_modelo = modelo.feature_names_in_

    # Seleccionar las columnas en el mismo orden
    datos_train = datos_train[columnas_modelo]
    datos_test = datos_test[columnas_modelo]

    # Predicción de probabilidades
    prob_train = modelo.predict_proba(datos_train)[:, 1]
    prob_test = modelo.predict_proba(datos_test)[:, 1]

    # Clasificación binaria con umbral 0.5
    pred_train = (prob_train >= 0.5).astype(int)
    pred_test = (prob_test >= 0.5).astype(int)

    # Cálculo de métricas
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)
    auc_test = roc_auc_score(y_test, prob_test)
    n_param = datos_train.shape[1]

    return acc_train, acc_test, auc_test, n_param

# Evaluación de los 6 modelos logísticos construidos
resultados_logisticos = {
    'Backward AIC': evaluar_modelo_logistico(modeloLogBackAIC, x_train, y_train, x_test, y_test),
    'Backward BIC': evaluar_modelo_logistico(modeloLogBackBIC, x_train, y_train, x_test, y_test),
    'Forward AIC': evaluar_modelo_logistico(modeloLogForwAIC, x_train, y_train, x_test, y_test),
    'Forward BIC': evaluar_modelo_logistico(modeloLogForwBIC, x_train, y_train, x_test, y_test),
    'Stepwise AIC': evaluar_modelo_logistico(modeloLogStepAIC, x_train, y_train, x_test, y_test),
    'Stepwise BIC': evaluar_modelo_logistico(modeloLogStepBIC, x_train, y_train, x_test, y_test),
}

# Convertir los resultados a un DataFrame
tabla_log = pd.DataFrame([
    [nombre, *valores] for nombre, valores in resultados_logisticos.items()
], columns=["Método", "Accuracy Train", "Accuracy Test", "AUC Test", "Nº Parámetros"])

# Mostrar tabla ordenada por AUC
print(tabla_log.sort_values("AUC Test", ascending=False))



#Selección Aleatoria de Variables (Regresión Logística)

# Diccionario para guardar fórmulas y variables seleccionadas
formulas_log = {"Formula": [], "Variables": []}

# Número de iteraciones
n_iter = 30

for i in range(n_iter):
    print(f"Iteración {i + 1}")
    x_train_sub, _, y_train_sub, _ = train_test_split(
        x_train, y_train, test_size=0.3, random_state=1000 + i
    )
    
    modelo_iter = glm_stepwise(y_train_sub, x_train_sub, var_cont, var_categ, [], 'BIC')
    
    # Extraer lista de variables usadas (continuas y categóricas)
    variables = modelo_iter["Variables"]
    todas_vars = variables["cont"] + variables["categ"]
    
    # Crear la fórmula ordenada como string (puedes añadir 'const' si quieres)
    formula = '+'.join(sorted(todas_vars + ['const']))
    
    formulas_log["Formula"].append(formula)
    formulas_log["Variables"].append(variables)

# Calcular frecuencias
frecuencia_formulas_log = Counter(formulas_log["Formula"])

# Convertir a DataFrame ordenado por frecuencia
df_frec_log = pd.DataFrame(frecuencia_formulas_log.items(), columns=["Formula", "Frecuencia"])
df_frec_log = df_frec_log.sort_values("Frecuencia", ascending=False).reset_index(drop=True)

# Mostrar las 5 fórmulas más frecuentes
print(df_frec_log.head())


#Selección del Modelo Ganador y Validación Cruzada (Regresión Logística)


# Definición de las variables empleadas en los modelos
vars_backbic = modeloLogBackBIC['Variables']

vars_top1 = {
    'cont': ['WomanPopulationPtge', 'Age_19_65_pct'],
    'categ': ['ActividadPpal', 'CCAA', 'CodigoProvincia', 'SameComAutonPtge']
}

vars_top2 = {
    'cont': ['WomanPopulationPtge', 'Age_19_65_pct'],
    'categ': ['ActividadPpal', 'CCAA', 'CodigoProvincia', 'DifComAutonPtge']
}

# Inicialización del DataFrame de resultados
resultados_val_log = pd.DataFrame(columns=['AUC', 'Modelo'])

# Validación cruzada: 5 bloques x 20 repeticiones
for rep in range(20):
    auc1 = validacion_cruzada_glm(5, x_train, y_train, vars_backbic['cont'], vars_backbic['categ'])
    auc2 = validacion_cruzada_glm(5, x_train, y_train, vars_top1['cont'], vars_top1['categ'])
    auc3 = validacion_cruzada_glm(5, x_train, y_train, vars_top2['cont'], vars_top2['categ'])

    resultados_val_log = pd.concat([
        resultados_val_log,
        pd.DataFrame({
            'AUC': auc1 + auc2 + auc3,
            'Modelo': ['Backward BIC']*5 + ['Aleatorio 1']*5 + ['Aleatorio 2']*5
        })
    ], ignore_index=True)

# Gráfico comparativo
plt.figure(figsize=(8, 5))
sns.boxplot(data=resultados_val_log, x='Modelo', y='AUC', palette="Set2")
plt.title("Comparación de modelos logísticos (AUC - Validación cruzada)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Promedio de AUC por modelo
print(resultados_val_log.groupby("Modelo")["AUC"].mean())


# Determinación del Punto de Corte Óptimo (Regresión Logística)    

# Extraer variables y modelo ganador
vars_usadas = modeloLogBackBIC['Variables']
modelo = modeloLogBackBIC['Modelo']

# Crear la matriz de diseño del conjunto de test (sin constante para scikit-learn)
X_test_log = crear_data_modelo(x_test, vars_usadas['cont'], vars_usadas['categ'])

# Alinear columnas con las utilizadas por el modelo
X_test_log = X_test_log[modelo.feature_names_in_]

# Verificar y binarizar y_test
y_test_bin = y_test.copy()
y_test_bin = y_test_bin.astype(int)

# Confirmar que es verdaderamente binario
assert set(np.unique(y_test_bin)) == {0, 1}, "La variable y_test no es binaria."

# Predecir probabilidades de clase positiva
probabilidades = modelo.predict_proba(X_test_log)[:, 1]

# Calcular curva ROC
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test_bin, probabilidades)

# Determinar punto óptimo en la curva (mínima distancia a (0,1))
distancias = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
indice_optimo = np.argmin(distancias)
punto_corte_optimo = thresholds[indice_optimo]

# Mostrar punto óptimo
print(f"Punto de corte óptimo: {punto_corte_optimo:.4f}")

# Graficar curva ROC con punto óptimo
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label='Curva ROC')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.scatter(fpr[indice_optimo], tpr[indice_optimo], color='red', label='Punto óptimo')
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC y punto de corte óptimo')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#Interpretación de Coeficientes del Modelo Ganador

# Acceder al modelo y sus coeficientes
modelo = modeloLogBackBIC['Modelo']
coeficientes = pd.Series(
    modelo.coef_[0],
    index=modelo.feature_names_in_
).sort_values()

# Mostrar los coeficientes ordenados
print(coeficientes)
