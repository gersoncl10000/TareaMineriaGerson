
# Predicción de la Abstención Electoral en Municipios de España (2019)

Este proyecto ha sido desarrollado como parte del Máster en Big Data, Data Science e Inteligencia Artificial en la Universidad Complutense de Madrid. El objetivo es modelar y predecir la abstención electoral utilizando técnicas de minería de datos y regresión.

---

## 🎯 Objetivo del Estudio

Construir dos modelos predictivos sobre datos reales de resultados electorales en España a nivel municipal:

- **Regresión lineal** para predecir el porcentaje de abstención (`AbstentionPtge`).
- **Regresión logística** para clasificar si un municipio tiene alta abstención (`AbstencionAlta`, binaria).

---

## 📦 Fuente de Datos

El conjunto de datos se encuentra en el archivo:

```
data/DatosEleccionesEspaña.xlsx
```

Contiene indicadores socioeconómicos, demográficos y administrativos para todos los municipios de España en 2019.

---

## 🧪 Metodología

### 1. Preprocesamiento de Datos

- Conversión de tipos de variables.
- Detección y tratamiento de valores erróneos y atípicos.
- Eliminación de observaciones con más del 50% de `NaNs`.
- Agrupación de categorías raras.

### 2. Modelado Predictivo

- **Modelos utilizados**:
  - Regresión Lineal: `lm_stepwise`, `lm_backward`, `lm_forward`.
  - Regresión Logística: `glm_stepwise`, `glm_backward`, `glm_forward`.

- **Criterios de selección**:
  - Métricas de información: `AIC` y `BIC`.
  - Validación cruzada: 5 bloques × 20 repeticiones.

### 3. Evaluación y Comparación

- Métricas utilizadas:
  - Regresión lineal: R² (train/test).
  - Regresión logística: Accuracy y AUC.
- Determinación del punto óptimo de corte mediante la curva ROC.

---

## 📊 Resultados

### 🔵 Validación Cruzada (Regresión Lineal)

![Boxplot R²](informe/boxplot_r2_modelos.png)

> El modelo elegido fue el **Stepwise BIC**, con buena generalización y bajo número de parámetros.

### 🟢 Validación Cruzada (Regresión Logística)

![Boxplot AUC](informe/boxplot_auc_modelos.png)

> El modelo **Backward BIC** superó en AUC a las combinaciones aleatorias.

### 🔺 Curva ROC y Punto Óptimo

![Curva ROC](informe/curva_roc_logistica.png)

> Punto de corte óptimo: 0.4765

---

## 📁 Estructura del Repositorio

```
├── data/
│   └── DatosEleccionesEspaña.xlsx
├── informe/
│   ├── TAREA GERSON CASTILLO MINERIA DE DATOS.pdf
│   ├── boxplot_r2_modelos.png
│   ├── boxplot_auc_modelos.png
│   └── curva_roc_logistica.png
├── src/
│   └── codigo_mineria.py
├── librerias.txt
└── README.md
```

---

## ⚙️ Requisitos del Entorno

Instalar entorno y librerías especificadas en:

```
librerias.txt
```

Entorno recomendado: `Spyder` (Python 3.9).

---

## ▶️ Ejecución

Ejecutar el código principal desde:

```
src/codigo_mineria.py
```

---

## 👤 Autor

**Gerson Castillo**  
Correo: [gersoncl10000@outlook.com](mailto:gersoncl10000@outlook.com)

---

Proyecto académico. No se ha utilizado generación automática de código ni herramientas de IA.
