
# Análisis Predictivo de Abstención Electoral en Municipios de España

**Autor**: Gerson Castillo  
**Máster**: Big Data, Data Science e Inteligencia Artificial  
**Asignatura**: Minería de Datos y Modelización Predictiva  
**Universidad**: Universidad Complutense de Madrid  

---

## 📌 Objetivo del Estudio

Este proyecto tiene como objetivo principal **analizar los patrones de abstención electoral** en municipios españoles mediante técnicas de minería de datos. Se han desarrollado dos modelos predictivos:

- **Modelo de Regresión Lineal**: para predecir el porcentaje de abstención (`AbstentionPtge`) como una variable continua.
- **Modelo de Regresión Logística**: para clasificar si un municipio presenta alta abstención (`AbstencionAlta`) según su mediana.

El análisis se llevó a cabo **estrictamente siguiendo las metodologías académicas enseñadas en clase**, evitando el uso de herramientas de generación automática de código.

---

## 🧪 Metodología

### 1. Preprocesamiento y Limpieza de Datos

- Revisión y corrección de tipos de variables (e.g., `CodigoProvincia` transformado a categórica).
- Tratamiento de valores erróneos (`-99`, `9999`, `?`, etc.) y atípicos.
- Eliminación de observaciones con más del 50% de datos perdidos.
- Agrupación de categorías con baja frecuencia.

### 2. Modelado Predictivo

#### 📈 Regresión Lineal

Se construyeron modelos con:
- **Stepwise**, **Forward** y **Backward** según criterios AIC y BIC.
- Selección aleatoria de subconjuntos de variables (`n=30`) con validación cruzada repetida (5 bloques × 20 repeticiones).

**Mejor modelo**: Stepwise BIC (validado por estabilidad de R² y frecuencia de aparición).

#### 📊 Regresión Logística

Se utilizó el mismo enfoque con regresión logística:
- Modelos evaluados mediante métricas: **Accuracy** y **AUC**.
- Se seleccionó el modelo **Backward BIC** por su mayor robustez media en AUC tras validación cruzada.

---

## 🔍 Fórmulas Utilizadas

- **Regresión lineal múltiple**:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p
$$

- **Regresión logística**:

$$
P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)}}
$$

- **Punto de corte óptimo** (mínima distancia a (0,1) en la curva ROC):

$$
\text{distancia} = \sqrt{(1 - \text{TPR})^2 + \text{FPR}^2}
$$

---

## 📊 Resultados Gráficos

### R² en validación cruzada de modelos lineales

<div align="center">
  <img src="informe/boxplot_r2_modelos.png" width="600px">
</div>

### Comparación AUC en modelos logísticos

<div align="center">
  <img src="informe/boxplot_auc_modelos.png" width="600px">
</div>

### Curva ROC y punto de corte óptimo

<div align="center">
  <img src="informe/curva_roc_punto_optimo.png" width="600px">
</div>

---

## ✅ Conclusiones

- El modelo lineal seleccionado (Stepwise BIC) presentó buena capacidad predictiva con pocos parámetros.
- En clasificación, el modelo logístico **Backward BIC** logró un equilibrio entre simplicidad y precisión, con un **AUC promedio superior al resto**.
- Se evidenció la importancia de variables como `ActividadPpal`, `CCAA` y `WomanPopulationPtge`.

Este trabajo evidencia la utilidad de la minería de datos aplicada al análisis político-social a nivel territorial.

---

## 📁 Estructura del Repositorio

```
├── data/                  <- Datos originales (.xlsx)
├── informe/               <- Informe académico y gráficos extraídos
│   ├── *.png              <- Gráficos usados en el README
├── src/                   <- Código principal de análisis en Python
├── README.md              <- Este documento
```

---

## 📘 Referencias

- Documentación y PDFs oficiales del curso
- Ejercicios de clase y guías metodológicas proporcionadas por los docentes
