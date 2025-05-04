# Proyecto de Minería de Datos y Modelización Predictiva

Este repositorio contiene el desarrollo completo del proyecto académico de la asignatura **Minería de Datos y Modelización Predictiva** del Máster en Big Data, Data Science e Inteligencia Artificial de la Universidad Complutense de Madrid.

---

## 🎯 Objetivo

El objetivo principal del proyecto es construir dos modelos predictivos sobre datos reales de resultados electorales en los municipios de España:

- 📉 Un modelo de **Regresión Lineal** para predecir el porcentaje de abstención electoral como variable continua.
- 🔍 Un modelo de **Regresión Logística** para clasificar si un municipio tiene alta o baja abstención (binario), usando como umbral la mediana.

---

## 📂 Estructura del Repositorio

```
TareaMineriaGerson/
│
├── data/                     # Contiene el archivo original con datos electorales
│   └── DatosEleccionesEspaña.xlsx
│
├── informe/                 # Informe académico del proyecto y gráficos utilizados
│   ├── TAREA GERSON CASTILLO MINERIA DE DATOS.pdf
│   ├── curva_roc_punto_optimo.png
│   ├── boxplot_auc_modelos.png
│   └── boxplot_r2_modelos.png
│
├── src/                     # Código principal del análisis
│   └── codigo_mineria.py
│
├── librerias.txt            # Dependencias necesarias para ejecutar el código
├── README.md                # Este archivo
└── .gitignore               # Archivos ignorados por git
```

---

## 🧠 Metodología

El análisis sigue una estructura rigurosa:

1. **Carga y exploración inicial** de datos.
2. **Corrección de errores** y valores atípicos.
3. **Tratamiento de valores perdidos** y recodificación.
4. **Modelado**:
   - **Regresión lineal múltiple** con selección de variables (`Stepwise`, `Backward`, `Forward`).
   - **Regresión logística** para clasificación binaria.
5. **Validación cruzada** con 20 repeticiones y 5 bloques.
6. **Evaluación de métricas**: R², AUC y punto óptimo ROC.
7. **Selección de modelo ganador y análisis de coeficientes.**

---

## 📊 Resultados Relevantes

### 🔷 Comparación de R² entre modelos lineales
![R2 modelos](informe/boxplot_r2_modelos.png)

El modelo **Stepwise BIC** obtuvo un R² promedio competitivo frente a modelos aleatorios seleccionados mediante submuestreo.

---

### 🔶 Comparación de AUC entre modelos logísticos
![AUC modelos](informe/boxplot_auc_modelos.png)

El modelo **Backward BIC** fue el mejor clasificador en validación cruzada, con un AUC promedio superior.

---

### 🟥 Curva ROC y punto de corte óptimo
![Curva ROC](informe/curva_roc_punto_optimo.png)

El punto óptimo fue determinado minimizando la distancia al punto (0,1), con un valor umbral de clasificación de **0.4765**.

---

## 📐 Fórmulas y técnicas utilizadas

- **R² (coeficiente de determinación):**  
  \( R^2 = 1 - rac{SS_{res}}{SS_{tot}} \)

- **Curva ROC y AUC:**  
  AUC (Área bajo la curva) mide la capacidad del modelo de distinguir entre clases.

- **Distancia al punto óptimo ROC:**  
  \( d = \sqrt{(1 - TPR)^2 + FPR^2} \)

- **Selección de variables:**  
  Mediante criterios AIC y BIC con procedimientos stepwise, forward y backward.

---

## ⚙️ Requisitos

Instalar las dependencias indicadas en `librerias.txt`. Recomendado crear un entorno virtual:

```bash
pip install -r librerias.txt
```

---

## 📑 Informe Académico

Consulta el informe completo en PDF en la carpeta [`informe/`](informe/TAREA%20GERSON%20CASTILLO%20MINERIA%20DE%20DATOS.pdf)

---

## ✍️ Autor

**Gerson Castillo López**  
Proyecto para la Universidad Complutense de Madrid  
Máster en Big Data, Data Science e Inteligencia Artificial

---