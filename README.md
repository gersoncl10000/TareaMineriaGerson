
# Análisis Predictivo de Abstención Electoral en Municipios de España

Este proyecto ha sido desarrollado como parte del Máster en Big Data, Data Science e Inteligencia Artificial de la Universidad Complutense de Madrid, dentro de la asignatura "Minería de Datos y Modelización Predictiva".

---

## 🎯 Objetivo del Estudio

El propósito principal de este trabajo es construir modelos predictivos sobre datos reales de resultados electorales en municipios de España. Se plantean dos enfoques:

- **Regresión Lineal:** para predecir el porcentaje de abstención electoral como variable continua.
- **Regresión Logística:** para predecir si un municipio tendrá una abstención alta o baja (superior a la mediana).

---

## 📊 Estructura del Proyecto

```
📁 TareaMineriaGerson/
├── data/                      <- Contiene el archivo `DatosEleccionesEspaña.xlsx`.
├── informe/                  <- Informe PDF y gráficos analíticos extraídos del estudio.
├── src/                      <- Código principal del análisis (`codigo_mineria.py`).
├── librerias.txt             <- Librerías necesarias para la ejecución.
├── README.md                 <- Este archivo.
└── .gitignore                <- Configuración para excluir archivos innecesarios.
```

---

## ⚙️ Metodología

1. **Carga y Limpieza de Datos:**
   - Eliminación de variables no utilizadas.
   - Tratamiento de outliers y valores perdidos.
   - Conversión de variables cualitativas codificadas numéricamente.

2. **Análisis Descriptivo:**
   - Frecuencias para variables categóricas.
   - Estadísticos extendidos (asimetría, curtosis, rango).

3. **Modelos Predictivos:**
   - **Regresión Lineal Múltiple:** selección de variables mediante métodos *stepwise*, *backward*, y *forward* (criterios AIC/BIC).
   - **Regresión Logística:** misma lógica aplicada a clasificación binaria.

4. **Validación Cruzada:**
   - 5 bloques x 20 repeticiones para comparar rendimiento de modelos.
   - Métricas: $R^2$ para regresión lineal, AUC para clasificación binaria.

---

## 📈 Resultados Destacados

### 📌 Comparación de Modelos de Regresión Lineal

<img src="informe/boxplot_r2_modelos.png" alt="Boxplot R²" width="500"/>

El modelo construido mediante **Stepwise con BIC** obtiene el mejor rendimiento promedio en $R^2$.

---

### 📌 Comparación de Modelos de Regresión Logística

<img src="informe/boxplot_auc_modelos.png" alt="Boxplot AUC" width="500"/>

El modelo **Backward BIC** obtiene el mejor AUC, indicando mayor capacidad predictiva de abstención alta.

---

### 📌 Curva ROC y Punto Óptimo

<img src="informe/curva_roc_punto_optimo.png" alt="Curva ROC" width="500"/>

El punto de corte óptimo se selecciona como el más cercano al vértice (0,1) de la curva ROC. Esto minimiza simultáneamente la tasa de falsos positivos y maximiza los verdaderos positivos.

---

## 🔍 Interpretación

El análisis muestra que las variables demográficas y socioeconómicas como `WomanPopulationPtge`, `Age_19_65_pct`, así como características regionales (`CCAA`, `ActividadPpal`, etc.), tienen impacto relevante en los patrones de abstención. La clasificación binaria aporta una visión complementaria y robusta al tratar el fenómeno como un problema de decisión.

---

## 📂 Datos

El archivo **`DatosEleccionesEspaña.xlsx`** incluye una muestra de municipios con información electoral y censal. Dada su naturaleza, no se incluye el archivo `.xlsx` completo por privacidad de la fuente original.

---

## 📌 Conclusiones

- Se han empleado metodologías estadísticas clásicas con validación robusta.
- Se obtuvo un modelo explicativo fiable para predecir abstención.
- Los resultados fueron consistentes entre los enfoques determinísticos y aleatorios.
- El proyecto ilustra una aplicación real de la minería de datos al análisis político-social.

---

## 🧠 Autor

**Gerson Castillo López**  
Estudiante del Máster en Big Data, UCM  
Repositorio GitHub: [gersoncl10000](https://github.com/gersoncl10000)

