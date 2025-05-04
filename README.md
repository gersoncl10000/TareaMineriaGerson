
# Análisis Predictivo de la Abstención Electoral en Municipios de España

**Autor**: Gerson Castillo  
**Máster**: Big Data, Data Science e Inteligencia Artificial  
**Universidad Complutense de Madrid**  
**Asignatura**: Minería de Datos y Modelización Predictiva  

---

## Objetivo del Estudio

Este proyecto se desarrolla en el marco de una tarea evaluable con el objetivo de **predecir el nivel de abstención electoral** en municipios españoles. Se plantea un enfoque riguroso basado en técnicas de regresión lineal y logística, aplicando criterios de selección de variables, tratamiento de datos reales y validación cruzada exhaustiva.

El análisis se divide en dos partes:

- **Regresión lineal** para predecir el porcentaje de abstención (`AbstentionPtge`) como variable continua.
- **Regresión logística** para predecir la probabilidad de alta abstención (variable binaria derivada del umbral mediana).

---

## Estructura del Proyecto

```
TareaMineriaGerson/
│
├── data/
│   └── DatosEleccionesEspaña.xlsx
│
├── src/
│   └── codigo_mineria.py
│
├── informe/
│   ├── TAREA GERSON CASTILLO MINERIA DE DATOS.pdf
│   ├── boxplot_r2_modelos.png
│   ├── curva_roc_punto_optimo.png
│   └── validacion_cruzada_auc.png
│
└── README.md
```

---

## Metodología Aplicada

### Preprocesamiento

- Imputación y eliminación de datos perdidos
- Recodificación de variables cualitativas
- Tratamiento de valores atípicos mediante reglas estadísticas
- Estandarización de nombres y formatos

### Modelos Construidos

#### 1. **Regresión Lineal Múltiple**

Se evaluaron seis modelos clásicos de selección de variables:

- `Stepwise` con criterios AIC y BIC
- `Forward` y `Backward` con AIC y BIC

Se incorporó además una estrategia aleatoria con selección repetida de subconjuntos. La fórmula con mayor frecuencia fue validada con:

```math
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
```

#### 2. **Regresión Logística Binaria**

La variable binaria `AbstencionAlta` se define como:

```math
AbstencionAlta = 
\begin{cases}
1 & \text{si } AbstentionPtge > \text{mediana} \\
0 & \text{en caso contrario}
\end{cases}
```

Modelos evaluados: `Stepwise`, `Forward`, `Backward` con AIC/BIC y selección aleatoria.

---

## Validación Cruzada

Se utilizó validación cruzada de 5 bloques repetida 20 veces.

### Resultados Lineales

<div align="center">
  <img src="informe/boxplot_r2_modelos.png" width="500"/>
</div>

- **Modelo Ganador**: Stepwise BIC
- **R² Promedio**: 0.53
- **Variables significativas**: población femenina, proporción de mayores de 65, densidad

---

### Resultados Logísticos

<div align="center">
  <img src="informe/validacion_cruzada_auc.png" width="500"/>
</div>

- **Modelo Ganador**: Backward BIC
- **AUC promedio**: 0.79
- Variables clave: `ActividadPpal`, `WomanPopulationPtge`, `SameComAutonPtge`

---

### Punto de Corte Óptimo

<div align="center">
  <img src="informe/curva_roc_punto_optimo.png" width="500"/>
</div>

- **Punto de corte óptimo**: 0.4765
- Determinado mediante la distancia mínima al punto ideal (0,1)

---

## Conclusiones

- Se demuestra una **relación estructural y regional** en los patrones de abstención.
- La combinación de técnicas de modelado clásicas con validación aleatoria mejora la robustez de los resultados.
- Se propone el uso de este tipo de modelización como base para **sistemas de alerta temprana electoral**.

---

## Contacto

- **GitHub**: [gersoncl10000](https://github.com/gersoncl10000/TareaMineriaGerson)
- **Email**: gersoncl10000@outlook.com

