# Predicción de la Abstención Electoral en los Municipios de España (2019)

Este repositorio presenta el desarrollo completo de un ejercicio práctico de **Minería de Datos y Modelización Predictiva** realizado por Gerson Castillo como parte del Máster en Big Data, Data Science e Inteligencia Artificial de la **Universidad Complutense de Madrid**.

El proyecto se centra en la **predicción del comportamiento electoral en España** utilizando datos reales de los municipios en las elecciones de 2019, abordando tanto modelos de regresión lineal como de regresión logística, y aplicando técnicas de selección clásica y aleatoria de variables.

---

## 📁 Estructura del Repositorio

```
TareaMineriaGerson/
│
├── data/
│   └── DatosEleccionesEspaña.xlsx       # Conjunto de datos original
│
├── informe/
│   ├── TAREA GERSON CASTILLO MINERIA DE DATOS.pdf  # Informe final del análisis
│   ├── boxplot_r2_modelos.png                       # Comparación modelos lineales
│   ├── boxplot_auc_modelos.png                      # Comparación modelos logísticos
│   └── curva_roc_punto_corte.png                    # Curva ROC con punto de corte óptimo
│
├── src/
│   └── codigo_mineria.py              # Código fuente principal del análisis
│
├── librerias.txt                      # Listado de paquetes Python utilizados
├── .gitignore
└── README.md
```

---

## 📌 Objetivo del Análisis

Este estudio busca predecir el **porcentaje de abstención electoral** en cada municipio de España, así como clasificar los municipios según si presentan una **alta abstención (superior al 30%)**, utilizando:

- **Modelos de regresión lineal** para predecir la variable continua `AbstentionPtge`.
- **Modelos de regresión logística** para la clasificación binaria `AbstencionAlta`.

Los datos incluyen variables demográficas, económicas, estructurales y regionales de cada municipio.

---

## 📊 Metodología

1. **Depuración de datos**: detección y corrección de errores, tratamiento de outliers y valores perdidos.
2. **Selección de variables**:
   - Métodos clásicos: Stepwise, Forward, Backward con criterios AIC y BIC.
   - Métodos aleatorios: submuestreo con selección stepwise BIC repetida.
3. **Evaluación de modelos**:
   - Validación cruzada con 5 bloques y 20 repeticiones.
   - Métricas: R², AUC, número de parámetros.
4. **Interpretación de resultados**: análisis de coeficientes, punto de corte óptimo para clasificación binaria.

---

## 📈 Resultados Destacados

### 📌 Comparación de Modelos de Regresión Lineal

![Boxplot R²](informe/boxplot_r2_modelos.png){ width=60% }

> El modelo ganador fue el **Stepwise BIC**, por su equilibrio entre rendimiento (R² ≈ 0.629) y simplicidad (27 variables).

### 📌 Comparación de Modelos Logísticos (AUC)

![Boxplot AUC](informe/boxplot_auc_modelos.png){ width=60% }

> El modelo **Backward BIC** obtuvo el mejor AUC promedio (≈ 0.8106), superando a modelos aleatorios.

### 📌 Punto de Corte Óptimo

![Curva ROC](informe/curva_roc_punto_corte.png){ width=60% }

> Punto de corte óptimo determinado: **0.4765**  
> (Balance ideal entre sensibilidad y especificidad en la curva ROC)

---

## 📐 Ejemplos de Interpretación de Coeficientes

- `Age_under19_Ptge`: a mayor porcentaje de jóvenes, mayor abstención esperada.
- `CCAA_Cataluña`: los municipios de Cataluña tienen una odds 6.5 veces mayor de alta abstención frente a la comunidad base.
- `WomanPopulationPtge`: mayor proporción de mujeres se asocia con menor probabilidad de abstención.

---

## 🧪 Fórmulas Estadísticas Relevantes

### Regresión Lineal

\[
Y = \beta_0 + \sum_{i=1}^p \beta_i X_i + \epsilon
\]

### Regresión Logística

\[
\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right) = \beta_0 + \sum_{i=1}^p \beta_i X_i
\]

---

## 🧠 Conclusiones

- Las variables sociodemográficas y regionales son altamente predictivas del comportamiento electoral.
- La validación cruzada permite justificar objetivamente la elección del modelo más robusto.
- El trabajo demuestra cómo aplicar buenas prácticas de modelado y evaluación en un caso real.

---

## 💻 Requisitos Técnicos

Para ejecutar el proyecto localmente:

```bash
# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # o .\venv\Scripts\activate en Windows

# Instalar dependencias
pip install -r librerias.txt
```

---

## 📚 Referencias

- **Universidad Complutense de Madrid**, Máster en Big Data, Data Science e IA.
- Documentación oficial de la asignatura: Minería de Datos y Modelización Predictiva.

---

Proyecto realizado de forma íntegra y original, respetando los principios metodológicos del curso.