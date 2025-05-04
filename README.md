
# 📊 Tarea de Minería de Datos y Modelización Predictiva

Este proyecto ha sido desarrollado como parte de la práctica evaluable del Máster en Big Data, Data Science e Inteligencia Artificial de la **Universidad Complutense de Madrid**, en el marco de la asignatura **Minería de Datos y Modelización Predictiva**.

El objetivo principal es construir dos modelos predictivos aplicados sobre un conjunto de datos reales de resultados electorales en los municipios de España:

- Un modelo de **regresión lineal** para predecir el porcentaje de abstención electoral.
- Un modelo de **regresión logística** para predecir la probabilidad de alta abstención.

---

## 📁 Estructura del repositorio

```
TareaMineriaGerson/
│
├── src/                         # Código Python principal
│   └── codigo_mineria.py       # Script principal (ordenado y modular)
│
├── informe/
│   └── TAREA GERSON CASTILLO MINERIA DE DATOS.pdf  # Informe final
│
├── TAREA GERSON CASTILLO MINERIA DE DATOS.docx     # Borrador de informe
├── librerias.txt                # Requisitos del entorno
├── .gitignore
└── README.md                   # Este archivo
```

---

## 🧠 Enfoque metodológico

Se ha seguido una metodología estructurada, replicando el flujo de trabajo visto en clase:

1. **Carga y limpieza de datos**
2. **Tratamiento de valores perdidos y atípicos**
3. **Revisión de tipos y codificación**
4. **Selección de variables con diferentes estrategias**
5. **Construcción de modelos clásicos (AIC/BIC/Forward/Backward/Stepwise)**
6. **Validación cruzada y comparación de modelos**
7. **Interpretación de coeficientes**
8. **Determinación del punto de corte óptimo (regresión logística)**

---

## 🧪 Modelos construidos

### 🔷 Regresión lineal
Se ha construido un modelo para predecir el porcentaje de abstención electoral (`AbstentionPtge`), evaluado con:

- Métricas: **R² en entrenamiento y test**
- Técnicas de selección: Stepwise, Forward y Backward con AIC y BIC
- Validación cruzada: 5 bloques x 20 repeticiones

### 🔶 Regresión logística
Se ha modelado la variable binaria `AbstencionAlta` (> mediana), utilizando:

- Métricas: **Accuracy** y **AUC**
- Comparación con selecciones aleatorias y técnicas clásicas
- Determinación del **punto de corte óptimo** a partir de la curva ROC

---

## 🧩 Tecnologías utilizadas

- Python 3.9+
- Pandas, NumPy, Matplotlib, Seaborn
- Statsmodels, Scikit-learn
- Entorno: **Spyder** (recomendado para ejecución)

---

## 🚫 Exclusiones

Este repositorio **no incluye** el archivo `FuncionesMineria.py` original de clase, ya que es material docente proporcionado por el profesor. Se asume que el usuario ya lo posee para la correcta ejecución.

---

## 📜 Créditos

Este trabajo ha sido desarrollado de forma íntegra y autónoma por:

**Gerson Castillo**  
Máster en Big Data, Data Science e Inteligencia Artificial  
Universidad Complutense de Madrid  
2025

---

## ✅ Estado del proyecto

✔️ Finalizado y entregado.  
🧪 Validado con múltiples ejecuciones y métricas.  
📄 Documentado y preparado para revisión académica.

---

## 🗂 Cómo ejecutar

1. Clona el repositorio:
   ```bash
   git clone https://github.com/gersoncl10000/TareaMineriaGerson.git
   cd TareaMineriaGerson
   ```

2. Instala las librerías necesarias:
   ```bash
   pip install -r librerias.txt
   ```

3. Ejecuta el script principal desde Spyder o cualquier entorno Python compatible.
