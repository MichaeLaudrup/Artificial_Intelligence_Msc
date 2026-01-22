# Optimización de Algoritmos en Python

Este repositorio contiene una colección de implementaciones de algoritmos clásicos y estructuras de datos en Python, con un enfoque en la optimización y el análisis de eficiencia.

## 📋 Índice
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Características](#características)
- [Requisitos](#requisitos)
- [Uso](#uso)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## 🚀 Instalación

### 1. Crear entorno virtual
```bash
python -m venv venv
```

### 2. Activar entorno virtual
```bash
.\venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Registrar entorno virtual en Jupyter Notebook
```bash
python -m ipykernel install --user --name venv --display-name "Python (venv)"
```

### 5. Iniciar Jupyter Notebook
```bash
jupyter notebook
```

## 📁 Estructura del Proyecto

El proyecto está organizado en tres directorios principales:

### 📁 Básicos
Implementaciones de algoritmos fundamentales:
- `mcd_euclides.py`: Implementación del algoritmo de Euclides para el Máximo Común Divisor (MCD)
  - Incluye versión básica (con restas) y optimizada (con división)
  - Comparación de rendimiento entre ambas implementaciones
- `list_common_elements.py`: Búsqueda de elementos comunes en listas usando list comprehension
- `square_root.py`: Cálculo de raíz cuadrada

### 📁 Clásicos
Algoritmos clásicos de la ciencia de la computación:
- `fibonacci.py`: Implementación de la secuencia de Fibonacci

### 📁 Ordenación
Algoritmos de ordenamiento:
- `quick_sort.py`: Implementación del algoritmo QuickSort

## ✨ Características

- ⚡ Implementaciones optimizadas
- 📊 Análisis de rendimiento incluido en algunos algoritmos
- 🔍 Comparativas entre diferentes versiones de algoritmos
- 📝 Código comentado para mejor comprensión

## 📋 Requisitos

- Python 3.x
- NumPy (para algunas implementaciones)

## 💻 Uso

Cada archivo puede ejecutarse de manera independiente. Por ejemplo:

```python
python basicos/mcd_euclides.py
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, asegúrate de:
1. Mantener el estilo de código consistente
2. Incluir comentarios explicativos
3. Agregar análisis de rendimiento cuando sea relevante

## 📄 Licencia

Este proyecto está disponible como código abierto.
