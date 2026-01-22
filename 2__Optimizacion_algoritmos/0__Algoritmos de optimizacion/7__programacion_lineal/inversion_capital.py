# Programación lineal entera (IV)
# Problema de Inversión de capital. Elección de la cartera de valores

# Se están considerando cuatro posibles inversiones:
# - La primera proporciona beneficios netos de 16,000 euros.
# - La segunda proporciona beneficios netos de 22,000 euros.
# - La tercera proporciona beneficios netos de 12,000 euros.
# - La cuarta proporciona beneficios netos de 8,000 euros.

# Cada inversión requiere una cantidad exacta de dinero en efectivo:
# - La primera requiere 5,000 euros.
# - La segunda requiere 7,000 euros.
# - La tercera requiere 4,000 euros.
# - La cuarta requiere 3,000 euros.

# Se dispone solamente de 14,000 euros para invertir.

# Objetivo:
# Determinar la combinación de inversiones que proporcione los máximos beneficios,
# respetando la restricción del capital disponible.

# Modelo matemático:
# Max z = 16x1 + 22x2 + 12x3 + 8x4
# sujeto a: 5x1 + 7x2 + 4x3 + 3x4 <= 14
# con xj ∈ {0,1}, j = 1,2,3,4

# Donde:
# xj = 1 si se elige la inversión j
# xj = 0 en otro caso
