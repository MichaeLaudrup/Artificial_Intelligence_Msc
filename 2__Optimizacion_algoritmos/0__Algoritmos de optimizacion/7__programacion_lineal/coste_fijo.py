# Programación lineal entera (V)
# Problema de coste fijo

# Tres compañías telefónicas me ofrecen sus servicios con diferentes condiciones:
# - MaBell: tarifa fija mensual de 16€, más 0.25 céntimos por minuto.
# - PaBell: tarifa fija mensual de 25€, pero reduce el coste por minuto a 0.21 céntimos.
# - BabyBell: tarifa fija mensual de 18€, y un coste por minuto de 0.22 céntimos.

# Importante: cada compañía solo cobra la tarifa fija si efectivamente se utiliza (es decir, si se realizan llamadas a través de ella).

# Se realizan en promedio 200 minutos de llamadas al mes, los cuales pueden repartirse libremente entre las tres compañías.

# Objetivo:
# Determinar cómo distribuir los 200 minutos entre las tres compañías para minimizar el coste total mensual de teléfono.

# Variables:
# x1 = minutos de llamadas al mes con MaBell
# x2 = minutos de llamadas al mes con PaBell
# x3 = minutos de llamadas al mes con BabyBell
# y1 = 1 si se usa MaBell (x1 > 0), 0 en otro caso
# y2 = 1 si se usa PaBell (x2 > 0), 0 en otro caso
# y3 = 1 si se usa BabyBell (x3 > 0), 0 en otro caso

# Modelo matemático:
# Min z = 0.25x1 + 0.21x2 + 0.22x3 + 16y1 + 25y2 + 18y3
# sujeto a:
# x1 + x2 + x3 = 200
# x1 <= 200y1
# x2 <= 200y2
# x3 <= 200y3
# y1, y2, y3 ∈ {0, 1}
