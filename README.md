# Deep-Learning-week2-.

# Análisis del perceptrón: efecto de pesos y bias

En este notebook implementamos una red neuronal **básica** (un solo perceptrón) que recibe dos
entradas: `horas_estudio` y `horas_sueno`. El cálculo que realiza la neurona es:

\[
z = w_1 \cdot x_1 + w_2 \cdot x_2 + b
\]

Luego aplica una función de **activación escalón**:

- Si \( z \ge 0 \) → salida \( y = 1 \)
- Si \( z < 0 \) → salida \( y = 0 \)

## Configuración 1: w = [0.8, 0.2], b = -2.0

- El peso de `horas_estudio` (0.8) es mayor que el de `horas_sueno` (0.2), por lo que la neurona
  considera el estudio más importante que el sueño.
- El bias = -2.0 implica que, si las horas de estudio y sueño son bajas, el valor de \( z \) será
  negativo y la salida tenderá a 0.
- Se observa que con pocas horas de estudio (1 o 2) es posible llegar a \( z \ge 0 \) solo si el
  valor de sueño también ayuda un poco. Es decir, el umbral no es extremadamente alto.

## Configuración 2: w = [1.0, 0.1], b = -3.0

- Ahora `horas_estudio` pesa aún más (1.0) y `horas_sueno` casi no contribuye (0.1).
- El bias = -3.0 hace que el umbral sea más difícil de superar: se necesita un valor alto de
  `horas_estudio` para que \( z \) pase de negativo a positivo.
- Al comparar las tablas, se ve que algunos casos que en la configuración 1 eran clasificados con
  salida 1 ahora pasan a 0. Es decir, el modelo se volvió más **estricto**.

## Conclusiones sobre el efecto de los parámetros

1. **Pesos (w)**  
   - Cuanto mayor es el peso de una característica, mayor influencia tiene esa entrada en el
     puntaje \( z \).  
   - Si aumentamos el peso de `horas_estudio`, la neurona se vuelve más sensible a cambios en esa
     variable: subir una hora de estudio tiene más impacto en la decisión final que subir una hora
     de sueño.

2. **Bias (b)**  
   - El bias desplaza el umbral de decisión. Un bias muy negativo hace más difícil alcanzar
     \( z \ge 0 \), por lo que el modelo tiende a predecir más ceros (clase negativa).  
   - Un bias más alto (menos negativo o incluso positivo) facilita que \( z \) sea mayor o igual a
     cero, aumentando la cantidad de unos (clase positiva).

3. **Comportamiento global**  
   - Modificar pesos y bias cambia la **frontera de decisión** del perceptrón, es decir, la línea
     que separa los casos que clasifica como 0 de los que clasifica como 1.  
   - En términos prácticos, esto permite ajustar qué consideramos suficiente "evidencia" en las
     entradas para tomar una decisión positiva.

En resumen, los pesos controlan **qué tan importante** es cada característica en la decisión, mientras
que el bias controla **qué tan difícil** es obtener una salida positiva. Juntos determinan el
comportamiento del perceptrón para todos los casos de prueba.

# ==========================================
# BLOQUE 7: GRÁFICA FRONTERA DE DECISIÓN
# (configuración 1)
# ==========================================

import matplotlib.pyplot as plt

# Volvemos a usar:
# test_data, w1 = [0.8, 0.2], b1 = -2.0

w1 = np.array([0.8, 0.2])
b1 = -2.0

X = test_data[['horas_estudio', 'horas_sueno']].values
z = X @ w1 + b1
y = (z >= 0).astype(int)

# Recta de frontera de decisión:
# z = 0 => w1*x1 + w2*x2 + b = 0
# => x2 = -(w1*x1 + b) / w2
x1_line = np.linspace(-1, 6, 100)
x2_line = -(w1[0]*x1_line + b1) / w1[1]

plt.figure(figsize=(6, 6))

# Puntos clasificados como 0 (rojo) y 1 (azul)
plt.scatter(
    test_data['horas_estudio'][y == 0],
    test_data['horas_sueno'][y == 0],
    c='red', label='Clase 0', edgecolor='k'
)
plt.scatter(
    test_data['horas_estudio'][y == 1],
    test_data['horas_sueno'][y == 1],
    c='blue', label='Clase 1', edgecolor='k'
)

# Línea de decisión
plt.plot(x1_line, x2_line, 'k--', label='Frontera de decisión (z = 0)')

plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.xticks(range(0, 6))
plt.yticks(range(0, 6))
plt.xlabel('Horas de estudio')
plt.ylabel('Horas de sueño')
plt.title('Perceptrón: frontera de decisión (configuración 1)')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
