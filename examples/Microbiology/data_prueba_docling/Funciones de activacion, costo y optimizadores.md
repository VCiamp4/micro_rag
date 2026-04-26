Procesado en: 8 segundos

## Funciones de activacion, costo y optimizadores Funciones de activación, costo y optimizadores para

## dummies

## Función de activación

- Es una función matemática que se le aplica a la salida (neta) de cada neurona para decir cuanto contribuye a la siguiente capa*
- Introduce la no linealidad
- Si yo tuviera una capa con función de activación lineal (o lo mismo, que salga neta derecho), y mi modelo no hace buenos cortes, entonces mi pensamiento sería agregar otra capa
- Sin embargo, si agrego capas lineales sigue siendo transformaciones lineales, ya que lo unico que hago es "multiplicar" los valores, por lo que sería igual a seguir teniendo una capa pero cambiando otras cosas (hiperparametros diferentes, mismo resultado pobre de no hacer buenos cortes)
- Pensa que lo que hace un ejemplo (que en realidad es un vector de valores numéricos) al pasar por la red es ser multiplicada por los pesos de la capa (que es una matriz en si)
- Entonces, un ejemplo al pasar por la red lo que hace es ser multiplicado por matrices en cada capa
- Si tengo tres capas lineales, multiplique tres veces
- Pero, las multiplicaciones con matrices lineales son asociativas, entonces la capacidad "de corte" de cada capa en realidad sigue siendo un corte lineal
- Las funciones de activación son
- Sigmoide =&gt; Explicación aqui
- tanh =&gt; Explicación aquí
- Softmax =&gt; Explicación aquí
- ReLU =&gt; Brevemente explicada aquí

## Función de costo

- Es la que uno usa para derivar para calcular el gradiente
- Por ejemplo, no es la misma si uso estocastico puro o por lotes (en esta, hay que promediar con el tamaño del lote)
- ¿Qué necesita una función de costo para ser una buena función de costo?
- Depende para que, se necesita que sea derivable. Pero como NOSOTROS SIEMPRE USAMOS GRADIENTE, DEBE SER DERIVABLE
- Tener valor positivo SIEMPRE
- Valer 0 cuando la función neta se acerca al valor correcto :)
- Alejarse de 0 (es decir, agrandar la groseridad del error) cuando la función esta mal

| Tipo de problema         | Función de costo típica        |
|--------------------------|--------------------------------|
| Clasificación binaria    | Binary Cross-Entropy Aquí      |
| Clasificación multiclase | Categorical Cross-Entropy Aquí |
| Regresión                | Mean Squared Error (MSE)       |

## Función de error

- Es la diferencia entre la respuesta que se quería y lo que se predijo

## Optimizador

- Es el algoritmo que ajusta los pesos de la red para minimizar la función de costo
- Sirve de guía para el aprendizaje modificando los parametros en la función del gradiente
- Si vez a la función de costo como el grafico en 3D donde hay un "pozo" al que se quiere llegar, el optimizador es la que le dice como moverse a "la pelota" para llegar a ese pozo* Ejemplos comunes:

| Optimizador              | Características                                              | Explicación         |
|--------------------------|--------------------------------------------------------------|---------------------|
| SGD (Stochastic Descent) | Simple, puede ser lento Clase 3 > descenso                   | Gradient Técnica de |
| Momentum                 | Acelera el aprendizaje acumulando gradientes Clase 7 >       | Momento             |
| RMSProp                  | Ajusta el paso según la magnitud de los gradientes Clase 7 > | RMSprop             |
| Adam                     | Combina Momentum + RMSProp, el más usado Clase 7 >ADAM       |                     |

## Relación entre función de activación y función de costo

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

| Función de costo ↓ / Activación →   | Sigmoid                                                                       | Tanh                                                             | ReLU                                                        | Softmax                                         | Lineal                                         |
|-------------------------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------|------------------------------------------------|
| Entropia cruzada binaria            | ✅ Sí- ambas producen valores en [0, 1] , interpretables como probabilidades. | ⚠ No ideal -salida en [-1,1], no representa bien probabilidades. | ❌ No - ReLU no está acotada; no sirve para probabilidades. | ❌ No - Softmax es para multiclase, no binaria. | ❌ No - salida sin límites, no probabilística. |

<!-- image -->

| Función de costo ↓ / Activación →   | Sigmoid                                                                             | Tanh                                                                  | ReLU                                                       | Softmax                                                                   | Lineal                                                   |
|-------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------|------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------|
| Entropia cruzada categorica         | ⚠ Posible pero rara - para 2 clases se puede usar sigmoid, pero no para multiclase. | ❌ No - salida en [-1, 1] , no tiene sentido como probabilidad.       | ❌ No - salida no normalizada, no representa distribución. | ✅ Sí - Softmax genera una distribución de probabilidad sobre las clases. | ❌ No -no limita ni normaliza las salidas.               |
| Error cuadrático medio              | ⚠ Se puede, pero no ideal -funciona, pero el gradiente es lento para clasificación. | ⚠ Sí, pero poco eficiente - históricamente usada, pero converge peor. | ⚠ Sí en regresión (no en clasificación) -puede saturar.    | ❌ No - Softmax con MSE no refleja bien las diferencias probabilísticas.  | ✅ Sí - ideal en regresión, salida lineal sin límites.   |
| Error                               | ⚠ Posible - pero menos sensible que MSE.                                            | ⚠ Sí, pero lenta convergencia.                                        | ✅ Sí (regresión) - útil si hay outliers.                  | ❌ No - Softmax con MAE no tiene sentido.                                 | ✅ Sí (regresión) - predicciones reales, sin saturación. |

## Resumen de cuando y donde usarla

| Problema                     | Capa de Salida             | Función de Costo             | Función de Activación (Salida)   |
|------------------------------|----------------------------|------------------------------|----------------------------------|
| Clasificación Binaria        | 1 neurona                  | Entropía Cruzada Binaria     | Sigmoid                          |
| Clasificación Multiclase     | N neuronas (una por clase) | Entropía Cruzada Categórica  | Softmax                          |
| Regresión                    | 1 o N neuronas             | Error Cuadrático Medio (MSE) | Lineal (Ninguna)                 |
| Regresión Robusta a Outliers | 1 o N neuronas             | Error Absoluto Medio (MAE)   | Lineal (Ninguna)                 |

Para las capas ocultas, usa ReLU (o Leaky ReLU / PReLU) en la gran mayoría de los casos.