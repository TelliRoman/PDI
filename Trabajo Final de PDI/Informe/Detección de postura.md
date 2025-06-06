Existen ciertos indicadores que permiten detectar una buena o mala postura durante el uso de una computadora, suponiendo que la dirección de la webcam es perpendicular al plano de la pantalla y que tanto la computadora como el usuario se encuentran ubicados sobre superficies paralelas al suelo. Sin embargo, estos indicadores son subjetivos, porque dependen de los hábitos del usuario, tales como el ángulo de apertura de la computadora portátil, la cercanía a la cámara, la altura del escritorio o la altura de la silla. Por lo tanto, tomando como referencia distintos estudios **[falta referenciar]**, se llegó a la conclusión de que los pasos para implementar un algoritmo que permita la detección de la postura de un usuario mientras utiliza una computadora son los siguientes:
1. Capturar video en tiempo real desde la cámara de la computadora y extraer la posición de las distintas partes del cuerpo.
2. Realizar un tratamiento de los datos para obtener valores generalizados que resulten en una mayor robustez del algoritmo.
3. Entrenar un modelo de inteligencia artificial en base a patrones provistos por el usuario que permita clasificar la calidad de la postura.
4. Clasificar la postura en tiempo real utilizando el modelo previamente entrenado.
5. Mostrar los resultados en pantalla.
### Captura de video y extracción de la posición de las partes del cuerpo
Para capturar la imagen de la cámara de la computadora se utiliza la biblioteca OpenCV. Luego, cada fotograma se procesa con la biblioteca Mediapipe para detectar las posiciones en el espacio de las distintas partes del cuerpo del usuario.

Mediapipe genera un modelo con 33 *keypoints*, y cada *keypoint* tiene 4 valores asociados: uno para cada coordenada del espacio ($x$, $y$, $z$) y uno que representa su visibilidad.

![[Keypoints mediapipe.png]]

Referencia de la foto:
https://www.researchgate.net/publication/376877613_A_Machine_Learning_App_for_Monitoring_Physical_Therapy_at_Home?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoiX2RpcmVjdCJ9fQ
### Tratamiento de los datos
La cámara de la computadora no alcanza a capturar el cuerpo completo del usuario, debido a que este se encuentra generalmente sentado. Por lo tanto, no se detectan todos los *keypoints*. De todas maneras, para una detección primaria de la postura, es suficiente analizar los *keypoints* que se encuentran por encima del torso, que son los siguientes:

| Parte del cuerpo | `mp_pose.PoseLandmark`            | Índice en el gráfico de Mediapipe | Índice en el arreglo filtrado |
| ---------------- | --------------------------------- | --------------------------------- | ----------------------------- |
| Nariz            | `NOSE`                            | 0                                 | 0                             |
| Ojos             | `LEFT_EYE`, `RIGHT_EYE`           | 2, 5                              | 1, 2                          |
| Orejas           | `LEFT_EAR`, `RIGHT_EAR`           | 7, 8                              | 3, 4                          |
| Boca             | `MOUTH_LEFT`, `MOUTH_RIGHT`       | 9, 10                             | 5, 6                          |
| Hombros          | `LEFT_SHOULDER`, `RIGHT_SHOULDER` | 11, 12                            | 7, 8                          |

De estos *keypoints* se extraen las componentes $(x,y,z)$. Por lo tanto, se define la posición  $X_k\in\mathbb{R}^3$ de un *keypoint* $k$ en el arreglo filtrado como el vector formado por sus coordenadas tridimensionales $(x_k,y_k,z_k)$.

Las posiciones de los *keypoints* de cada patrón son normalizadas para poder detectar la postura del usuario incluso si se desplaza horizontal o verticalmente, o si cambia su distancia con respecto a la cámara. Para ello, se toma como referencia la posición del punto medio $P_{med}$ entre la posición de los hombros izquierdo $X_7$ y derecho $X_8$:
$$
P_{med} = \frac{X_7 + X_8}{2}
$$
Luego, se calcula posición relativa $X´_k$ de cada *keypoint* $k$ con respecto al punto de referencia $P_{med}$:
$$
X'_k = X_k - P_{med}
$$

Por último, se toma como referencia la norma entre los hombros izquierdo $X_7$ y derecho $X_8$,
$$
D = ||X_8 - X_7||
$$
para así obtener la posición normalizada $X''_k$ de cada *keypoint*:
$$
X''_k = \frac{X'_k}{D}
$$

### Entrenamiento del modelo clasificador
Se utiliza un modelo de *gradient boosting* para la clasificación de la postura del usuario. Para su entrenamiento, se realiza primero una recopilación de patrones provista por el usuario:
- Inicialmente, el usuario debe ubicarse con una postura correcta frente a la cámara, haciendo movimientos leves para un entrenamiento generalizado. El programa captura los *keyframes* de 250 fotogramas para esta clase.
- Luego, el usuario debe ubicarse con una postura incorrecta frente a la cámara, haciendo movimientos y mostrando diferentes posturas incorrectas para un entrenamiento generalizado. El programa captura los *keyframes* de 250 fotogramas para esta clase.

Los *keyframes* son normalizados utilizando el método previamente descrito, y se almacenan en archivos cuyo nombre depende de la clasificación, `buena_k.csv` o `mala_k.csv`, donde `k` es el número de fotograma, para mejorar la mantenibilidad del sistema en caso de necesitar corregir errores.

Con los datos guardados, se entrena el modelo de *gradient boosting* utilizando la biblioteca Scikit-learn y se lo almacena en un archivo llamado `modelo_postura.pkl`, 
 
### Clasificación de la postura en tiempo real
Una vez entrenado el modelo, se procede a clasificar la postura del usuario en tiempo real. Para cada fotograma que se captura en la cámara, se realiza lo siguiente:
1. Se calculan los 33 *keypoints* del fotograma con Mediapipe.
2. Se filtra los *keypoints* para utilizar sólo los relevantes al análisis, que son los 8 que se encuentran por encima del torso.
3. Se extraen las coordenadas $(x,y,z)$ de los 8 *keypoints* y se normalizan.
4. Se clasifica la postura capturada con el modelo entrenado.
5. Se muestra la información relevante en pantalla.

### Diagrama de flujo para la interfaz de usuario
Se presenta un diagrama de flujo que detalla en forma general el funcionamiento dentro de la ejecución del bucle del programa.

![[Diagrama de Flujo]]