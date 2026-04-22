### Preprocesamiento de los datos antes de entrenar los modelos
Primeros pasos del proyecto:  
1. Data collection: obtenemos los datos raw. En este caso, los ficheros .root que contienen los pulso light y dark
2. Data cleaning: limpiamos los datos. Eliminamos columnas vacías, datos duplicados, datos irrelevantes, etc. Para este
proyecto, solo nos quedaremos con la columna dataChA, que contiene los pusos. Dentro de esta columna revisaremos que no 
haya pulsos duplicados.
3. Data Exploring: eliminamos valores atípicos que puedan sesgar el entrenamiento de los modelos. Examinaremos tmabién si 
hay relaciones entre caracterísitcas, en este caso no tenemos más caracteristicas.
4. Data preprocessing: si es necesario, normalizamos los datos o los escalamos, o convertirlos a un formato que el modelo
pueda entender. No es el caso, están todos los pulsos en el mismo formato.
5. Data splitting: división de los datos en **Training - validation - testing**
    - 1000 light + 1000 dark para entrenamiento
        - 80% entrenamiento y 20% validation
    - el resto para testing

### Finalmente se ha decidido por NO filtrar el dataset
**Conlusión:**
En reuniones con Alberto se ha tomado la decisión de solo tomar en cuenta una ventana significativa de muestras por cada pulso. Con el objetivo de
reducir carga computacional y suprimir los pulsos dobles.

## DUDAS:
Si tengo dos máximos locales en un solo pulso, al eliminar UNO  de ellos, no estoy eliminando el puslo, estaría eliminando muestras del pulso
Seguirían siendo 4722 pulsos pero esa muestra en vex de 10000 tendrá (por ejemplo) 9000 muestras.
**EN todo caso si quiero eliminar pulsos dobles sería comparar aquellos que estén practicamente superpuestos y quedarme con uno solo.**
**O lo que viene a ser lo mismo, ver si coinciden sus picos maximos y minimos, ya que si coinciden o son practiamente iguales, puedo quedarme con solo un pulso representativo**.

## ERRORES EN LA SEGMENTACION DE DATASET
El error too many indices for array ocurre porque, aunque creas tener una matriz de dos dimensiones, para NumPy tu objeto actual es una "lista" unidimensional de 4700 elementos donde cada elemento es, a su vez, otro array. Tu shape de (4700, 1) confirma que NumPy lo está tratando como una columna de objetos, no como una matriz de números de dos ejes.

### Opción 1: Convertirlo en una matriz 2D real
Si todos tus arrays internos miden exactamente 10,000, puedes "apilarlos" para crear una matriz de verdad. Esto te permitirá usar el rebanado (slicing) de dos índices que intentaste originalmente.

- Convertimos la estructura de objetos a una matriz 2D real
light_pulses_2d = np.stack(light_pulses.flatten())
- Ahora el slicing funcionará perfectamente
X_light = light_pulses_2d[0:1000, 1200:2000]

### Usar una lista de comprensión (Si no quieres convertir todo)
Solucionado! Se ha aplicado faltten para aplanar el array y sea mas manejable
Iteramos por los primeros 1000 arrays y cortamos cada uno:
X_light = np.array([fila[1200:2000] for fila in light_pulses[:1000].flatten()])

### Dia 22/04
Tenemos una primera versión del autoencoder. Se abren dos caminos:
1) Entrenar el autoencoder solo con pulsos DARK para que el modelo aprenda ha reconstruir, y por tanto identificar los pulsos DARK.
2) Entrenar el autoencoder con los os tipos de pulsos. El modelo transforma cada tipo de pulso en su code simplificado. Esta alternativa es interesante si queremos 
ver un mapa de distribución de los pulsos.

Idea: aplicaremos ambas opciones y nos quedaremos con la que mejor separe los pulsos LIGHT de los DARK.