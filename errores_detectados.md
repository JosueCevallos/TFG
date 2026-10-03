# Errores detectados — `light_and_dark_models_classifyer_v5.py`

Revisión completa del script (1835 líneas). Se listan de más a menos grave.

## 🔴 Bug crítico 1 — Offset de baseline no se aplica al conjunto de test

**Ubicación:** `seleccionar_ventana_muestras` (líneas ~403-421)

```python
for traza in LIGHT_PULSES[:n_samples]:
    offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
    trazas_light.append(traza[inicio:fin] - offset)      # train: SÍ resta offset
...
for traza in LIGHT_PULSES[n_samples:]:
    #offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
    trazas_light_test.append(traza[inicio:fin])           # test: NO resta offset (comentado)
```

Las trazas de **entrenamiento** se corrigen de baseline (se les resta la media de la
zona previa a la ventana), pero las de **test** no — el cálculo del offset está
comentado. Esto significa que todo lo que se evalúe contra `X_light_test` /
`X_dark_test` (CNN binaria, autoencoders, comparación final) ve datos con un
preprocesado distinto al usado en entrenamiento, lo que puede degradar o falsear
todas las métricas de evaluación.

**Estado:** corregido — se descomenta el cálculo del offset y se resta también en
el conjunto de test.

## 🔴 Bug crítico 2 — GMM: normalización de DARK con su propio scaler

**Ubicación:** `entrenar_gmm` (líneas ~1143-1144)

```python
F_light_train_norm, F_light_test_norm, scaler_gmm = normalizar_caracteristicas(F_light_train, F_light_test)
F_dark_train_norm, F_dark_test_norm, scaler_dark =  normalizar_caracteristicas(F_dark_train, F_dark_test)
```

El docstring dice explícitamente "Normalización: parámetros calculados solo con
light", pero el código ajusta un **scaler independiente sobre dark**
(`scaler_dark`) en vez de reutilizar `scaler_gmm.transform(F_dark_test)`. Como el
GMM se entrena únicamente con features light normalizadas con `scaler_gmm`,
evaluar dark con su propia normalización recentra artificialmente sus
características, reduciendo la separación real entre clases y falseando el AUC
calculado en `evaluar_gmm`.

**Estado:** corregido — dark se normaliza ahora con `scaler_gmm` (el scaler
ajustado sobre light).

## 🟠 Bug funcional — `PASOS_PREVIOS` desalineado

**Ubicación:** líneas ~180-193

| Opción | Exigía antes | Ahora exige |
|---|---|---|
| 2 (Cargar datasets) | *(sin entrada)* | `light_dataset_raw`, `dark_dataset_raw` |
| 9 (Entrenar GMM) | `resultados_autoencoders` | `X_light` |
| 10 (Evaluar GMM) | `X_light` | `gmm` |
| 11 (Isolation Forest) | `gmm` | `X_light` |
| 13 (Comparación) | `X_light` | `roc_auc_cnn`, `errores_autoencoders`, `resultados_gmm`, `resultados_if`, `roc_auc_reg` |

La lista correcta para "Comparación" estaba guardada bajo la clave `"14"`, opción
que no existe en el menú (código muerto). Consecuencias reales: se podía entrar en
"10. Evaluación GMM" sin haber entrenado el GMM y el programa petaba con
`TypeError`; igual con "13" si faltaba algún resultado. Tampoco existía entrada
para la opción "2. Cargar datasets": si se ejecutaba sin haber pasado antes por la
opción "1", `light_dataset_raw` / `dark_dataset_raw` eran `None` y
`preparacion_datos` petaba.

**Estado:** corregido — se realinean las claves 9-13, se añade la clave 2 y se
elimina la clave muerta 14.

## 🟡 Bug metodológico menor — normalización del test set en CNN binaria

**Ubicación:** `preparar_datos_cnn_binaria` (línea ~535)

```python
X = ((X - np.mean(X)) / np.std(X))[..., np.newaxis]                    # train: stats propios (correcto)
X_test = ((X_test - np.mean(X_test))/np.std(X_test))[..., np.newaxis]  # test: stats propios (¡debería usar los de train!)
```

Lo habitual es normalizar el test con la media/std calculadas sobre train, no
recalcularlas sobre el propio test. Aquí se recalculan de forma independiente, lo
cual es inconsistente con el resto del pipeline (autoencoders, Isolation Forest y
CNN regresión sí reutilizan las estadísticas del conjunto de referencia
correctamente).

**Estado:** corregido — `X_test` se normaliza ahora con la media/std calculadas
sobre `X` (train).

## ⚪ Detalle cosmético — nombre de variable engañoso

**Ubicación:** `main()`, línea ~119

El 5º valor devuelto por `evaluar_cnn` es en realidad `tpr_cnn`, pero se guardaba
en una variable llamada `fnr_cnn`. No rompía nada (se pasaba posicionalmente bien
a `comparar_modelos`), pero el nombre confundía al leer el código.

**Estado:** corregido — renombrada a `tpr_cnn`.
