"""light_and_dark_models_classifyer_v2.ipynb

=========================================================
    Boceto Menu funcional. DISCRIMINACIÓN DE RUIDO EN SEÑALES TES
=========================================================
[1] Introducción
[2] Preparación de datos
[3] Estudio de datasets
[4] Selección de ventana de muestras
---------------------------------------------------------
[5] CNN de clasificación binaria
[6] CNN de regresión
[7] Autoencoders
[8] Gaussian Mixture Model (GMM)
[9] Isolation Forest
---------------------------------------------------------
[10] Comparación de modelos
[11] Visualización de resultados
[12] Conclusiones
---------------------------------------------------------
[0] Salir
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.layers as layers
import keras.models as models
import keras.initializers as initializers
from keras import Input
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
from keras.layers import Conv1D
from keras.models import Model
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest

""" BLOQUE MENU PRINCIPAL """

def mostrar_menu():
    """Muestra el menú principal."""

    print("\n" + "=" * 30)
    print("      TFG - DISCRIMINACIÓN DE RUIDO EN TES")
    print("=" * 30)
    print("1. Mostrar jerarquía datasets")
    print("2. Cargar datasets")
    print("3. Estudio de datasets")
    print("4. Selección de ventana de muestras")
    print("5. Entrenar CNN de clasificación binaria")
    print("6. Evaluar CNN de clasificación binaria")
    print("7. Entrenar Autoencoders")
    print("8. Evaluación de los Autoencoders")
    print("9. Visualizar scatterplots del espacio latente")
    print("10. Gaussian Mixture Model (GMM)")
    print("11. Evaluación GMM")
    print("12. Isolation Forest")
    print("13. Modelo CNN DE regresión")
    print("14. Comparación de modelos")
    print("0. Salir")
    print("=" * 30)

def main():

    while True:

        mostrar_menu()
        opcion = input("Seleccione una opción: ")
        # --------------------------------------------------
        # 0. Salir
        # --------------------------------------------------
        if opcion == "0":
            print("\nFinalizando programa...")
            break

        # --------------------------------------------------
        # 1. Mostrar jerarquía datasets
        # --------------------------------------------------
        elif opcion == "1":
            light_dataset_raw = mostrar_estructura_dataset("TES1-0.3RN-5GHz-50MHz-laser-5s-trigger-10mV.root")
            dark_dataset_raw = mostrar_estructura_dataset("TES1-0.3RN-5GHz-50MHz-extrinsic-2d-trigger-10mV.root")
        # --------------------------------------------------
        # 2. Cargar datasets
        # --------------------------------------------------
        elif opcion == "2":
            LIGHT_PULSES, DARK_PULSES = preparacion_datos(light_dataset_raw, dark_dataset_raw)
        # --------------------------------------------------
        # 3. Estudio datasets
        # --------------------------------------------------
        elif opcion == "3":
            if LIGHT_PULSES is None:
                print("Primero debe cargar los datasets.")
                continue
            n = int(input("Número de trazas a visualizar: "))
            estudio_datasets(LIGHT_PULSES,DARK_PULSES,n,1200,2500)
        # --------------------------------------------------
        # 4. Selección ventana
        # --------------------------------------------------
        elif opcion == "4":
            if LIGHT_PULSES is None:
                print("Primero debe cargar los datasets.")
                continue
            n_samples = int(input("Número de muestras por clase para entrenamiento: "))
            inicio = int(input("Inicio ventana de muestras: "))
            fin = int(input("Fin ventana de muestras: "))
            X_light, X_dark, X_light_test, X_dark_test = seleccionar_ventana_muestras(LIGHT_PULSES, DARK_PULSES,n_samples, inicio, fin)
        # --------------------------------------------------
        # 5 y 6. CNN
        # --------------------------------------------------
        elif opcion == "5":

            print("\n[Módulo CNN Binaria]")
            model, history,X_test_cnn, y_test_cnn = entrenar_modelo_cnn(X_light, X_dark, X_light_test, X_dark_test)

        elif opcion == "6":

            print("\n[Evaluación CNN Binaria]")
            y_proba, y_pred, roc_auc_cnn , fpr_cnn, fnr_cnn = evaluar_cnn(model, history, X_test_cnn, y_test_cnn)
            print("\n Fin Evaluación CNN Binaria ")
        # --------------------------------------------------
        # 7 y 8. AUTOENCODERS
        # --------------------------------------------------
        elif opcion == "7":
            print("\n[Módulo Autoencoders]")
            resultados_autoencoders = entrenar_autoencoders(X_light, X_dark, X_light_test, X_dark_test)
            print("Entrenamiento completado.")
        
        elif opcion == "8":
            print("\n[Evaluación de los Autoencoders]")
            errores_autoencoders = evaluar_autoencoders(resultados_autoencoders)
            print("Entrenamiento completado.")
        
        elif opcion == "9":
            print("\n[Visaulizar scatterplosts del espacio latente de los Autoencoders]")
            visualizar_scatterplots_autoencoders(resultados_autoencoders)
            print("Fin de visualización.")
        # --------------------------------------------------
        # 10. GMM
        # --------------------------------------------------
        elif opcion == "10":
            print("\n[Módulo GMM]")
            gmm = entrenar_gmm(X_light, X_dark, 1) #light, dark, nº campanas
            print("Entrenamiento modelo GMM Completado.")
        
        elif opcion == "11":
            print("\n[Módulo GMM]")
            print("Evaluación modelo GMM.")
            resultados_gmm = evaluar_gmm(gmm)
        # --------------------------------------------------
        # 9. Isolation Forest
        # --------------------------------------------------
        elif opcion == "12":
            print("\n[Módulo Isolation Forest]")
            resultados_if = entrenar_y_evaluar_isolation_forest(X_light, X_dark)
        
        # --------------------------------------------------
        # 13. MODELO CNN DE REGRESION
        # --------------------------------------------------
        elif opcion == "13":
            print("\n[Módulo CNN de regresión]")
            entrenar__cnn_regresion(X_light, X_dark, X_light_test, X_dark_test)
        # --------------------------------------------------
        # 10. Comparación
        # --------------------------------------------------
        elif opcion == "14":
            print("\nComparación de modelos.")
            comparar_modelos(roc_auc_cnn, fpr_cnn, fnr_cnn, errores_autoencoders, resultados_gmm["roc_auc_gmm"],
                             resultados_gmm["fpr_gmm"], resultados_gmm["tpr_gmm"],
                             resultados_if["roc_auc_if"], resultados_if["fpr_if"], resultados_if["tpr_if"])

        # --------------------------------------------------

        else:
            print("Opción no válida.")


# ================================================
# --------- BLOQUE FUNCIONES AUXILIARES ---------
# ===============================================
def mostrar_estructura_dataset(root_file):
    """
    Muestra la estructura interna de los
    datasets.
    """
    dataset_raw = uproot.open(root_file)
    trees = dataset_raw.keys()
    branches = []
    for tree in trees:
        branches.append(dataset_raw[tree].keys())

    print("root_file")
    for i in range(len(trees)):
        print(f"└── TTree: {trees[i]}")
        print(f"    └── branches: {branches[i]}")
    
    return dataset_raw

def preparacion_datos(light_dataset_raw, dark_dataset_raw):
    """
    OBJ: Cargar los datasets LIGHT y DARK desde los ficheros ROOT.
    PARAMS:
        light_dataset_raw : (uproot.Dataset) Dataset RAW de pulsos LIGHT.
        dark_dataset_raw : (uproot.Dataset) Dataset RAW de pulsos DARK.
    RETURNS:
        LIGHT_PULSES : (np.ndarray) Pulsos procedentes del dataset LIGHT.
        DARK_PULSES : (np.ndarray) Pulsos procedentes del dataset DARK.
    """

    # Apertura de árboles ROOT
    tree_light_data = light_dataset_raw["alazar_data"]
    tree_light_parameters = light_dataset_raw["alazar_parameters"]
    tree_dark_data = dark_dataset_raw["alazar_data"]
    tree_dark_parameters = dark_dataset_raw["alazar_parameters"]

    # Extracción de pulsos
    LIGHT_PULSES = tree_light_data["dataChA"].array(library="np")
    DARK_PULSES = tree_dark_data["dataChA"].array(library="np")

    # Información de carga
    print("\n======================================")
    print("      PREPARACIÓN DE DATOS")
    print("======================================")

    print("\nDataset LIGHT")
    print(f"\nNúmero de pulsos LIGHT: {LIGHT_PULSES.shape[0]}")
    print(f"Muestras por pulso: {LIGHT_PULSES[0].shape[0]}")

    print("\nDataset DARK")
    print(f"\nNúmero de pulsos DARK: {DARK_PULSES.shape[0]}")
    print(f"Muestras por pulso: {DARK_PULSES[0].shape[0]}")

    print("\nShapes completos:")
    print(f"LIGHT_PULSES.shape = {LIGHT_PULSES.shape}")
    print(f"DARK_PULSES.shape  = {DARK_PULSES.shape}")

    return LIGHT_PULSES, DARK_PULSES

def estudio_datasets(light_dataset, dark_dataset,n_trazas,ventana_ini,ventana_fin):
    """
    OBJ: estudio exploratorio de los datasets LIGHT y DARK.
    PARAMS:
        LIGHT_PULSES : (np.ndarray) Pulsos de luz.
        DARK_PULSES : (np.ndarray) Pulsos de fondo.
        n_trazas : (int) Número de trazas a visualizar.
        ventana_ini : (int) Inicio de la ventana de interés.
        ventana_fin : (int) Fin de la ventana de interés.
    """

    print("\n=====================================")
    print("        ESTUDIO DE DATASETS")
    print("======================================")

    print(f"\nNúmero de pulsos LIGHT: {len(light_dataset)}")
    print(f"Número de pulsos DARK : {len(dark_dataset)}")
    print(f"Muestras por traza    : {light_dataset[0].shape[0]}")

    # ===================================
    # Trazas LIGHT en escala temporal
    # ===================================

    fig, axes = plt.subplots(n_trazas, 1, figsize=(10, 2 * n_trazas))
    if n_trazas == 1:
        axes = [axes]
    for i in range(n_trazas):
        t = np.arange(len(light_dataset[i])) / 50
        axes[i].plot(t, light_dataset[i] * 1000)

        axes[i].set_ylabel("V (mV)")
        axes[i].set_xlabel("t (µs)")
        axes[i].set_title(f"Traza LIGHT {i}")

    plt.tight_layout()
    plt.show()

    # ===================================
    # Primeras trazas LIGHT
    # ===================================

    plt.figure(figsize=(12, 6))
    for i in range(n_trazas):
        plt.plot(light_dataset[i], alpha=0.7)

    plt.title(f"Primeras {n_trazas} trazas LIGHT")
    plt.xlabel("Muestras")
    plt.ylabel("Vout (V)")
    plt.grid(color='gray', linestyle='--', linewidth=0.2)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.0025))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(600))
    plt.show()

    # ===================================
    # Primeras trazas DARK
    # ===================================

    plt.figure(figsize=(12, 6))

    for i in range(n_trazas):
        plt.plot(dark_dataset[i], alpha=0.7)

    plt.title(f"Primeras {n_trazas} trazas DARK")
    plt.xlabel("Muestras")
    plt.ylabel("Vout (V)")
    plt.grid(color='gray', linestyle='--', linewidth=0.2)

    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.0025))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(600))

    plt.show()

    # ===================================
    # Pulsos medios
    # ===================================

    mean_light = np.mean(light_dataset, axis=0)
    mean_dark = np.mean(dark_dataset, axis=0)

    plt.figure(figsize=(12, 6))

    plt.plot(mean_light, label="Light", color="orange")
    plt.plot(mean_dark, label="Background", color="black")

    plt.axvline(x=ventana_ini, color='green', linestyle='--',
        label='Ventana inicio')
    plt.axvline(x=ventana_fin, color='red', linestyle='--',
        label='Ventana fin')
    plt.title("Pulso medio con ventana de interés")
    plt.ylabel("Vout (V)")
    plt.xlabel("Muestras")

    plt.grid(color='gray', linestyle='--', linewidth=0.2)

    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.0025))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(600))

    plt.legend()
    plt.show()

def seleccionar_ventana_muestras(LIGHT_PULSES, DARK_PULSES, n_samples, inicio, fin):
    """
    OBJ: Recorta cada traza en la ventana fija [inicio:fin] y elimina
    el offset de voltaje usando el baseline previo a la ventana.
    PARAMS:
    n_samples  : (int)Número de muestras a recortar.
    inicio  :  (int)Índice de inicio de la ventana (incluido).
    fin     :  (int)Índice de fin de la ventana (excluido).
    RETURN:
    light_pulses  : (np.ndarray) Trazas recortadas con offset eliminado.
    dark_pulses   : (np.ndarray) Trazas recortadas con offset eliminado.
    """
    assert inicio >= 0, "El índice de inicio debe ser >= 0."
    assert fin <= len(LIGHT_PULSES[0]), (
        f"El índice fin ({fin}) supera la longitud de la traza ({len(LIGHT_PULSES[0])})."
    )
    assert fin <= len(DARK_PULSES[0]), (
        f"El índice fin ({fin}) supera la longitud de la traza ({len(DARK_PULSES[0])})."
    )
    assert inicio < fin, "El índice inicio debe ser menor que fin."

    trazas_light = []
    trazas_dark = []
    trazas_light_test = []
    trazas_dark_test = []
    for traza in LIGHT_PULSES[:n_samples]:
        offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
        trazas_light.append(traza[inicio:fin] - offset)
    for traza in DARK_PULSES[:n_samples]:
        offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
        trazas_dark.append(traza[inicio:fin] - offset)
    for traza in LIGHT_PULSES[n_samples:]:
        #offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
        trazas_light_test.append(traza[inicio:fin])
    for traza in DARK_PULSES[n_samples:]:
        #offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
        trazas_dark_test.append(traza[inicio:fin])

    input_len = fin - inicio
    print(f"Ventana seleccionada: [{inicio}:{fin}]")
    print(f"Longitud de traza resultante: {input_len} muestras "
          f"({input_len / 50:.1f} µs a 50 MHz)")

    return np.array(trazas_light), np.array(trazas_dark), np.array(trazas_light_test), np.array(trazas_dark_test)

def entrenar_modelo_cnn(X_light,X_dark,X_light_test,X_dark_test):
    """
    OBJ: llama a funciones auxiliares para entrenar el modelo CNN 
        de clasificación binaria.
    PARAMS:
        X_light : (np.ndarray) Pulsos LIGHT de entrenamiento.
        X_dark : (np.ndarray) Pulsos DARK de entrenamiento.
        X_light_test : (np.ndarray) Pulsos LIGHT de test.
        X_dark_test : (np.ndarray) Pulsos DARK de test.
    RETURNS:
        model : (tf.keras.Model) Modelo CNN entrenado.
        history : (tf.keras.callbacks.History) Historial de entrenamiento.
        X_test_cnn : (np.ndarray) Conjunto de test para evaluación.
        y_test_cnn : (np.ndarray) Etiquetas de test para evaluación.
    """
    print("\n[Entrenamiento CNN Binaria]")

    X_train_cnn,X_val_cnn,X_test_cnn,y_train_cnn,y_val_cnn,y_test_cnn = preparar_datos_cnn_binaria(X_light, X_dark, X_light_test, X_dark_test)
    model = build_cnn(input_shape= X_train_cnn.shape[1:],num_conv_layers= 6,num_filters= 45,kernel_size= 5,
                      dense_initial_units= 188, num_dense_layers= 3, dropout_rate= 0.18, learning_rate= 5.2e-4)
    print(f"Train: {X_train_cnn.shape} | Val: {X_val_cnn.shape}")
    #parametros escogidos segun el articulo, se pueden modificar para experimentar con la arquitectura de la red
    history =model.fit(X_train_cnn, y_train_cnn,validation_data=(X_val_cnn, y_val_cnn),epochs=20,batch_size=99)
    model.summary()

    print("Entrenamiento de la red CNN completado.")

    return model, history, X_test_cnn, y_test_cnn

def preparar_datos_cnn_binaria(X_light_cnn, X_dark_cnn, X_light_cnn_test,
                    X_dark_cnn_test, test_size_val=0.2, random_state=42):
    """
    OBJ:Preparar los conjuntos de entrenamiento,
        validación y test para la CNN binaria.
    PARAMS:
        X_light_cnn : (np.ndarray)Pulsos LIGHT de entrenamiento.
        X_dark_cnn : (np.ndarray)Pulsos DARK de entrenamiento.
        X_light_cnn_test : (np.ndarray)Pulsos LIGHT de test.
        X_dark_cnn_test : (np.ndarray)Pulsos DARK de test.
        test_size_val : (float)Porcentaje destinado a validación.
        random_state : (int)Semilla para reproducibilidad.
    RETURNS:
        X_train, X_val, X_test,
        y_train, y_val, y_test
    """

    # Variables dependientes
    y_light = np.ones(len(X_light_cnn))
    y_dark = np.zeros(len(X_dark_cnn))
    y_light_test = np.ones(len(X_light_cnn_test))
    y_dark_test = np.zeros(len(X_dark_cnn_test))

    # Combinar conjuntos
    X = np.concatenate([X_light_cnn, X_dark_cnn], axis=0)
    X_test = np.concatenate([X_light_cnn_test, X_dark_cnn_test],axis=0)
    y = np.concatenate([y_light, y_dark],axis=0)
    y_test = np.concatenate([y_light_test, y_dark_test],axis=0)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    indices_test = np.random.permutation(len(X_test))
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]

    # Normalización
    X = ((X - np.mean(X)) / np.std(X))[..., np.newaxis]
    X_test = ((X_test - np.mean(X_test))/np.std(X_test))[..., np.newaxis]

    # Dividir conjunto Train / Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,y,test_size=test_size_val,stratify=y,
        random_state=random_state)

    # Resumen
    print("\n===================================")
    print(" PREPARACIÓN CNN BINARIA")
    print("===================================")

    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")

    print(f"y_train: {y_train.shape}")
    print(f"y_val  : {y_val.shape}")
    print(f"y_test : {y_test.shape}")

    return (X_train,X_val,X_test,y_train,y_val,y_test)

def build_cnn(input_shape, num_conv_layers, num_filters, kernel_size, dense_initial_units, 
              
              num_dense_layers, dropout_rate, learning_rate):

    model = models.Sequential()

    initializer = initializers.GlorotUniform()

    # Capas convolucionales
    for i in range(num_conv_layers):
        if i == 0:
            model.add(layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                activation='tanh',
                kernel_initializer=initializer,
                input_shape=input_shape
            ))
        else:
            model.add(layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                activation='tanh',
                kernel_initializer=initializer
            ))

        model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Flatten())

    # Capas densas
    units = dense_initial_units
    for _ in range(num_dense_layers):
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_initializer=initializer
        ))
        model.add(layers.Dropout(dropout_rate))
        units = units // 2

    # Salida binaria final
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compilación
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def evaluar_cnn(model, history, X_test_cnn, y_test_cnn):

    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    print("\n")
    loss, accuracy = model.evaluate(X_test_cnn, y_test_cnn)

    print(f"Validation accuracy: {accuracy:.4f}")
    print("\n")

    # Predicciones
    y_proba = model.predict(X_test_cnn,verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    # Informe
    print(
        classification_report(y_test_cnn,y_pred,target_names=["Dark", "Light"]))

    # Matriz de confusión
    cm = confusion_matrix(y_test_cnn, y_pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Dark", "Light"],
        yticklabels=["Dark", "Light"]
    )

    plt.ylabel("Etiqueta real")
    plt.xlabel("Etiqueta predicha")
    plt.title("Matriz de confusión — CNN Binaria")

    plt.tight_layout()
    plt.show()
    print("\n")
    # ROC
    y_scores_anomalia = 1 - y_proba

    fpr_cnn, tpr_cnn, thresholds = roc_curve(
        1 - y_test_cnn,
        y_scores_anomalia
    )

    roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr_cnn,
        tpr_cnn,
        color="steelblue",
        linewidth=2,
        label=f"CNN Binaria (AUC={roc_auc_cnn:.3f})"
    )

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)

    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC — CNN Binaria")

    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

    return y_proba, y_pred, roc_auc_cnn, fpr_cnn, tpr_cnn
# ARQUITECTURA COMÚN DE LOS AUTOENCODERS
def build_autoencoder(input_len, latent_dim):
    """
    OBJ:
        Autoencoder convolucional 1D con adaptación automática de la
        longitud de entrada mediante ZeroPadding y Cropping.

    PARAMS:
        input_len  : número de muestras por traza.
        latent_dim : dimensión del espacio latente.

    RETURN:
        autoencoder, encoder
    """

    # -------------------------------------------------
    # Padding automático para que la longitud sea
    # múltiplo de 2^3 = 8 (3 MaxPooling1D)
    # -------------------------------------------------

    padding = (8 - input_len % 8) % 8
    padded_len = input_len + padding

    inputs = Input(shape=(input_len, 1), name="encoder_input")

    x = inputs

    if padding > 0:
        x = layers.ZeroPadding1D((0, padding), name="input_padding")(x)

    # ENCODER
    x = layers.Conv1D(32, kernel_size=7, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(16, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(8, kernel_size=3, activation="relu", padding="same")(x)

    x = layers.MaxPooling1D(2)(x)

    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]

    x = layers.Flatten()(x)

    bottleneck = layers.Dense(latent_dim,activation="relu",name="bottleneck")(x)

    # DECODER

    x = layers.Dense(shape_before_flatten[0] * shape_before_flatten[1],activation="relu")(bottleneck)
    x = layers.Reshape(shape_before_flatten)(x)
    x = layers.Conv1DTranspose(8,kernel_size=3,activation="relu",padding="same")(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(16,kernel_size=5,activation="relu",padding="same")(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(32,kernel_size=7,activation="relu",padding="same")(x)
    x = layers.UpSampling1D(2)(x)

    outputs = layers.Conv1D(
        1,
        kernel_size=1,
        activation="linear",
        padding="same",
        name="decoder_output"
    )(x)

    # -------------------------------------------------
    # Elimina automáticamente el padding añadido
    # -------------------------------------------------

    if padding > 0:
        outputs = layers.Cropping1D(cropping=(0, padding),name="output_cropping")(outputs)

    autoencoder = Model(inputs,outputs,name="autoencoder")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5.2e-4),loss="mae")

    encoder = Model(inputs,bottleneck,name="encoder")

    return autoencoder, encoder

# BLOQUE FUNCIONES DE ENTRENAMIENTO PARA AUTOENCODER
def entrenar_autoencoders(X_light, X_dark, X_light_test, X_dark_test):
    """
    OBJ: Entrena tres autoencoders: uno para pulsos LIGHT, otro para pulsos DARK y un tercero para ambos combinados.
    PARAMS: X_light, X_dark, X_light_test, X_dark_test
    RETURNS: Diccionario con los resultados de cada autoencoder."""

    """
    PARA EL AUTOENCODER LIGHT-DARK, SE COMBINA EL CONJUNTO DE DATOS LIGHT+DARK PARA SU ENTRENAMIENTO
    LA SUMA DE AMBOS CONJUNTOS DOBLA EL CONJUNTO DE ENTRENAMIENTO CON RESPECTO A LOS AUTOENCODERS LIGHT
    Y DARK POR SEPARADO, POR TANTO SE PERDERÍA LA OBJETIVIDAD DEL EXPERIMENTO.
    SOLUCION: DE CADA TIPO ME QUEDO CON LA MITAD, PERO VAMOS A PROBAR QUE SUCEDE SI ENTRENO CON AMBOS (DOBLE)
    """
    resultados = {
            "light": entrenar_autoencoder_light(X_light, X_dark),
            "dark": entrenar_autoencoder_dark(X_light, X_dark),
            "combined": entrenar_autoencoder_combined(X_light, X_dark, X_light_test, X_dark_test)
            }

    return resultados

def entrenar_autoencoder_light(X_light, X_dark):

    autoencoder_light, encoder_light = build_autoencoder(X_light.shape[1], latent_dim=16)
    autoencoder_light.summary()
    X_light_norm, [X_dark_for_ae_light_test], mean_l, std_l = normalizar(X_light, [X_dark])
    X_light_train, X_light_val = train_test_split(X_light_norm, test_size=0.2, random_state=42)
    print(f"Train light: {X_light_train.shape} | Val light: {X_light_val.shape}")
    history_light = autoencoder_light.fit(
        X_light_train, X_light_train,
        validation_data=(X_light_val, X_light_val),
        epochs=20,
        batch_size=99,
        shuffle=True
    )
    print("Fin entrenamiento Autoencoder Light")
    return {"autoencoder": autoencoder_light,
            "encoder": encoder_light,
            "history": history_light,
            "X_train": X_light_train,
            "X_val": X_light_val,
            "X_test": X_dark_for_ae_light_test,
            "mean": mean_l,
            "std": std_l}

def entrenar_autoencoder_dark(X_light, X_dark):
    autoencoder_dark, encoder_dark = build_autoencoder(X_light.shape[1], latent_dim=16)
    autoencoder_dark.summary()
    X_dark_norm, [X_light_for_ae_dark_test], mean_d, std_d = normalizar(X_dark, [X_light])
    X_dark_train, X_dark_val = train_test_split(X_dark_norm, test_size=0.2, random_state=42)
    print(f"Train dark: {X_dark_train.shape} | Val dark: {X_dark_val.shape}")
    history_dark = autoencoder_dark.fit(
        X_dark_train, X_dark_train,
        validation_data=(X_dark_val, X_dark_val),
        epochs=20,
        batch_size=99,
        shuffle=True
    )
    print("Fin entrenamiento Autoencoder Dark")
    return {"autoencoder": autoencoder_dark,
            "encoder": encoder_dark,
            "history": history_dark,
            "X_train": X_dark_train,
            "X_val": X_dark_val,
            "X_test": X_light_for_ae_dark_test,
            "mean": mean_d,
            "std": std_d}

def entrenar_autoencoder_combined(X_light, X_dark, X_light_test, X_dark_test):
    
    autoencoder_combined, encoder_combined = build_autoencoder(X_light.shape[1], latent_dim=16)
    autoencoder_combined.summary()
    """mitad = len(X_light) // 2 #dark tiene el mismo tamaño
        X_light = X_light[:mitad]
        X_dark  =  X_dark[:mitad]"""
    X_combined = np.concatenate([X_light, X_dark], axis=0)
    X_combined_norm, [X_light_for_combined_test, X_dark_for_combined_test], mean_b, std_b = normalizar(X_combined, [X_light_test, X_dark_test])
    X_combined_train, X_combined_val = train_test_split(X_combined_norm, test_size=0.2, random_state=42)
    print(f"Train combined: {X_combined_train.shape} | Val combined: {X_combined_val.shape}")
    history_combined = autoencoder_combined.fit(
        X_combined_train, X_combined_train,
        validation_data=(X_combined_val, X_combined_val),
        epochs=20,
        batch_size=99,
        shuffle=True
    )
    print("Fin entrenamiento Autoencoder Light + Dark")
    return {"autoencoder": autoencoder_combined,
            "encoder": encoder_combined,
            "history": history_combined,
            "X_train": X_combined_train,
            "X_val": X_combined_val,
            "X_test": [X_light_for_combined_test, X_dark_for_combined_test],
            "mean": mean_b,
            "std": std_b}

def evaluar_autoencoders(resultados):

    funciones_perdida_autoencoders(resultados["light"]["history"], resultados["dark"]["history"], resultados["combined"]["history"])
    errores_autoencoders = calcula_error_reconstruccion_autoencoders(resultados)
    visualizar_scatterplots_autoencoders(resultados)

    return errores_autoencoders

def funciones_perdida_autoencoders(history_light, history_dark, history_combined):
    """
    OBJ: Representa las funciones de pérdida de cada autoencoder.
    PARAMS:
        history_light : (History) Historial de entrenamiento del autoencoder light.
        history_dark : (History) Historial de entrenamiento del autoencoder dark.
        history_combined : (History) Historial de entrenamiento del autoencoder combinado.
    RETURNS:
        None
    """
    
    # FUNCIONES DE PERDIDA DE CADA AUTOENCODER
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    configs = [
        (history_light, 'Autoencoder — Solo Light',     'orange'),
        (history_dark,  'Autoencoder — Solo Dark',      'steelblue'),
        (history_combined,  'Autoencoder — Light + Dark',   'seagreen'),]

    for ax, (hist, title, color) in zip(axes, configs):
        ax.plot(hist.history['loss'],     color=color, label='Train')
        ax.plot(hist.history['val_loss'], color=color, linestyle='--', label='Val')
        ax.set_title(title)
        ax.set_xlabel('Época')
        ax.set_ylabel('MAE')
        ax.legend()
        ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    #plt.savefig("curvas_perdida_autoencoders.png", dpi=120)
    plt.show()

    """ Conclusiones de las gráficas:

    En la primera gráfica se ve que la función de pérdida para los datos de validación es bastante pequeño Este primer modelo se entrenó SOLO con pulsos Light, lo que nos está diciendo es que está reconstruyendo bastante bien los pulsos de luz. No parece que haya overfitting.

    En la segunda gráfica, tenemos un comportamiento bastante similar a la primera. El segundo modelo de autoencoder ha sido entrenado solo conk pulsos DARK, es decir, este modelo está reconstruyedo con bastante precisión los pulsos DARK. Destaca que la función de pérdida de validación cae más rápido que la 'train loss'. Esto puede ser porque el conjunto de validación tiene una distribución más homogenea que el del conjunto de entrenamiento simplemente por azar en la división. Hay que tener en cuenta que el conjunto de pulsos dark tiene más dispersión en los valores que los light.

    Por último, en la tercera gráfica, vemos que las funciones de pérdida para entrenamiento y validación NO convergen; además el MAE es mayor que los anteriores modelos.
    
    curvas_perdida_autoencoder_v1.png

    Repetiremos el experimento subiendo el nº de épocas a 35 y observamos que las funciones se estabilizan.

    **Sin embargo, por cada iteración las funciones de perdida pueden cambiar. Solución realizar al menos 3 iteraciones y quedarme con la media"""
# ============================================================
# EVALUACIÓN FINAL: ERROR DE RECONSTRUCCIÓN POR CLASE
# ============================================================
def calcula_error_reconstruccion_autoencoders(resultados):
    # --- Autoencoder light ---
    err_light_on_light, X_light_pred_onLight = error_reconstruccion(resultados["light"]["autoencoder"], resultados["light"]["X_val"])
    err_light_on_dark, X_dark_pred_onLight  = error_reconstruccion(resultados["light"]["autoencoder"], resultados["light"]["X_test"]) #pulsos dark normalizados obtenidos de la normalizacion de los pulsos light

    # --- Autoencoderante dark ---
    err_dark_on_dark, X_dark_pred_onDark   = error_reconstruccion(resultados["dark"]["autoencoder"], resultados["dark"]["X_val"])
    err_dark_on_light, X_light_pred_onDark  = error_reconstruccion(resultados["dark"]["autoencoder"], resultados["dark"]["X_test"]) #pulsos light normalizados obtenidoa de la normalizacion de los pulsos dark

    # --- Autoencoder ambos pulsos (normalizamos pulsos light y dark)
    #X_light_for_both = ((X_light - mean_b) / std_b)[..., np.newaxis]
    err_both_on_light, X_light_pred_onBoth = error_reconstruccion(resultados["combined"]["autoencoder"], resultados["combined"]["X_test"][0])
    err_both_on_dark,  X_dark_pred_onBoth  = error_reconstruccion(resultados["combined"]["autoencoder"], resultados["combined"]["X_test"][1]) #pulsos dark normalizados obtenidos de la normalizacion de los pulsos light

    # Distribuciones de error
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    configs_eval = [
        (err_light_on_light, err_light_on_dark, 'Entrenado con Light', 'orange', 'black'),
        (err_dark_on_light,  err_dark_on_dark, 'Entrenado con Dark',  'orange', 'black'),
        (err_both_on_light,  err_both_on_dark, 'Entrenado con Light+Dark', 'orange', 'black')]

    for ax, (err_l, err_d, title, c_l, c_d) in zip(axes, configs_eval):
        ax.hist(err_l, bins=40, alpha=0.6, color=c_l, label='Light', density=True)
        ax.hist(err_d, bins=40, alpha=0.6, color=c_d, label='Dark',  density=True)
        ax.set_title(title)
        ax.set_xlabel('Error de reconstrucción (MAE)')
        ax.set_ylabel('Densidad')
        ax.legend()
        ax.grid(True, linewidth=0.4)

    plt.suptitle('Distribución del error de reconstrucción por variante', y=1.02)
    plt.tight_layout()
    plt.show()

    errores_autoencoders = {
        "light":    {"err_on_light": err_light_on_light, "err_on_dark": err_light_on_dark},
        "dark":     {"err_on_light": err_dark_on_light,  "err_on_dark": err_dark_on_dark},
        "combined": {"err_on_light": err_both_on_light,   "err_on_dark": err_both_on_dark}
    }

    return errores_autoencoders

def error_reconstruccion(modelo, X):
    """MAE por muestra."""
    X_pred = modelo.predict(X, verbose=0)
    return np.mean(np.abs(X - X_pred), axis=(1, 2)), X_pred

def normalizar(X_train, X_test_list):
    """
    OBJ: Normaliza el conjunto X_train con su propia media y
        desviación. Aplica los mismos parámetros a cada array en X_test_list.
        Añade canal de dimensión.
    PARAMS:
        X_train : (np.ndarray) Conjunto de entrenamiento.
        X_test_list : (lista de np.ndarrady) Conjunto de testing.
    RETURNS:
        X_train_norm : (np.ndarray) Conjunto de entrenamiento normalizado.
        X_test_ae  : (lista de np.ndarrady) Conjunto de testing normalizado.
        mean : (float) Media del conjunto de entrenamiento.
        std : (float) Desviación estándar del conjunto de entrenamiento.
    """
    mean = np.mean(X_train)
    std  = np.std(X_train)
    X_train_norm = ((X_train - mean) / std)[..., np.newaxis]
    X_test_ae  = [((X - mean) / std)[..., np.newaxis] for X in X_test_list]
    return X_train_norm, X_test_ae, mean, std
# =============================================
# SCATTER PLOTS DEL ESPACIO LATENTE (PCA 2D)
# =============================================
def visualizar_scatterplots_autoencoders(resultados):
    """
    OBJ: Dibuja el espacio latente de los tres autoencoders.
    """

    scatter_latente(
        resultados["light"]["encoder"],
        resultados["light"]["X_train"],
        resultados["light"]["X_test"],
        "Espacio latente — Entrenado con Light"
    )

    scatter_latente(
        resultados["dark"]["encoder"],
        resultados["dark"]["X_train"],
        resultados["dark"]["X_test"],
        "Espacio latente — Entrenado con Dark"
    )

    scatter_latente(
        resultados["combined"]["encoder"],
        resultados["combined"]["X_test"][0],
        resultados["combined"]["X_test"][1],
        "Espacio latente — Entrenado con Light + Dark"
    )

def scatter_latente(encoder, X_light, X_dark, titulo):
    z_light = encoder.predict(X_light, verbose=0)
    z_dark  = encoder.predict(X_dark,  verbose=0)

    z_all = np.vstack([z_light, z_dark])
    pca   = PCA(n_components=2)
    z_2d  = pca.fit_transform(z_all)

    varianza = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(z_2d[:len(z_light), 0], z_2d[:len(z_light), 1],
               c='orange', alpha=0.5, s=12, label='Light', zorder=3)
    ax.scatter(z_2d[len(z_light):, 0], z_2d[len(z_light):, 1],
               c='black',  alpha=0.5, s=12, label='Dark', zorder=3)

    ax.set_xlabel(f'PC1 ({varianza[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({varianza[1]*100:.1f}%)')
    ax.set_title(titulo)
    ax.legend()

    # Cuadrícula más densa
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.grid(which='major', color='gray',  linestyle='--', linewidth=0.5, zorder=0)
    ax.grid(which='minor', color='gray',  linestyle=':',  linewidth=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(f"scatter_{titulo.replace(' ', '_')}.png", dpi=120)
    plt.show()

# ==================================================
# --- BLOQUE FUNCIONES DE ENTRENAMIENTO PARA GMM ---
# ==================================================
 
def normalizar_caracteristicas(F_train, F_test=None):
    """
    OBJ:Normalizar las características utilizando únicamente
        las estadísticas del conjunto de entrenamiento.
    PARAMS:
        F_train : (np.ndarray) Características de entrenamiento.
        F_test : (np.ndarray) opcional Características de test.
    RETURNS:
        Si F_test es None: F_train_norm, scaler
        Si F_test no es None: F_train_norm, F_test_norm, scaler
    """

    scaler = StandardScaler()
    F_train_norm = scaler.fit_transform(F_train)
    if F_test is None:
        return F_train_norm, scaler
    F_test_norm = scaler.transform(F_test)

    return F_train_norm, F_test_norm, scaler

# ==================================================
#------------EXTRACCION DE CARACTERISTICAS ---------
# ==================================================
def extraer_caracteristicas(pulsos):
    """
    OBJ: Extraer un vector reducido de características para el GMM.
    PARAMS:
        pulsos :(np.ndarray) Conjunto de trazas.
    RETURNS:
        features:(np.ndarray) de forma (N, 3)
    """

    features = []

    """Características:
        - v_min      : amplitud mínima del pulso.
        - tau_rise   : tiempo entre el 10% y el 90% de la amplitud.
        - tau_decay  : tiempo de recuperación desde el mínimo hasta el 10%."""

    for traza in pulsos:

        # Amplitud mínima
        v_min = np.min(traza)
        idx_min = np.argmin(traza)

        # Tiempo de bajada (10% -> 90%)
        flanco_bajada = traza[:idx_min]
        umbral_10 = 0.1 * v_min
        umbral_90 = 0.9 * v_min
        idxs_10 = np.where(flanco_bajada < umbral_10)[0]
        idxs_90 = np.where(flanco_bajada < umbral_90)[0]
        idx_10 = idxs_10[0] if len(idxs_10) > 0 else idx_min
        idx_90 = idxs_90[0] if len(idxs_90) > 0 else idx_min
        tau_rise = idx_90 - idx_10

        # Tiempo de recuperación
        flanco_subida = traza[idx_min:]
        idxs_rec = np.where(flanco_subida > umbral_10)[0]
        idx_rec = idxs_rec[0] if len(idxs_rec) > 0 else len(flanco_subida) - 1
        tau_decay = idx_rec

        features.append([v_min, tau_rise, tau_decay])

    return np.array(features)

def entrenar_gmm(X_light, X_dark, n_components):
    """
    OBJ: Entrena un GMM sobre las características extraídas de los pulsos LIGHT.
         El modelo aprende la distribución 'normal' (luz); los pulsos DARK,
         al no encajar en esa distribución, deberían obtener log-verosimilitudes bajas.
    PARAMS:
        X_light : (np.ndarray) Pulsos LIGHT de entrenamiento (trazas crudas, sin canal).
        X_dark  : (np.ndarray) Pulsos DARK, usados solo para normalizar sus características
                  con los mismos parámetros del conjunto light (no se usan para entrenar el GMM).
        n_components : (int) Número de componentes gaussianas.
    RETURNS:
        resultados_gmm : (dict) con el modelo, scaler, features normalizadas
                          y nombres de las características, para uso en evaluación.
    """
    print("\n[Entrenamiento GMM]")

    # Extracción de características (v_min, tau_rise, tau_decay, asimetria)
    F_light = extraer_caracteristicas(X_light)  #Features light pulses
    F_dark  = extraer_caracteristicas(X_dark)   #Features dark pulses

    nombres_features = ['v_min', 'tau_rise', 'tau_decay', 'asimetria']

    # Comprobación de NaN/Inf antes de entrenar
    print(f"NaN en F_light: {np.isnan(F_light).sum()} | NaN en F_dark: {np.isnan(F_dark).sum()}")
    print(f"Inf en F_light: {np.isinf(F_light).sum()} | Inf en F_dark: {np.isinf(F_dark).sum()}")

    # Normalización: parámetros calculados solo con light
    F_light_norm, F_dark_norm, scaler_gmm = normalizar_caracteristicas(F_light, F_dark)

    # Entrenamiento del GMM (solo con pulsos light)
    gmm_light = GaussianMixture(
        n_components    = n_components,
        covariance_type = 'full',
        random_state    = 42,
        max_iter        = 200,
        n_init          = 5
    )

    """gmm_2 = GaussianMixture(
        n_components    = n_components,
        covariance_type = 'full',
        random_state    = 42,
        max_iter        = 200,
        n_init          = 5
    )"""

    gmm_light.fit(F_light_norm)

    print(f"Convergencia: {gmm_light.converged_}")
    print(f"Iteraciones:  {gmm_light.n_iter_}")
    print("Pesos:", gmm_light.weights_)

    print("Entrenamiento GMM completado.")

    gmm = {
        "modelo": gmm_light,
        "scaler": scaler_gmm,
        "F_light_norm": F_light_norm,
        "F_dark_norm": F_dark_norm,
        "nombres_features": nombres_features}
    
    return gmm

def evaluar_gmm(resultados_gmm):
    """
    OBJ: Evalúa el GMM mediante la log-verosimilitud como score de anomalía
         y calcula la curva ROC / AUC.
         Score bajo (muy negativo) = pulso improbable bajo el modelo = anomalía (dark).
    PARAMS:
        resultados_gmm : (dict) salida de entrenar_gmm().
    RETURNS:
        log_prob_light, log_prob_dark : (np.ndarray) log-verosimilitudes por clase.
        roc_auc_gmm : (float) AUC de la curva ROC.
        fpr_gmm, tpr_gmm : (np.ndarray) puntos de la curva ROC.
    """
    print("\n[Evaluación GMM]")

    gmm_light    = resultados_gmm["modelo"]
    F_light_norm = resultados_gmm["F_light_norm"]
    F_dark_norm  = resultados_gmm["F_dark_norm"]

    # Log-verosimilitud como score de anomalía
    log_prob_light = gmm_light.score_samples(F_light_norm)  #Compute the log-likelihood of each sample.
    log_prob_dark  = gmm_light.score_samples(F_dark_norm)

    # Distribución de scores
    plt.figure(figsize=(8, 4))
    plt.hist(log_prob_light, bins=40, alpha=0.6, color='orange', density=True, label='Light')
    plt.hist(log_prob_dark,  bins=40, alpha=0.6, color='black',  density=True, label='Dark')
    plt.xlabel('Log-verosimilitud')
    plt.ylabel('Densidad')
    plt.title('Distribución del score de anomalía GMM')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

    # ROC: Dark = anomalía = 1, Light = normal = 0
    # Score de anomalía = -log_prob (cuanto menos probable, más score de anomalía)
    scores_all = np.concatenate([-log_prob_light, -log_prob_dark])
    labels_all = np.concatenate([
        np.zeros(len(log_prob_light)),  # light = 0
        np.ones(len(log_prob_dark))     # dark  = 1
    ])

    fpr_gmm, tpr_gmm, _ = roc_curve(labels_all, scores_all)
    roc_auc_gmm = auc(fpr_gmm, tpr_gmm)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_gmm, tpr_gmm, color='purple', linewidth=2, label=f'GMM (AUC={roc_auc_gmm:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC — GMM')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

    print(f"GMM AUC: {roc_auc_gmm:.4f}")

    resultados_evaluacion_gmm = {
        "log_prob_light": log_prob_light,
        "log_prob_dark": log_prob_dark,
        "roc_auc_gmm": roc_auc_gmm,
        "fpr_gmm": fpr_gmm,
        "tpr_gmm": tpr_gmm
    }

    return resultados_evaluacion_gmm

def entrenar_y_evaluar_isolation_forest(X_light, X_dark, n_estimators=100, contamination=0.05):
    """
    OBJ: Entrena un Isolation Forest sobre las trazas normalizadas (sin
         ingeniería de características) y evalúa su capacidad de discriminación.
         Entrena SOLO con pulsos LIGHT, igual criterio que el autoencoder "solo light".
    PARAMS:
        X_light, X_dark : (np.ndarray) Pulsos crudos (N, 1200), sin canal.
        n_estimators : (int) Número de árboles de aislamiento.
        contamination : (float) Fracción esperada de anomalías en el conjunto de entrenamiento.
    RETURNS:
        modelo_if : (IsolationForest) Modelo entrenado.
        roc_auc_if : (float) AUC de la curva ROC.
        fpr_if, tpr_if : (np.ndarray) Puntos de la curva ROC.
        scores_light, scores_dark : (np.ndarray) Scores de anomalía (negativo del path length).
    """
    print("\n[Entrenamiento Isolation Forest]")

    # Normalización con parámetros del conjunto light (mismo criterio que autoencoder)
    mean_if = np.mean(X_light)
    std_if  = np.std(X_light)

    X_light_if_norm = (X_light - mean_if) / std_if
    X_dark_if_norm  = (X_dark  - mean_if) / std_if

    print(f"Shape light para IF: {X_light_if_norm.shape}")
    print(f"Shape dark para IF:  {X_dark_if_norm.shape}")

    modelo_if = IsolationForest(
        n_estimators  = n_estimators,
        max_samples   = 'auto',
        contamination = contamination,
        random_state  = 42,
        n_jobs        = -1
    )
    modelo_if.fit(X_light_if_norm)
    print("Entrenamiento completado.")

    print("\n[Evaluación Isolation Forest]")

    # score_samples: valores más negativos = más anómalos
    scores_light = modelo_if.score_samples(X_light_if_norm)
    scores_dark  = modelo_if.score_samples(X_dark_if_norm)

    print(f"Score medio light: {np.mean(scores_light):.4f} ± {np.std(scores_light):.4f}")
    print(f"Score medio dark:  {np.mean(scores_dark):.4f} ± {np.std(scores_dark):.4f}")

    # Distribución de scores
    plt.figure(figsize=(8, 4))
    plt.hist(-scores_light, bins=40, alpha=0.6, color='orange', density=True, label='Light')
    plt.hist(-scores_dark,  bins=40, alpha=0.6, color='black',  density=True, label='Dark')
    plt.xlabel('Score de anomalía (negativo del path length)')
    plt.ylabel('Densidad')
    plt.title('Distribución del score de anomalía — Isolation Forest')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

    # ROC: Dark = anomalía = 1, Light = normal = 0
    scores_all = np.concatenate([-scores_light, -scores_dark])
    labels_all = np.concatenate([
        np.zeros(len(scores_light)),
        np.ones(len(scores_dark))
    ])

    fpr_if, tpr_if, _ = roc_curve(labels_all, scores_all)
    roc_auc_if = auc(fpr_if, tpr_if)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_if, tpr_if, color='darkgreen', linewidth=2, label=f'Isolation Forest (AUC={roc_auc_if:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC — Isolation Forest')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

    print(f"Isolation Forest AUC: {roc_auc_if:.4f}")

    # ── Scatter plot PCA del espacio de trazas ────────────────────────────────
    # A diferencia del GMM, aquí el PCA opera sobre las trazas directamente,
    # no sobre un vector de características extraídas manualmente.

    pca = PCA(n_components=2)
    X_all_2d = pca.fit_transform(np.vstack([X_light_if_norm, X_dark_if_norm]))
    varianza = pca.explained_variance_ratio_

    # Score de anomalía como color continuo para los dark
    scores_dark_color = -scores_dark  # más alto = más anómalo

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel izquierdo: coloreado por clase
    axes[0].scatter(X_all_2d[:len(X_light_if_norm), 0],
                    X_all_2d[:len(X_light_if_norm), 1],
                    c='orange', alpha=0.5, s=12, label='Light', zorder=3)
    axes[0].scatter(X_all_2d[len(X_light_if_norm):, 0],
                    X_all_2d[len(X_light_if_norm):, 1],
                    c='black', alpha=0.5, s=12, label='Dark', zorder=3)
    axes[0].set_xlabel(f'PC1 ({varianza[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({varianza[1]*100:.1f}%)')
    axes[0].set_title('PCA — Coloreado por clase')
    axes[0].legend()
    axes[0].grid(True, linewidth=0.4)

    # Panel derecho: coloreado por score de anomalía (solo dark)
    sc = axes[1].scatter(
        X_all_2d[len(X_light_if_norm):, 0],
        X_all_2d[len(X_light_if_norm):, 1],
        c=scores_dark_color, cmap='RdYlGn_r',
        alpha=0.7, s=15, zorder=3
    )
    axes[1].scatter(X_all_2d[:len(X_light_if_norm), 0],
                    X_all_2d[:len(X_light_if_norm), 1],
                    c='orange', alpha=0.3, s=8, label='Light', zorder=2)
    plt.colorbar(sc, ax=axes[1], label='Score de anomalía')
    axes[1].set_xlabel(f'PC1 ({varianza[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({varianza[1]*100:.1f}%)')
    axes[1].set_title('PCA — Score de anomalía (dark)')
    axes[1].legend()
    axes[1].grid(True, linewidth=0.4)

    plt.suptitle('Isolation Forest — Espacio de trazas (PCA 2D)', y=1.01)
    plt.tight_layout()
    plt.show()

    resultados_if = {
        "modelo": modelo_if,
        "roc_auc_if": roc_auc_if,
        "fpr_if": fpr_if,
        "tpr_if": tpr_if,
        "scores_light": scores_light,
        "scores_dark": scores_dark
    }

    return resultados_if

def comparar_modelos(roc_auc_cnn, fpr_cnn, tpr_cnn,
                      errores_autoencoders,
                      roc_auc_gmm, fpr_gmm, tpr_gmm,
                      roc_auc_if, fpr_if, tpr_if):
    """
    OBJ: Representa en una sola figura las curvas ROC de todos los modelos
         entrenados, para comparar su capacidad de discriminación Light/Dark.
    PARAMS:
        roc_auc_cnn, fpr_cnn, tpr_cnn : salida de evaluar_cnn().
        errores_autoencoders : (dict) salida de calcula_error_reconstruccion_autoencoders().
        roc_auc_gmm, fpr_gmm, tpr_gmm : salida de evaluar_gmm().
        roc_auc_if, fpr_if, tpr_if : salida de entrenar_y_evaluar_isolation_forest().
    """
    print("\n[Comparación de modelos]")

    plt.figure(figsize=(8, 7))

    # CNN binaria
    plt.plot(fpr_cnn, tpr_cnn, color='steelblue', linewidth=2,
              label=f'CNN Binaria (AUC={roc_auc_cnn:.3f})')

    # Isolation Forest
    plt.plot(fpr_if, tpr_if, color='darkgreen', linewidth=2,
              label=f'Isolation Forest (AUC={roc_auc_if:.3f})')

    # GMM
    plt.plot(fpr_gmm, tpr_gmm, color='purple', linewidth=2,
              label=f'GMM (AUC={roc_auc_gmm:.3f})')

    # Autoencoders (3 variantes), reconstruidas a partir de los errores guardados
    colores_ae = {'light': 'orange', 'dark': 'steelblue', 'combined': 'seagreen'}
    estilos_ae = {'light': '-', 'dark': '--', 'combined': '--'}
    etiquetas_ae = {'light': 'Autoencoder Solo Light',
                     'dark': 'Autoencoder Solo Dark',
                     'combined': 'Autoencoder Light+Dark'}

    for variante, errores in errores_autoencoders.items():
        err_on_light = errores["err_on_light"]
        err_on_dark  = errores["err_on_dark"]

        labels = np.concatenate([np.zeros(len(err_on_light)), np.ones(len(err_on_dark))])
        scores = np.concatenate([err_on_light, err_on_dark])

        fpr_ae, tpr_ae, _ = roc_curve(labels, scores)
        roc_auc_ae = auc(fpr_ae, tpr_ae)

        plt.plot(fpr_ae, tpr_ae, color=colores_ae[variante], linewidth=2,
                  linestyle=estilos_ae[variante],
                  label=f'{etiquetas_ae[variante]} (AUC={roc_auc_ae:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Clasificador aleatorio')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curvas ROC — Comparación de todos los modelos')
    plt.legend(loc='lower right')
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.savefig("roc_comparacion_completa.png", dpi=120)
    plt.show()

    print("Comparación completada.")

# ==================================================
#------------ BLOQUE CNN DE REGRESION --------------
# ==================================================
def entrenar_cnn_regresion():
    pass

# ===========================================================
# ----------------------- BLOQUE MAIN -----------------------
# ===========================================================
main()

"""Estudio y Análisis de los datos

Se representan los 10 primeros pulsos Light y Dark para tener una tenemos una idea general de la forma que tienen.
Observamos que los pulsos de ambos conjuntos se concentran en un rango de 1200 a 2500 muestras. Para comprobarlo se 
van a calcular la media para cada conjunto.
"""


"""División del conjunto de datos

Incialmente trabajaremos con los 4722 pulsos. Sin embargo, para reducir la carga computacional durante el entrenamiento
del modelo, se ha decidido seleccionar una ventana de muestras significativas. Para ello se ha calculado el pulso medio
del conjunto de datos light y dark. Se observa que la media en ambos casos se centra en el margen de muestra de 1000 a 
2000 muestras. Ajustando un poco más, se ha tomado como inicio 1300 como inicio. Se observa que los pulsos dark tienen
mayor dispersión, para que el modelo tenga más información de los pulsos dark, se ha decidido, tomar como limite final, 2500.
Además el usuario puede seleccionar con cuantas muestras quiere entrenar el modelo.

Tal y como esta el código, se seleccionan los 1000 primeros pulsos de cada conjunto. Si queremos que esos mil pulsos de
cada tipo sean aleatorio:

"""

"""Etiquetado de los pulsos light y dark y Normalización para modelo CNN

Ya tenemos dividido el conjunto de variables independientes (o características):
- X_light : para los light pulses.
- X_dark : para los dark pulses.
Ahora debemos crear el conjunto de variables dependientes (o etiquetas) para cada conjunto.
- y_light : 1, para light pulses. Dimension: (1000)
- y_dark: 0, para dark pulses: (1000)

"estas 1000 muestras es un pulso light"  
"estas 1000 muestras es un pulso dark"  
Así para los 1000 pulsos de cada conjunto de datos.
"""



"""### Construcción del modelo de CNN para clasificación binaria

Los parámetros y funciones de perdida son las que se mencionan en el artículo.
- input_shape=(1200, 1): tamaño de los datos que recibirá la red. En este caso, espera una señal de 1200 puntos de datos y 1 solo canal (como un pulso de luz de una sola dimensión). Pero esto puede cambiar dependiendo de la selecciona de ventanas del usuario.
- num_conv_layers=6: Define cuántas "capas de convolución" tendrá la red. Estas capas son las que detectan patrones (picos, valles, pendientes) en las señales
- num_filters=45: número de "neuronas" o detectores en cada capa convolucional. Más filtros permiten detectar patrones más complejos, pero consumen más memoria
- kernel_size=5: ancho de la "ventana" que desliza la red sobre tu señal. Un 5 significa que la red mira los puntos de 5 en 5 para buscar características
- dense_initial_units=128: Tras las capas de convolución, vienen las capas "densas" (clásicas). Este número indica cuántas neuronas tendrá la primera de esas capas.
- num_dense_layers=188: Indica cuántas capas de neuronas tradicionales habrá al final de la red antes de dar el resultado
- dropout_rate=0.18: técnica de "apagado aleatorio" del 30% de las neuronas durante el entrenamiento. Sirve para evitar el overfitting (que la red memorice los datos en lugar de aprender).
"""


"""## ----------- BLOQUE AUTOENCODERS ------------"""

# AUTOENCODER 1

# Autoencoder 2 ENTRENADO SOLO CON DARK

"""### Autoencoder 3 entrenado con pulsos Light y Dark

Este autoencoder se entrena con los dos tipos de pulsos, al sumar ambos 1000 light + 1000 dark obtenemos que el
autoencoder 3 se entrena con 2000 pulsos, el doble que los individuales. Se perdería la objetividad del experimento
ya que uno de los modelos se entrena con más datos.
solucion propuesta: de los 2000 pulsos de cada tipo elegiremos los 1000 primeros de cada tipo
"""

# AUTOENCODER 3: LIGHT + DARK

"""**Conclusiones del entrenamiento de las 3 variantes de Autoencoder.**
El primer Autoencdoer **solo** se ha entrenado con pulsos Light
"""

# =============================================
# SCATTER PLOTS DEL ESPACIO LATENTE (PCA 2D)
# =============================================

# =========================================================
# CURVAS ROC COMPARATIVAS DE LAS 3 VARIANTES DE AUTOENCODER
# =========================================================

"""## Modelo Mixto Gaussiano (GMM)

El GMM modela una distribución de probabilidad mediante combinación de gaussianas. La calidad de ese ajuste depende de la dimensionalidad de  
los pulsos del conjunto de entrenamiento. En este caso 1200.

Con 1200 dimensiones (muestras crudas), cada gaussiana necesita estimar una matriz de covarianza de 1200x1200 ≈ 720.000 parámetros. Con solo 1000 pulsos de entrenamiento, esto es un caso de "maldición de la dimensionalidad".

**Solución: extraer las características más relevantes.**

"""


"""**k ES EL Nº DE GAUSSIANAS QUE VA A COLOCAR EL MODELO Y EN FUNCIÓN A ESA GAUSSIANA SE VAN A DISTRIBUIR LOS PULSOS. Cuando lleguen pulsos nuevo, aquellos cuya verosimilitud sea baja (log alto) estarán lejos de esta gassiana**"""

"""## Isolation forest como detector de anomalías.


"""

# ===================== Entrenamiento  ===============================
# Se entrena SOLO con pulsos de luz, igual que el autoencoder entrenado solo con pulsos light.
# contamination: fracción esperada de anomalías en el conjunto de entrenamiento.
# Como entrenamos solo con luz, ponemos un valor bajo (0.05 = 5%).


# ── Curva ROC y AUC ───────────────────────────────────────────────────────
# Dark = anomalía = etiqueta 1
# Light = normal  = etiqueta 0
# Usamos -scores porque score más negativo = más anómalo

# ── Scatter plot PCA del espacio de trazas ────────────────────────────────
# A diferencia del GMM, aquí el PCA opera sobre las trazas directamente,
# no sobre un vector de características extraídas manualmente.

"""### CNN de regresión

La otra alternativa que se sugiere en el artículo, es usar la CNN como modelo de regresión. En este caso, en lugar de calsificar el pulso como 
Light o Dark (salida binaria, 1 ó 0). El modelo tratará de predecir la engergía que tiene un pulso de Light (salida continua, un número). 
La detección de fondo se hace indirectamente, si el modelo predice bien la energía de un pulso de luz es porque se parece a la señal. Si la predicción
es mala, es porque la morfología del pulso es diferente, Fondo.
Para la regresión debemos elegir que característica es la que se tratará de rpedecir (el target). El artículo menciona directamente el Vmin y el FFT.

Sin embargo, como en el pipeline no se ha implementado el ajuste en frecuencia, la alternativa directa es usar el mínimo de voltaje de la traza (V_min), que 
es el valor del pico del pulso en dominio temporal y es proporcional a la energía absorbida.
"""

# ==========================================
# PREPARACIÓN DEL TARGET DE REGRESIÓN (Vmin)
# ==========================================

# El target es el mínimo de cada traza (proporcional a la energía)
# Shape X_light, X_dark: (N, 1200) en V, con offset eliminado

y_light_reg = np.min(X_light, axis=1)  # (N,) valores negativos en V
y_dark_reg  = np.min(X_dark,  axis=1)  # (N,) valores negativos en V

print(f"Rango V_min light: {y_light_reg.min():.4f} a {y_light_reg.max():.4f} V")
print(f"Rango V_min dark:  {y_dark_reg.min():.4f} a {y_dark_reg.max():.4f} V")

# Visualizar distribuciones del target
plt.figure(figsize=(8, 4))
plt.hist(y_light_reg * 1000, bins=40, alpha=0.6,
         color='orange', density=True, label='Light')
plt.hist(y_dark_reg  * 1000, bins=40, alpha=0.6,
         color='black',  density=True, label='Dark')
plt.xlabel('V_min (mV)')
plt.ylabel('Densidad')
plt.title('Distribucion del target de regresion por clase')
plt.legend()
plt.grid(True, linewidth=0.4)
plt.show()

# ===============
# NORMALIZACIÓN
# ===============
# Las trazas se normalizan igual que en el autoencoder:
# con parámetros del conjunto de entrenamiento (solo light)
# porque el modelo aprende la distribución de la señal

# Normalización de las trazas (entrada)
mean_reg = np.mean(X_light)
std_reg  = np.std(X_light)

X_light_norm_reg = ((X_light - mean_reg) / std_reg)[..., np.newaxis]  # (N, 1200, 1)
X_dark_norm_reg  = ((X_dark  - mean_reg) / std_reg)[..., np.newaxis]  # (N, 1200, 1)

# Normalización del target (salida)
# Importante: el target también se normaliza para estabilizar el entrenamiento
mean_y = np.mean(y_light_reg)
std_y  = np.std(y_light_reg)

y_light_norm_reg = (y_light_reg - mean_y) / std_y  # (N,)
y_dark_norm_reg  = (y_dark_reg  - mean_y) / std_y  # (N,) con params de light

# Split train/val solo con light (mismo criterio que autoencoder)
X_l_train_reg, X_l_val_reg, y_l_train_reg, y_l_val_reg = train_test_split(
    X_light_norm_reg, y_light_norm_reg,
    test_size=0.2, random_state=42
)

print(f"Train: {X_l_train_reg.shape} | Val: {X_l_val_reg.shape}")

"""Seguimos la misma arquitectura de la CNN binaria del artículo. El cambio está en la salida, en lugar de 
sigmoide(0/1) se usa la función de activación lineal (valor continuo). También cambia la función de perdida 
cross binary entropy por MAE."""

# =================================
# ARQUITECTURA CNN DE REGRESIÓN
# =================================

def build_cnn_regresion(input_shape, num_conv_layers, num_filters,
                        kernel_size, dense_initial_units,
                        num_dense_layers, dropout_rate, learning_rate):

    model = models.Sequential()
    initializer = initializers.GlorotUniform()

    # Capas convolucionales (idénticas a la CNN binaria)
    for i in range(num_conv_layers):
        if i == 0:
            model.add(layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                activation='tanh',
                kernel_initializer=initializer,
                input_shape=input_shape
            ))
        else:
            model.add(layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                activation='tanh',
                kernel_initializer=initializer
            ))
        model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Flatten())

    # Capas densas (idénticas a la CNN binaria)
    units = dense_initial_units
    for _ in range(num_dense_layers):
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_initializer=initializer
        ))
        model.add(layers.Dropout(dropout_rate))
        units = units // 2

    # Salida de regresión: 1 neurona con activación lineal
    model.add(layers.Dense(1, activation='linear', name='output_regresion'))

    # Loss: MAE en lugar de binary_crossentropy
    # Métrica: MAE para monitorizar el error de predicción
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mae']
    )

    return model

cnn_reg = build_cnn_regresion(
    input_shape        = (1200, 1),
    num_conv_layers    = 6,
    num_filters        = 45,
    kernel_size        = 5,
    dense_initial_units= 188,
    num_dense_layers   = 3,
    dropout_rate       = 0.18,
    learning_rate      = 5.2e-4
)

cnn_reg.summary()

# ===============
# ENTRENAMIENTO
# ===============

history_reg = cnn_reg.fit(
    X_l_train_reg, y_l_train_reg,
    validation_data=(X_l_val_reg, y_l_val_reg),
    epochs=20,
    batch_size=99,
    shuffle=True
)

# Curva de aprendizaje
plt.figure(figsize=(7, 4))
plt.plot(history_reg.history['loss'],     color='steelblue', label='Train MAE')
plt.plot(history_reg.history['val_loss'], color='steelblue',
         linestyle='--', label='Val MAE')
plt.xlabel('Epoca')
plt.ylabel('MAE')
plt.title('Curva de aprendizaje — CNN Regresion')
plt.legend()
plt.grid(True, linewidth=0.4)
plt.tight_layout()
plt.show()

# ============================================================
# EVALUACIÓN: ERROR DE PREDICCIÓN COMO SCORE DE ANOMALÍA
# ============================================================
# El score de anomalía es el error absoluto entre
# el V_min real y el predicho por el modelo

y_pred_light = cnn_reg.predict(X_light_norm_reg, verbose=0).flatten()
y_pred_dark  = cnn_reg.predict(X_dark_norm_reg,  verbose=0).flatten()

# Error de predicción por pulso (en unidades normalizadas)
err_reg_light = np.abs(y_light_norm_reg - y_pred_light)
err_reg_dark  = np.abs(y_dark_norm_reg  - y_pred_dark)

# Distribución de errores
plt.figure(figsize=(8, 4))
plt.hist(err_reg_light, bins=40, alpha=0.6,
         color='orange', density=True, label='Light')
plt.hist(err_reg_dark,  bins=40, alpha=0.6,
         color='black',  density=True, label='Dark')
plt.xlabel('Error de prediccion de V_min (unidades normalizadas)')
plt.ylabel('Densidad')
plt.title('Distribucion del error de prediccion — CNN Regresion')
plt.legend()
plt.grid(True, linewidth=0.4)
plt.tight_layout()
plt.show()

# Scatter plot: V_min real vs predicho
plt.figure(figsize=(7, 6))
plt.scatter(y_light_norm_reg, y_pred_light,
            c='orange', alpha=0.4, s=10, label='Light')
plt.scatter(y_dark_norm_reg,  y_pred_dark,
            c='black',  alpha=0.4, s=10, label='Dark')
plt.plot([-3, 3], [-3, 3], 'r--', linewidth=1, label='Prediccion perfecta')
plt.xlabel('V_min real (normalizado)')
plt.ylabel('V_min predicho (normalizado)')
plt.title('V_min real vs predicho — CNN Regresion')
plt.legend()
plt.grid(True, linewidth=0.4)
plt.tight_layout()
plt.show()

# ================
# CURVA ROC
# ================
from sklearn.metrics import roc_curve, auc

scores_all = np.concatenate([err_reg_light, err_reg_dark])
labels_all = np.concatenate([
    np.zeros(len(err_reg_light)),   # light = normal = 0
    np.ones(len(err_reg_dark))      # dark  = anomalia = 1
])

fpr_reg, tpr_reg, _ = roc_curve(labels_all, scores_all)
roc_auc_reg = auc(fpr_reg, tpr_reg)

plt.figure(figsize=(7, 6))
plt.plot(fpr_reg, tpr_reg, color='crimson',
         label=f'CNN Regresion (AUC={roc_auc_reg:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC — CNN Regresion')
plt.legend()
plt.grid(True, linewidth=0.4)
plt.tight_layout()
plt.show()

print(f"CNN Regresion AUC: {roc_auc_reg:.4f}")

"""  A partir de este este punto estudiamos el rendimiento de cada modelo. 
Empezamos representando las las cruvas ROC de todos los modelos."""

plt.figure(figsize=(8, 7))
# CNN MODEL
...
plt.plot(fpr_cnn, tpr_cnn, color='steelblue',
         label=f'CNN (AUC={roc_auc:.3f})', linewidth=2)

# Isolation Forest
plt.plot(fpr, tpr, color='darkgreen',
         label=f'Isolation Forest (AUC={roc_auc:.3f})', linewidth=2)

# Autoencoder Solo Light (recupera fpr/tpr del bloque anterior)
fpr_ae, tpr_ae, _ = roc_curve(
    np.concatenate([np.zeros(len(err_light_on_light)),
                    np.ones(len(err_light_on_dark))]),
    np.concatenate([err_light_on_light, err_light_on_dark])
)
plt.plot(fpr_ae, tpr_ae, color='orange',
         label=f'Autoencoder Solo Light (AUC=0.908)', linewidth=2)

# Autoencoder Light+Dark
fpr_ld, tpr_ld, _ = roc_curve(
    np.concatenate([np.zeros(len(err_both_on_light)),
                    np.ones(len(err_both_on_dark))]),
    np.concatenate([err_both_on_light, err_both_on_dark])
)
plt.plot(fpr_ld, tpr_ld, color='seagreen',
         label=f'Autoencoder Light+Dark (AUC=0.666)', linewidth=2, linestyle='--')

# Autoencoder Solo Dark
fpr_sd, tpr_sd, _ = roc_curve(
    np.concatenate([np.zeros(len(err_dark_on_light)),
                    np.ones(len(err_dark_on_dark))]),
    np.concatenate([err_dark_on_light, err_dark_on_dark])
)
plt.plot(fpr_sd, tpr_sd, color='steelblue',
         label=f'Autoencoder Solo Dark (AUC=0.352)', linewidth=2, linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Clasificador aleatorio')
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curvas ROC — Comparación de todos los modelos')
plt.legend(loc='lower right')
plt.grid(True, linewidth=0.4)
plt.tight_layout()
plt.savefig("roc_comparacion_completa.png", dpi=120)
plt.show()