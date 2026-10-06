import os
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

FIGURAS_DIR = "figuras_memoria"

def guardar_figura(nombre):
    """Guardar la figura actual"""
    os.makedirs(FIGURAS_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURAS_DIR, f"{nombre}.png"), dpi=150, bbox_inches="tight")

""" BLOQUE MENU PRINCIPAL """

def mostrar_menu():
    """Muestra el menú principal."""

    print("\n" + "=" * 50)
    print("      TFG - DISCRIMINACIÓN DE RUIDO")
    print("=" * 50)
    print("1. Mostrar jerarquía datasets")
    print("2. Cargar datasets")
    print("3. Estudio de datasets")
    print("4. Selección de ventana de muestras")
    print("5. CNN de clasificación binaria (entrenar y evaluar)")
    print("6. Autoencoders (entrenar y evaluar)")
    print("7. Gaussian Mixture Model - GMM (entrenar y evaluar)")
    print("8. Isolation Forest")
    print("9. Modelo CNN de regresión")
    print("10. Comparación de modelos")
    print("0. Salir")
    print("=" * 50)

def main():

    # Variables de estado
    light_dataset_raw = None
    dark_dataset_raw  = None
    LIGHT_PULSES = None
    DARK_PULSES  = None
    X_light = X_dark = X_light_test = X_dark_test = None
    model = history = X_test_cnn = y_test_cnn = None
    y_proba = y_pred = roc_auc_cnn = fpr_cnn = tpr_cnn = None
    resultados_autoencoders = None
    errores_autoencoders    = None
    gmm = None
    resultados_gmm = None
    resultados_if = None
    resultados_reg = None
    err_reg_light = err_reg_dark = roc_auc_reg = fpr_reg = tpr_reg = None

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
        # Comprobación de pasos previos
        # --------------------------------------------------
        if not comprobar_pasos_previos(opcion, locals()):
            continue

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
            LIGHT_PULSES, DARK_PULSES = cargar_datasets(light_dataset_raw, dark_dataset_raw)
        # --------------------------------------------------
        # 3. Estudio datasets
        # --------------------------------------------------
        elif opcion == "3":
            n = int(input("Número de trazas a visualizar: "))
            LIGHT_PULSES_SIN_OFFSET, DARK_PULSES_SIN_OFFSET = estudio_datasets(LIGHT_PULSES,DARK_PULSES,n,1200,2600)
        # --------------------------------------------------
        # 4. Selección ventana
        # --------------------------------------------------
        elif opcion == "4":
            n_samples = int(input("Número de muestras por clase para entrenamiento: "))
            inicio = int(input("Inicio ventana de muestras: "))
            fin = int(input("Fin ventana de muestras: "))
            X_light, X_dark, X_light_test, X_dark_test = seleccionar_ventana_muestras(LIGHT_PULSES_SIN_OFFSET, DARK_PULSES_SIN_OFFSET,n_samples, inicio, fin)
        # --------------------------------------------------
        # 5. CNN de clasificación binaria
        # --------------------------------------------------
        elif opcion == "5":

            print("\n[Módulo CNN Binaria]")
            model, history, X_test_cnn, y_test_cnn = entrenar_modelo_cnn(X_light, X_dark, X_light_test, X_dark_test)
            y_proba, y_pred, roc_auc_cnn, fpr_cnn, tpr_cnn = evaluar_cnn(model, history, X_test_cnn, y_test_cnn)
            print("\n Fin CNN Binaria ")
        # --------------------------------------------------
        # 6. AUTOENCODERS
        # --------------------------------------------------
        elif opcion == "6":
            print("\n[Módulo Autoencoders]")
            resultados_autoencoders = entrenar_autoencoders(X_light, X_dark, X_light_test, X_dark_test)
            print("Entrenamiento completado.")
            errores_autoencoders = evaluar_autoencoders(resultados_autoencoders)
            print("Fin de la evaluación.")

        # --------------------------------------------------
        # 7. GMM
        # --------------------------------------------------
        elif opcion == "7":
            print("\n[Módulo GMM]")
            gmm = entrenar_gmm(X_light, X_dark)
            print("Entrenamiento modelo GMM Completado.")
            print("Evaluación modelo GMM.")
            resultados_gmm = evaluar_gmm(gmm)
        # --------------------------------------------------
        # 8. Isolation Forest
        # --------------------------------------------------
        elif opcion == "8":
            print("\n[Módulo Isolation Forest]")
            resultados_if = entrenar_y_evaluar_isolation_forest(X_light, X_dark)

        # --------------------------------------------------
        # 9. MODELO CNN DE REGRESION
        # --------------------------------------------------
        elif opcion == "9":
            print("\n[Módulo CNN de regresión]")
            resultados_reg = entrenar_cnn_regresion(X_light, X_dark)
            err_reg_light, err_reg_dark, roc_auc_reg, fpr_reg, tpr_reg = evaluar_cnn_regresion(resultados_reg)
        # --------------------------------------------------
        # 10. Comparación
        # --------------------------------------------------
        elif opcion == "10":
            print("\nComparación de modelos.")
            comparar_modelos(roc_auc_cnn, fpr_cnn, tpr_cnn, errores_autoencoders, resultados_gmm["roc_auc_gmm"],
                             resultados_gmm["fpr_gmm"], resultados_gmm["tpr_gmm"],
                             resultados_if["roc_auc_if"], resultados_if["fpr_if"], resultados_if["tpr_if"],
                             roc_auc_reg, fpr_reg, tpr_reg,
                             nombre_gmm=resultados_gmm["nombre_mejor_gmm"])

        # --------------------------------------------------

        else:
            print("Opción no válida.")


# ================================================
# --------- BLOQUE FUNCIONES AUXILIARES ---------
# ===============================================
#validacion por si falta un paso previo
PASOS_PREVIOS = {
    "2":  ["light_dataset_raw", "dark_dataset_raw"],
    "3":  ["LIGHT_PULSES"],
    "4":  ["LIGHT_PULSES"],
    "5":  ["X_light"],
    "6":  ["X_light"],
    "7":  ["X_light"],
    "8":  ["X_light"],
    "9":  ["X_light"],
    "10": ["roc_auc_cnn", "errores_autoencoders", "resultados_gmm", "resultados_if", "roc_auc_reg"],
}


def comprobar_pasos_previos(opcion, variables_locales):
    """
    OBJ: Comprueba si las variables necesarias para ejecutar 'opcion'
         ya existen (no son None).
    PARAMS:
        opcion : (str) Opción del menú seleccionada.
        variables_locales : (dict) resultado de locals() en el punto de llamada.
    RETURNS:
        (bool) True si todas las dependencias están satisfechas, False si falta alguna.
    """
    for nombre_var in PASOS_PREVIOS.get(opcion, []):
        if variables_locales.get(nombre_var) is None:
            print("\nHay un paso previo pendiente de procesar.")
            return False
    return True

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

def cargar_datasets(light_dataset_raw, dark_dataset_raw):
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

    trazas_light_sin_offset = []
    trazas_dark_sin_offset = []

    for traza in light_dataset:
        offset = np.mean(traza[:1200]) if ventana_ini > 0 else 0.0 #1200 es el limite antes de empezar un pulso
        trazas_light_sin_offset.append(traza - offset)
    for traza in dark_dataset:
        offset = np.mean(traza[:1200]) if ventana_ini > 0 else 0.0
        trazas_dark_sin_offset.append(traza - offset)

    # ===================================
    # Trazas LIGHT individuales en escala temporal (Con vs Sin Offset)
    # ===================================

    fig, axes = plt.subplots(n_trazas, 2, figsize=(18, 2 * n_trazas))
    if n_trazas == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_trazas):
        t = np.arange(len(light_dataset[i])) / 50
        # Con offset
        axes[i, 0].plot(t, light_dataset[i] * 1000)
        axes[i, 0].set_ylabel("V (mV)")
        axes[i, 0].set_xlabel("t (µs)")
        axes[i, 0].set_title(f"Traza LIGHT {i} (Con offset)")
        axes[i, 0].yaxis.set_major_locator(ticker.MultipleLocator(5))

        # Sin offset
        axes[i, 1].plot(t, trazas_light_sin_offset[i] * 1000, color="orange")
        axes[i, 1].set_ylabel("V (mV)")
        axes[i, 1].set_xlabel("t (µs)")
        axes[i, 1].set_title(f"Traza LIGHT {i} (Sin offset)")
        axes[i, 1].yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    #guardar_figura("comparativa_trazas_temporales_light")
    plt.show()

    # ===================================
    # Primeras trazas LIGHT (Con vs Sin Offset)
    # ===================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # izquierda: Con offset
    for i in range(n_trazas):
        ax1.plot(light_dataset[i], alpha=0.7)
    ax1.set_title(f"Primeras {n_trazas} trazas LIGHT (Con offset)")
    ax1.set_xlabel("Muestras")
    ax1.set_ylabel("Vout (V)")
    ax1.grid(color='gray', linestyle='--', linewidth=0.2)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.0025))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(600))

    # derecha: Sin offset
    for i in range(n_trazas):
        ax2.plot(trazas_light_sin_offset[i], alpha=0.7)
    ax2.set_title(f"Primeras {n_trazas} trazas LIGHT (Sin offset)")
    ax2.set_xlabel("Muestras")
    ax2.set_ylabel("Vout (V)")
    ax2.grid(color='gray', linestyle='--', linewidth=0.2)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.0025))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(600))

    plt.tight_layout()
    #guardar_figura("comparativa_primeras_trazas_light")
    plt.show()

    # ===================================
    # Primeras trazas DARK (Con vs Sin Offset)
    # ===================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # izquierda: Con offset
    for i in range(n_trazas):
        ax1.plot(dark_dataset[i], alpha=0.7)
    ax1.set_title(f"Primeras {n_trazas} trazas DARK (Con offset)")
    ax1.set_xlabel("Muestras")
    ax1.set_ylabel("Vout (V)")
    ax1.grid(color='gray', linestyle='--', linewidth=0.2)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.0025))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(600))

    # derecha: Sin offset
    for i in range(n_trazas):
        ax2.plot(trazas_dark_sin_offset[i], alpha=0.7)
    ax2.set_title(f"Primeras {n_trazas} trazas DARK (Sin offset)")
    ax2.set_xlabel("Muestras")
    ax2.set_ylabel("Vout (V)")
    ax2.grid(color='gray', linestyle='--', linewidth=0.2)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.0025))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(600))

    plt.tight_layout()
    #guardar_figura("comparativa_trazas_dark")
    plt.show()

    # ===================================
    # Subplots de Pulsos Medios (con vs sin offset)
    # ===================================

    mean_light = np.mean(light_dataset, axis=0)
    mean_dark = np.mean(dark_dataset, axis=0)

    mean_light_sin_offset = np.mean(trazas_light_sin_offset, axis=0)
    mean_dark_sin_offset = np.mean(trazas_dark_sin_offset, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # izquierda: Con offset
    ax1.plot(mean_light, label="Light (Con offset)", color="orange")
    ax1.plot(mean_dark, label="Background (Con offset)", color="black")
    ax1.axvline(x=ventana_ini, color='green', linestyle='--', label='Ventana inicio')
    ax1.axvline(x=ventana_fin, color='red', linestyle='--', label='Ventana fin')
    ax1.set_title("Pulso medio CON offset")
    ax1.set_ylabel("Vout (V)")
    ax1.set_xlabel("Muestras")
    ax1.grid(color='gray', linestyle='--', linewidth=0.2)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.0025))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(600))
    ax1.legend()

    # derecha: Sin offset
    ax2.plot(mean_light_sin_offset, label="Light (Sin offset)", color="orange")
    ax2.plot(mean_dark_sin_offset, label="Background (Sin offset)", color="black")
    ax2.axvline(x=ventana_ini, color='green', linestyle='--', label='Ventana inicio')
    ax2.axvline(x=ventana_fin, color='red', linestyle='--', label='Ventana fin')
    ax2.set_title("Pulso medio SIN offset (Baseline corregido)")
    ax2.set_ylabel("Vout (V)")
    ax2.set_xlabel("Muestras")
    ax2.grid(color='gray', linestyle='--', linewidth=0.2)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.0025))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(600))
    ax2.legend()

    plt.tight_layout()
    #guardar_figura("estudio_pulso_medio_ventana")
    plt.show()
    
    return trazas_light_sin_offset, trazas_dark_sin_offset

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
    trazas_light_scatter = []
    trazas_dark_scatter = []
    trazas_light_test = []
    trazas_dark_test = []

    for traza in LIGHT_PULSES[:10]: #4700
        offset = np.mean(traza[:1000]) if inicio > 0 else 0.0 #se puede modificar el limite hasta 1
        trazas_light_scatter.append(traza[1000:5000] - offset)
    for traza in DARK_PULSES[:4700]:
        offset = np.mean(traza[:1000]) if inicio > 0 else 0.0
        trazas_dark_scatter.append(traza[1000:5000] - offset)

    for traza in LIGHT_PULSES[:n_samples]:
        offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
        trazas_light.append(traza[inicio:fin] - offset)
    for traza in DARK_PULSES[:n_samples]:
        offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
        trazas_dark.append(traza[inicio:fin] - offset)
    for traza in LIGHT_PULSES[n_samples:]:
        offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
        trazas_light_test.append(traza[inicio:fin] - offset)
    for traza in DARK_PULSES[n_samples:]:
        offset = np.mean(traza[:inicio]) if inicio > 0 else 0.0
        trazas_dark_test.append(traza[inicio:fin] - offset)

    input_len = fin - inicio
    print(f"Ventana seleccionada: [{inicio}:{fin}]")
    print(f"Longitud de traza resultante: {input_len} muestras "
          f"({input_len / 50:.1f} µs a 50 MHz)")
    
    # ===================================
    # Scatter plots
    # ===================================

    X_all = np.vstack([trazas_light_scatter, trazas_dark_scatter])   # (N+M, 1200)

    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X_all)          # (N+M, 2)

    varianza = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(X_2d[:len(trazas_light_scatter), 0], X_2d[:len(trazas_light_scatter), 1],
               c='orange', alpha=0.3,  s=12, label='Light', zorder=3)
    ax.scatter(X_2d[len(trazas_light_scatter):, 0], X_2d[len(trazas_light_scatter):, 1],
               c='black',  alpha=0.15, s=12, label='Dark',  zorder=3)

    ax.set_xlabel(f'PC1 ({varianza[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({varianza[1]*100:.1f}%)')
    ax.set_title('PCA sobre pulsos iniciales')
    ax.legend()

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(which='major', color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.grid(which='minor', color='gray', linestyle=':',  linewidth=0.3, zorder=0)

    plt.tight_layout()
    #guardar_figura("pca_trazas_crudas")
    plt.show()

    print(f"Varianza explicada — PC1: {varianza[0]*100:.1f}%  "
          f"PC2: {varianza[1]*100:.1f}%  "
          f"Total: {sum(varianza)*100:.1f}%")

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

    # Normalización (test con las estadísticas de train)
    mean_train = np.mean(X)
    std_train = np.std(X)
    X = ((X - mean_train) / std_train)[..., np.newaxis]
    X_test = ((X_test - mean_train) / std_train)[..., np.newaxis]

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
    #guardar_figura("cnn_binaria_curva_accuracy")
    plt.show()
    print("\n")
    loss, accuracy = model.evaluate(X_test_cnn, y_test_cnn)

    print(f"Validation accuracy: {accuracy:.4f}")
    print("\n")

    # Predicciones
    y_proba = model.predict(X_test_cnn,verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    # Informe
    print(classification_report(y_test_cnn,y_pred,target_names=["Dark", "Light"]))

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
    #guardar_figura("cnn_binaria_matriz_confusion")
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
    #guardar_figura("cnn_binaria_roc")
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
    comparar_roc_autoencoders(errores_autoencoders)
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
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax.legend()
        ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    #guardar_figura("autoencoders_curvas_perdida")
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

    # --- Autoencoder dark ---
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
    #guardar_figura("autoencoders_error_reconstruccion")
    plt.show()

    errores_autoencoders = {
        "light":    {"err_on_light": err_light_on_light, "err_on_dark": err_light_on_dark},
        "dark":     {"err_on_light": err_dark_on_light,  "err_on_dark": err_dark_on_dark},
        "combined": {"err_on_light": err_both_on_light,   "err_on_dark": err_both_on_dark}
    }

    return errores_autoencoders

def comparar_roc_autoencoders(errores_autoencoders):
    """
    OBJ: Calcular y comparar la curva ROC/AUC de las 3 variantes de
         autoencoder (Solo Light, Solo Dark, Light+Dark), para poder ver
         qué variante discrimina mejor antes de pasar a la comparación
         final de todos los modelos (opción 10 del menú).
    PARAMS:
        errores_autoencoders : (dict) salida de
            calcula_error_reconstruccion_autoencoders().
    RETURNS:
        resultados_roc : (dict) por variante, con fpr/tpr/auc/titulo.
        mejor_variante : (str) clave ("light"/"dark"/"combined") de la
                          variante con mayor AUC.
    """
    colores_ae = {'light': 'orange', 'dark': 'steelblue', 'combined': 'seagreen'}

    resultados_roc = calcular_roc_autoencoders(errores_autoencoders)

    plt.figure(figsize=(7, 6))
    for variante, r in resultados_roc.items():
        plt.plot(r["fpr"], r["tpr"], color=colores_ae[variante], linewidth=2,
                  label=f'{r["titulo"]} (AUC={r["auc"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Comparación de las 3 variantes de Autoencoder')
    plt.legend(loc='lower right')
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    #guardar_figura("autoencoders_roc_comparacion_variantes")
    plt.show()

    mejor_variante = max(resultados_roc, key=lambda k: resultados_roc[k]["auc"])
    print(f"Mejor variante de Autoencoder: {resultados_roc[mejor_variante]['titulo']} "
          f"(AUC={resultados_roc[mejor_variante]['auc']:.3f})")

    return resultados_roc, mejor_variante


def calcular_roc_autoencoders(errores_autoencoders):
    """
    OBJ: Calcular fpr/tpr/AUC de las 3 variantes de autoencoder a partir de
         sus errores de reconstrucción, teniendo en cuenta que la clase
         considerada "anomalía" depende de con qué clase se entrenó cada
         variante (igual criterio que evaluar_gmm(): con Light se entrena
         para reconocer Light como normal y Dark como anomalía; con Dark,
         al revés; con Light+Dark no hay clase "no vista", se reporta con
         Dark como anomalía a modo de referencia). Función pura (sin
         gráficas), reutilizada tanto por comparar_roc_autoencoders() como
         por comparar_modelos().
    PARAMS:
        errores_autoencoders : (dict) salida de
            calcula_error_reconstruccion_autoencoders().
    RETURNS:
        resultados_roc : (dict) por variante, con fpr/tpr/auc/titulo.
    """
    etiquetas_ae = {'light': 'Autoencoder Solo Light',
                     'dark': 'Autoencoder Solo Dark',
                     'combined': 'Autoencoder Light+Dark'}
    clase_anomala_ae = {'light': 'dark', 'dark': 'light', 'combined': 'dark'}

    resultados_roc = {}
    for variante, errores in errores_autoencoders.items():
        err_on_light = errores["err_on_light"]
        err_on_dark  = errores["err_on_dark"]

        scores = np.concatenate([err_on_light, err_on_dark])
        if clase_anomala_ae[variante] == 'dark':
            labels = np.concatenate([np.zeros(len(err_on_light)), np.ones(len(err_on_dark))])
        else:
            labels = np.concatenate([np.ones(len(err_on_light)), np.zeros(len(err_on_dark))])

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        resultados_roc[variante] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "titulo": etiquetas_ae[variante]}

    return resultados_roc

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
        "Espacio latente — Entrenado con Light",
        "scatter_latente_light"
    )

    scatter_latente(
        resultados["dark"]["encoder"],
        resultados["dark"]["X_train"],
        resultados["dark"]["X_test"],
        "Espacio latente — Entrenado con Dark",
        "scatter_latente_dark"
    )

    scatter_latente(
        resultados["combined"]["encoder"],
        resultados["combined"]["X_test"][0],
        resultados["combined"]["X_test"][1],
        "Espacio latente — Entrenado con Light + Dark",
        "scatter_latente_combinado"
    )

def scatter_latente(encoder, X_light, X_dark, titulo, nombre_archivo):
    z_light = encoder.predict(X_light, verbose=0)
    z_dark  = encoder.predict(X_dark,  verbose=0)

    z_all = np.vstack([z_light, z_dark])
    pca   = PCA(n_components=2)
    z_2d  = pca.fit_transform(z_all)

    varianza = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(z_2d[:len(z_light), 0], z_2d[:len(z_light), 1],
               c='orange', alpha=0.3, s=12, label='Light', zorder=3)
    ax.scatter(z_2d[len(z_light):, 0], z_2d[len(z_light):, 1],
               c='black',  alpha=0.15, s=12, label='Dark', zorder=3)

    ax.set_xlabel(f'PC1 ({varianza[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({varianza[1]*100:.1f}%)')
    ax.set_title(titulo)
    ax.legend()

    # Cuadrícula más densa (espaciado adaptado al rango real de cada gráfica,
    # en vez de un paso fijo que puede saturar el eje si el rango es amplio)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.grid(which='major', color='gray',  linestyle='--', linewidth=0.5, zorder=0)
    ax.grid(which='minor', color='gray',  linestyle=':',  linewidth=0.3, zorder=0)

    plt.tight_layout()
    #guardar_figura(nombre_archivo)
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

def comparar_bic(F_train, titulo, nombre_figura, max_componentes=10):
    """
    OBJ: Calcular el criterio BIC para distintos números de componentes
         gaussianas y seleccionar el que lo minimiza.
    PARAMS:
        F_train : (np.ndarray) Características normalizadas de entrenamiento.
        titulo : (str) Nombre de la variante (para el título de la gráfica).
        nombre_figura : (str) Nombre de fichero para guardar_figura().
        max_componentes : (int) Número máximo de gaussianas a evaluar.
    RETURNS:
        mejor_k : (int) Número de componentes con el BIC mínimo.
    """
    componentes = range(1, max_componentes + 1)
    bic = []

    for k in componentes:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=5)
        gmm.fit(F_train)
        bic.append(gmm.bic(F_train))

    plt.figure(figsize=(7, 5))
    plt.plot(componentes, bic, marker='o')
    plt.xlabel('Número de gaussianas')
    plt.ylabel('BIC')
    plt.title(f'Selección del número de gaussianas — {titulo}')
    plt.grid(True)
    plt.tight_layout()
    #guardar_figura(nombre_figura)
    plt.show()

    mejor_k = componentes[bic.index(min(bic))]
    print(f"[{titulo}] Mejor número de gaussianas: {mejor_k} (BIC mínimo: {min(bic):.2f})")

    return mejor_k


def _entrenar_gmm_variante(F_train_norm, titulo, nombre_figura_bic):
    """Selecciona el nº de componentes vía BIC y entrena el GMM final de esa variante."""
    n_components = comparar_bic(F_train_norm, titulo, nombre_figura_bic)

    gmm = GaussianMixture(
        n_components    = n_components,
        covariance_type = 'full',
        random_state    = 42,
        max_iter        = 200,
        n_init          = n_components
    )
    gmm.fit(F_train_norm)

    print(f"=== {titulo} ===")
    print(f"Componentes : {n_components}")
    print(f"Convergencia: {gmm.converged_}")
    print(f"Iteraciones : {gmm.n_iter_}")

    return gmm


def entrenar_gmm(X_light, X_dark):
    """
    OBJ: Entrena tres GMM (solo Light, solo Dark, y Light+Dark combinado),
         siguiendo el mismo esquema que entrenar_autoencoders(): tres
         variantes independientes, evaluadas después conjuntamente. El
         número de componentes gaussianas de cada variante se selecciona
         automáticamente mediante el criterio BIC (ver comparar_bic()).
    PARAMS:
        X_light, X_dark : (np.ndarray) Pulsos LIGHT y DARK de entrenamiento
                           (trazas crudas, sin canal).
    RETURNS:
        resultados : (dict) con el modelo, scaler y características de
                     evaluación (ya normalizadas) de cada variante.
    """
    print("\n[Entrenamiento GMM]")

    # Extracción de características (v_min, tau_rise, tau_decay)
    F_light = extraer_caracteristicas(X_light)
    F_dark  = extraer_caracteristicas(X_dark)
    F_both  = np.vstack([F_light, F_dark])

    # Comprobación de NaN/Inf
    print(f"NaN en F_light: {np.isnan(F_light).sum()} | NaN en F_dark: {np.isnan(F_dark).sum()}")
    print(f"Inf en F_light: {np.isinf(F_light).sum()} | Inf en F_dark: {np.isinf(F_dark).sum()}")

    # División train/val de cada conjunto
    F_light_train, F_light_val = train_test_split(F_light, test_size=0.2, random_state=42)
    F_dark_train,  F_dark_val  = train_test_split(F_dark,  test_size=0.2, random_state=42)
    F_both_train,  F_both_val  = train_test_split(F_both,  test_size=0.2, random_state=42)

    # Normalización: cada variante usa los estadísticos de SU PROPIO conjunto
    # de entrenamiento; esos mismos estadísticos se reutilizan para
    # normalizar la clase contraria en la evaluación (sin fuga de información).
    F_light_train_norm, F_light_val_norm, scaler_light = normalizar_caracteristicas(F_light_train, F_light_val)
    F_dark_train_norm,  F_dark_val_norm,  scaler_dark  = normalizar_caracteristicas(F_dark_train,  F_dark_val)
    F_both_train_norm,  F_both_val_norm,  scaler_both  = normalizar_caracteristicas(F_both_train,  F_both_val)

    gmm_light = _entrenar_gmm_variante(F_light_train_norm, "GMM — Solo Light",     "gmm_bic_light")
    gmm_dark  = _entrenar_gmm_variante(F_dark_train_norm,  "GMM — Solo Dark",      "gmm_bic_dark")
    gmm_both  = _entrenar_gmm_variante(F_both_train_norm,  "GMM — Light + Dark",   "gmm_bic_combinado")

    print("Entrenamiento GMM completado.")

    resultados = {
        "light": {
            "modelo": gmm_light,
            "scaler": scaler_light,
            "F_light_norm": F_light_val_norm,
            "F_dark_norm": scaler_light.transform(F_dark_val),
        },
        "dark": {
            "modelo": gmm_dark,
            "scaler": scaler_dark,
            "F_light_norm": scaler_dark.transform(F_light_val),
            "F_dark_norm": F_dark_val_norm,
        },
        "combined": {
            "modelo": gmm_both,
            "scaler": scaler_both,
            "F_light_norm": scaler_both.transform(F_light_val),
            "F_dark_norm": scaler_both.transform(F_dark_val),
        },
    }

    return resultados


def calcular_scores_gmm(gmm, F_light_norm, F_dark_norm):
    """Score de anomalía = log-verosimilitud negada (score alto = más anómalo)."""
    scores_light = -gmm.score_samples(F_light_norm)
    scores_dark  = -gmm.score_samples(F_dark_norm)
    return scores_light, scores_dark


def evaluar_gmm(resultados):
    """
    OBJ: Evalúa las 3 variantes de GMM (solo Light, solo Dark, Light+Dark):
         histogramas de score comparados y curva ROC/AUC de cada una. La
         clase considerada "anomalía" depende de con qué clase se entrenó
         cada variante (con Light se entrena para reconocer Light como
         normal y Dark como anomalía; con Dark, al revés).
    PARAMS:
        resultados : (dict) salida de entrenar_gmm().
    RETURNS:
        resultados_gmm : (dict) con el detalle (fpr/tpr/auc) de las 3
                          variantes y, para mantener compatibilidad con
                          comparar_modelos(), las claves roc_auc_gmm /
                          fpr_gmm / tpr_gmm / nombre_mejor_gmm apuntando
                          a la variante con mayor AUC.
    """
    print("\n[Evaluación GMM]")

    # (clave, título, clase considerada "anomalía" para esa variante)
    variantes = [
        ("light",    "GMM — Solo Light",   "dark"),
        ("dark",     "GMM — Solo Dark",    "light"),
        ("combined", "GMM — Light + Dark", "dark"),
    ]

    # Distribución de scores de las 3 variantes
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (clave, titulo, _) in zip(axes, variantes):
        r = resultados[clave]
        scores_light, scores_dark = calcular_scores_gmm(r["modelo"], r["F_light_norm"], r["F_dark_norm"])
        r["scores_light"], r["scores_dark"] = scores_light, scores_dark

        ax.hist(scores_light, bins=40, alpha=0.6, color='orange', label='Light', density=True)
        ax.hist(scores_dark,  bins=40, alpha=0.6, color='black',  label='Dark',  density=True)
        ax.set_title(titulo)
        ax.set_xlabel('Score de anomalía (-log-verosimilitud)')
        ax.set_ylabel('Densidad')
        ax.legend()
        ax.grid(True, linewidth=0.4)

    plt.suptitle('Distribución del score de anomalía por variante', y=1.02)
    plt.tight_layout()
    #guardar_figura("gmm_distribucion_score")
    plt.show()

    # ROC de cada variante (positivo=1 -> clase considerada anómala en esa variante)
    resultados_evaluacion = {}
    for clave, titulo, clase_anomala in variantes:
        r = resultados[clave]
        scores = np.concatenate([r["scores_light"], r["scores_dark"]])
        if clase_anomala == "dark":
            y_true = np.concatenate([np.zeros(len(r["scores_light"])), np.ones(len(r["scores_dark"]))])
        else:
            y_true = np.concatenate([np.ones(len(r["scores_light"])), np.zeros(len(r["scores_dark"]))])

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        resultados_evaluacion[clave] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "titulo": titulo}
        print(f"{titulo}: AUC = {roc_auc:.3f}")

    # Mejor variante (mayor AUC), la que se usará como "el GMM" en la
    # comparación final de todos los modelos (opción 10 del menú)
    mejor_variante = max(resultados_evaluacion, key=lambda k: resultados_evaluacion[k]["auc"])
    r_mejor = resultados_evaluacion[mejor_variante]
    print(f"Mejor variante de GMM: {r_mejor['titulo']} (AUC={r_mejor['auc']:.3f})")

    plt.figure(figsize=(7, 6))
    plt.plot(r_mejor["fpr"], r_mejor["tpr"], color='purple', linewidth=2, label=f'{r_mejor["titulo"]} (AUC={r_mejor["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC — GMM (mejor variante)')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    #guardar_figura("gmm_roc")
    plt.show()

    # Comparación de las 3 variantes
    colores = {"light": "steelblue", "dark": "darkorange", "combined": "seagreen"}
    plt.figure(figsize=(8, 6))
    for clave, titulo, _ in variantes:
        r = resultados_evaluacion[clave]
        plt.plot(r["fpr"], r["tpr"], color=colores[clave], linewidth=2, label=f'{titulo} (AUC={r["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Comparación de las 3 variantes de GMM')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    #guardar_figura("gmm_roc_comparacion_variantes")
    plt.show()

    resultados_gmm = {
        "light":    resultados_evaluacion["light"],
        "dark":     resultados_evaluacion["dark"],
        "combined": resultados_evaluacion["combined"],
        # Compatibilidad con comparar_modelos(): se usa la mejor variante
        "roc_auc_gmm":       r_mejor["auc"],
        "fpr_gmm":           r_mejor["fpr"],
        "tpr_gmm":           r_mejor["tpr"],
        "nombre_mejor_gmm":  r_mejor["titulo"],
    }

    return resultados_gmm

def entrenar_y_evaluar_isolation_forest(X_light, X_dark, n_estimators=100, contamination=0.05):
    """
    OBJ: Entrenar Isolation Forest sobre las trazas normalizadas (sin
        extracción de características) y evaluar su capacidad de discriminación.
        Entrena SOLO con el 80% de los pulsos LIGHT (mismo criterio que
        Autoencoder-Light, GMM-Light y CNN de regresión), evaluando sobre el
        20% de LIGHT restante (no visto en el ajuste) y sobre todos los DARK.
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

    # División train/val de LIGHT (mismo criterio que el resto de modelos de
    # una clase: Autoencoder-Light, GMM-Light y CNN de regresión), para
    # evaluar sobre pulsos LIGHT que el modelo no ha visto en el ajuste.
    X_light_train, X_light_val = train_test_split(X_light, test_size=0.2, random_state=42)

    # Normalización con parámetros del conjunto de entrenamiento LIGHT
    mean_if = np.mean(X_light_train)
    std_if  = np.std(X_light_train)

    X_light_train_if_norm = (X_light_train - mean_if) / std_if
    X_light_val_if_norm   = (X_light_val   - mean_if) / std_if
    X_dark_if_norm        = (X_dark        - mean_if) / std_if

    print(f"Shape light train para IF: {X_light_train_if_norm.shape}")
    print(f"Shape light val para IF:   {X_light_val_if_norm.shape}")
    print(f"Shape dark para IF:        {X_dark_if_norm.shape}")

    modelo_if = IsolationForest(
        n_estimators  = n_estimators,
        max_samples   = 'auto',
        contamination = contamination,
        random_state  = 42,
        n_jobs        = -1
    )
    modelo_if.fit(X_light_train_if_norm)
    print("Entrenamiento completado.")

    print("\n[Evaluación Isolation Forest]")

    # score_samples: valores más negativos = más anómalos (evaluado sobre
    # LIGHT de validación, no visto en el entrenamiento)
    scores_light = modelo_if.score_samples(X_light_val_if_norm)
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
    #guardar_figura("isolation_forest_distribucion_score")
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
    #guardar_figura("isolation_forest_roc")
    plt.show()

    print(f"Isolation Forest AUC: {roc_auc_if:.4f}")

    # ── Scatter plot PCA del espacio de trazas ────────────────────────────────
    # A diferencia del GMM, aquí el PCA opera sobre las trazas directamente,
    # no sobre un vector de características extraídas manualmente.

    pca = PCA(n_components=2)
    X_all_2d = pca.fit_transform(np.vstack([X_light_val_if_norm, X_dark_if_norm]))
    varianza = pca.explained_variance_ratio_

    # Score de anomalía como color continuo para los dark
    scores_dark_color = -scores_dark  # más alto = más anómalo

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel izquierdo: coloreado por clase
    axes[0].scatter(X_all_2d[:len(X_light_val_if_norm), 0],
                    X_all_2d[:len(X_light_val_if_norm), 1],
                    c='orange', alpha=0.5, s=12, label='Light', zorder=3)
    axes[0].scatter(X_all_2d[len(X_light_val_if_norm):, 0],
                    X_all_2d[len(X_light_val_if_norm):, 1],
                    c='black', alpha=0.5, s=12, label='Dark', zorder=3)
    axes[0].set_xlabel(f'PC1 ({varianza[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({varianza[1]*100:.1f}%)')
    axes[0].set_title('PCA — Coloreado por clase')
    axes[0].legend()
    axes[0].grid(True, linewidth=0.4)

    # Panel derecho: coloreado por score de anomalía (solo dark)
    sc = axes[1].scatter(
        X_all_2d[len(X_light_val_if_norm):, 0],
        X_all_2d[len(X_light_val_if_norm):, 1],
        c=scores_dark_color, cmap='RdYlGn_r',
        alpha=0.7, s=15, zorder=3
    )
    axes[1].scatter(X_all_2d[:len(X_light_val_if_norm), 0],
                    X_all_2d[:len(X_light_val_if_norm), 1],
                    c='orange', alpha=0.3, s=8, label='Light', zorder=2)
    plt.colorbar(sc, ax=axes[1], label='Score de anomalía')
    axes[1].set_xlabel(f'PC1 ({varianza[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({varianza[1]*100:.1f}%)')
    axes[1].set_title('PCA — Score de anomalía (dark)')
    axes[1].legend()
    axes[1].grid(True, linewidth=0.4)

    plt.suptitle('Isolation Forest — Espacio de trazas (PCA 2D)', y=1.01)
    plt.tight_layout()
    #guardar_figura("isolation_forest_pca_espacio_trazas")
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

def fpr_en_tpr(fpr, tpr, tpr_objetivo):
    """
    OBJ: Interpola la curva ROC para obtener la tasa de falsos positivos
         (FPR) correspondiente a una tasa de verdaderos positivos (TPR)
         objetivo fija (punto de operación).
    PARAMS:
        fpr, tpr : (np.ndarray) Puntos de la curva ROC (salida de roc_curve).
        tpr_objetivo : (float) TPR deseado (p.ej. 0.95).
    RETURNS:
        (float) FPR interpolado en ese punto de operación.
    """
    return float(np.interp(tpr_objetivo, tpr, fpr))


def tabla_operacion_modelos(modelos, tprs_objetivo=(0.95, 0.99)):
    """
    OBJ: Complementa el AUC (métrica agregada) con la FPR a puntos de
         operación fijos (TPR=0.95 y TPR=0.99 por defecto): dado que se
         exige detectar ese porcentaje de pulsos DARK, qué fracción de
         pulsos LIGHT se rechaza por error.
    PARAMS:
        modelos : (list) de tuplas (nombre, fpr, tpr, auc), una por modelo.
        tprs_objetivo : (tuple) TPR fijos a evaluar.
    RETURNS:
        filas : (list) de dicts con los valores de la tabla (auc y FPR a
                 cada TPR objetivo), por si se quieren reutilizar luego
                 en la memoria.
    """
    print("\n" + "=" * 72)
    print(" TABLA DE PUNTOS DE OPERACIÓN (FPR a TPR fijo)")
    print("=" * 72)

    cabecera = f"{'Modelo':<28}{'AUC':>8}" + "".join(
        f"{'FPR@TPR=' + str(t):>16}" for t in tprs_objetivo
    )
    print(cabecera)
    print("-" * len(cabecera))

    filas = []
    for nombre, fpr, tpr, roc_auc in modelos:
        fila = {"modelo": nombre, "auc": roc_auc}
        linea = f"{nombre:<28}{roc_auc:>8.3f}"
        for t in tprs_objetivo:
            fpr_t = fpr_en_tpr(fpr, tpr, t)
            fila[f"fpr_tpr_{t}"] = fpr_t
            linea += f"{fpr_t:>16.3f}"
        print(linea)
        filas.append(fila)

    print("=" * 72)
    print("Lectura: FPR@TPR=0.95 = proporción de pulsos LIGHT rechazados por")
    print("error al exigir detectar el 95% de los pulsos DARK (análogo para")
    print("0.99). Cuanto más bajo, mejor.")

    return filas


def comparar_modelos(roc_auc_cnn, fpr_cnn, tpr_cnn,
                      errores_autoencoders,
                      roc_auc_gmm, fpr_gmm, tpr_gmm,
                      roc_auc_if, fpr_if, tpr_if,
                      roc_auc_reg, fpr_reg, tpr_reg,
                      nombre_gmm='GMM'):
    """
    OBJ: Visualizar las curvas ROC de todos los modelos entrenados, para
         comparar su capacidad de discriminación Light/Dark. Del
         autoencoder y del GMM (cada uno con varias variantes) solo se
         pinta la mejor variante (mayor AUC) de cada uno, para no saturar
         la gráfica final con curvas ya comparadas en sus propios
         apartados (Secciones de evaluación de Autoencoders y GMM).
    PARAMS:
        roc_auc_cnn, fpr_cnn, tpr_cnn : salida de evaluar_cnn().
        errores_autoencoders : (dict) salida de calcula_error_reconstruccion_autoencoders().
        roc_auc_gmm, fpr_gmm, tpr_gmm : salida de evaluar_gmm() (ya es la mejor variante).
        roc_auc_if, fpr_if, tpr_if : salida de entrenar_y_evaluar_isolation_forest().
        roc_auc_reg, fpr_reg, tpr_reg: salida de entrenar_CNN_regresion
        nombre_gmm : (str) nombre de la variante de GMM representada (p.ej. "GMM — Solo Dark").
    """
    print("\n[Comparación de modelos]")

    plt.figure(figsize=(8, 7))

    # Se recopila (nombre, fpr, tpr, auc) de cada curva dibujada, para
    # reutilizarlo después en la tabla de puntos de operación.
    modelos_roc = []

    # CNN binaria
    plt.plot(fpr_cnn, tpr_cnn, color='steelblue', linewidth=2,
              label=f'CNN Binaria (AUC={roc_auc_cnn:.3f})')
    modelos_roc.append(('CNN Binaria', fpr_cnn, tpr_cnn, roc_auc_cnn))

    # Isolation Forest
    plt.plot(fpr_if, tpr_if, color='darkgreen', linewidth=2,
              label=f'Isolation Forest (AUC={roc_auc_if:.3f})')
    modelos_roc.append(('Isolation Forest', fpr_if, tpr_if, roc_auc_if))

    # GMM (mejor variante)
    plt.plot(fpr_gmm, tpr_gmm, color='purple', linewidth=2,
              label=f'{nombre_gmm} (AUC={roc_auc_gmm:.3f})')
    modelos_roc.append((nombre_gmm, fpr_gmm, tpr_gmm, roc_auc_gmm))

    #CNN Regresion
    plt.plot(fpr_reg, tpr_reg, color='crimson', linewidth=2,
          label=f'CNN Regresión (AUC={roc_auc_reg:.3f})')
    modelos_roc.append(('CNN Regresión', fpr_reg, tpr_reg, roc_auc_reg))

    # Autoencoder: se calculan las 3 variantes (mismo criterio de "clase
    # anómala por variante" que calcular_roc_autoencoders()) pero solo se
    # pinta la mejor (mayor AUC); la comparación de las 3 ya se muestra en
    # su propio apartado (comparar_roc_autoencoders(), opción 6 del menú).
    roc_ae_variantes = calcular_roc_autoencoders(errores_autoencoders)
    mejor_variante_ae = max(roc_ae_variantes, key=lambda k: roc_ae_variantes[k]["auc"])
    r_ae = roc_ae_variantes[mejor_variante_ae]

    plt.plot(r_ae["fpr"], r_ae["tpr"], color='orange', linewidth=2,
              label=f'{r_ae["titulo"]} (AUC={r_ae["auc"]:.3f})')
    modelos_roc.append((r_ae["titulo"], r_ae["fpr"], r_ae["tpr"], r_ae["auc"]))

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Clasificador aleatorio')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curvas ROC — Comparación de todos los modelos')
    plt.legend(loc='lower right')
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    #guardar_figura("roc_comparacion_completa")
    plt.show()

    tabla_operacion_modelos(modelos_roc)

    print("Comparación completada.")

# ==================================================
#------------ BLOQUE CNN DE REGRESION --------------
# ==================================================
def build_cnn_regresion(input_shape, num_conv_layers, num_filters,
                        kernel_size, dense_initial_units,
                        num_dense_layers, dropout_rate, learning_rate):
    """
    OBJ: Construye la arquitectura de la CNN de regresión.
        Misma estructura convolucional que la CNN binaria; cambia
        la salida (sigmoid) y función de pérdida (MAE).
    PARAMS:
        input_shape : (tuple) Forma de entrada, p.ej. (1200, 1).
        num_conv_layers, num_filters, kernel_size : (int) Hiperparámetros conv.
        dense_initial_units, num_dense_layers, dropout_rate : (int/float) Hiperparámetros densos.
        learning_rate : (float) Tasa de aprendizaje del optimizador Adam.
    RETURNS:
        model : (tf.keras.Model) Modelo compilado, sin entrenar.
    """
    model = models.Sequential()
    initializer = initializers.GlorotUniform()

    for i in range(num_conv_layers):
        if i == 0:
            model.add(layers.Conv1D(
                filters=num_filters, kernel_size=kernel_size,
                activation='tanh', kernel_initializer=initializer,
                input_shape=input_shape
            ))
        else:
            model.add(layers.Conv1D(
                filters=num_filters, kernel_size=kernel_size,
                activation='tanh', kernel_initializer=initializer
            ))
        model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Flatten())

    units = dense_initial_units
    for _ in range(num_dense_layers):
        model.add(layers.Dense(units, activation='relu', kernel_initializer=initializer))
        model.add(layers.Dropout(dropout_rate))
        units = units // 2

    # Salida de regresión: 1 neurona, activación lineal
    model.add(layers.Dense(1, activation='linear', name='output_regresion'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mae']
    )

    return model

def preparar_datos_cnn_regresion(X_light, X_dark, test_size_val=0.2, random_state=42):
    """
    OBJ: Calcular el target (V_min), normalizar trazas y target (con
         parámetros calculados solo a partir de light). Split
         train/val (solo light, mismo criterio que el autoencoder).
    PARAMS:
        X_light, X_dark : (np.ndarray) Pulsos crudos (N, 1200), en V, offset eliminado.
        test_size_val : (float) Porcentaje de validación.
        random_state : (int) Semilla.
    RETURNS:
        X_l_train, X_l_val, y_l_train, y_l_val : conjuntos de entrenamiento/validación (solo light).
        X_light_norm, X_dark_norm : (np.ndarray) trazas normalizadas con canal, (N, 1200, 1).
        y_light_norm, y_dark_norm : (np.ndarray) targets normalizados, (N,).
        mean_reg, std_reg, mean_y, std_y : (float) parámetros de normalización (para desnormalizar luego si hace falta).
    """
    # Target: V_min de cada traza (proporcional a la energía)
    y_light_reg = np.min(X_light, axis=1)
    y_dark_reg  = np.min(X_dark,  axis=1)

    print(f"Rango V_min light: {y_light_reg.min():.4f} a {y_light_reg.max():.4f} V")
    print(f"Rango V_min dark:  {y_dark_reg.min():.4f} a {y_dark_reg.max():.4f} V")

    plt.figure(figsize=(8, 4))
    plt.hist(y_light_reg * 1000, bins=40, alpha=0.6, color='orange', density=True, label='Light')
    plt.hist(y_dark_reg  * 1000, bins=40, alpha=0.6, color='black',  density=True, label='Dark')
    plt.xlabel('V_min (mV)')
    plt.ylabel('Densidad')
    plt.title('Distribución del target de regresión por clase')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

    #Normalización de las trazas (entrada), con parámetros de light
    mean_reg = np.mean(X_light)
    std_reg  = np.std(X_light)

    X_light_norm = ((X_light - mean_reg) / std_reg)[..., np.newaxis]
    X_dark_norm  = ((X_dark  - mean_reg) / std_reg)[..., np.newaxis]

    #Normalización del target (salida), con parámetros de light
    mean_y = np.mean(y_light_reg)
    std_y  = np.std(y_light_reg)

    y_light_norm = (y_light_reg - mean_y) / std_y
    y_dark_norm  = (y_dark_reg  - mean_y) / std_y

    # Split train/val solo con light
    X_l_train, X_l_val, y_l_train, y_l_val = train_test_split(X_light_norm, y_light_norm,
        test_size=test_size_val, random_state=random_state)

    print(f"Train: {X_l_train.shape} | Val: {X_l_val.shape}")

    return (X_l_train, X_l_val, y_l_train, y_l_val,
            X_light_norm, X_dark_norm, y_light_norm, y_dark_norm,
            mean_reg, std_reg, mean_y, std_y)

def entrenar_cnn_regresion(X_light, X_dark):
    """
    OBJ: entrenar la CNN de regresión.
    PARAMS:
        X_light, X_dark : (np.ndarray) Pulsos crudos de entrenamiento (N, 1200).
    RETURNS:
        resultados : (dict) modelo, history y todo lo necesario para evaluar después.
    """
    print("\n[Entrenamiento CNN Regresión]")

    (X_l_train, X_l_val, y_l_train, y_l_val,
     X_light_norm, X_dark_norm, y_light_norm, y_dark_norm,
     mean_reg, std_reg, mean_y, std_y) = preparar_datos_cnn_regresion(X_light, X_dark)

    cnn_reg = build_cnn_regresion(
        input_shape         = X_l_train.shape[1:],
        num_conv_layers     = 6,
        num_filters         = 45,
        kernel_size         = 5,
        dense_initial_units = 188,
        num_dense_layers    = 3,
        dropout_rate        = 0.18,
        learning_rate       = 5.2e-4
    )
    cnn_reg.summary()

    history_reg = cnn_reg.fit(
        X_l_train, y_l_train,
        validation_data=(X_l_val, y_l_val),
        epochs=20,
        batch_size=99,
        shuffle=True
    )

    print("Entrenamiento de la CNN de regresión completado.")

    return {
        "modelo": cnn_reg,
        "history": history_reg,
        "X_light_norm": X_light_norm,
        "X_dark_norm": X_dark_norm,
        "y_light_norm": y_light_norm,
        "y_dark_norm": y_dark_norm,
        "mean_y": mean_y,
        "std_y": std_y
    }

# EVALUACIÓN
def evaluar_cnn_regresion(resultados_reg):
    """
    OBJ: Evaluar la CNN de regresión: curva de aprendizaje, distribución
         del error de predicción por clase, scatter V_min real vs predicho
         y curva ROC (usando el error de predicción como score de anomalía).
    PARAMS:
        resultados_reg : (dict) salida de entrenar_cnn_regresion().
    RETURNS:
        err_reg_light, err_reg_dark : (np.ndarray) errores de predicción por clase.
        roc_auc_reg : (float) AUC de la curva ROC.
        fpr_reg, tpr_reg : (np.ndarray) puntos de la curva ROC.
    """
    print("\n[Evaluación CNN Regresión]")

    cnn_reg      = resultados_reg["modelo"]
    history_reg  = resultados_reg["history"]
    X_light_norm = resultados_reg["X_light_norm"]
    X_dark_norm  = resultados_reg["X_dark_norm"]
    y_light_norm = resultados_reg["y_light_norm"]
    y_dark_norm  = resultados_reg["y_dark_norm"]

    # Curva de aprendizaje
    plt.figure(figsize=(7, 4))
    plt.plot(history_reg.history['loss'],     color='steelblue', label='Train MAE')
    plt.plot(history_reg.history['val_loss'], color='steelblue', linestyle='--', label='Val MAE')
    plt.xlabel('Epoca')
    plt.ylabel('MAE')
    plt.title('Curva de aprendizaje — CNN Regresión')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    #guardar_figura("cnn_regresion_curva_aprendizaje")
    plt.show()

    # Error de predicción por pulso (score de anomalía)
    y_pred_light = cnn_reg.predict(X_light_norm, verbose=0).flatten()
    y_pred_dark  = cnn_reg.predict(X_dark_norm,  verbose=0).flatten()

    err_reg_light = np.abs(y_light_norm - y_pred_light)
    err_reg_dark  = np.abs(y_dark_norm  - y_pred_dark)

    # Distribución del error
    plt.figure(figsize=(8, 4))
    plt.hist(err_reg_light, bins=40, alpha=0.6, color='orange', density=True, label='Light')
    plt.hist(err_reg_dark,  bins=40, alpha=0.6, color='black',  density=True, label='Dark')
    plt.xlabel('Error de predicción de V_min (unidades normalizadas)')
    plt.ylabel('Densidad')
    plt.title('Distribución del error de predicción — CNN Regresión')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    #guardar_figura("cnn_regresion_distribucion_error")
    plt.show()

    # Scatter V_min real vs predicho
    plt.figure(figsize=(7, 6))
    plt.scatter(y_light_norm, y_pred_light, c='orange', alpha=0.4, s=10, label='Light')
    plt.scatter(y_dark_norm,  y_pred_dark,  c='black',  alpha=0.4, s=10, label='Dark')
    plt.plot([-3, 3], [-3, 3], 'r--', linewidth=1, label='Predicción perfecta')
    plt.xlabel('V_min real (normalizado)')
    plt.ylabel('V_min predicho (normalizado)')
    plt.title('V_min real vs predicho — CNN Regresión')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    #guardar_figura("cnn_regresion_scatter_real_vs_predicho")
    plt.show()

    # Curva ROC: error de predicción como score de anomalía
    scores_all = np.concatenate([err_reg_light, err_reg_dark])
    labels_all = np.concatenate([
        np.zeros(len(err_reg_light)),  # light = normal = 0
        np.ones(len(err_reg_dark))     # dark  = anomalía = 1
    ])

    fpr_reg, tpr_reg, _ = roc_curve(labels_all, scores_all)
    roc_auc_reg = auc(fpr_reg, tpr_reg)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_reg, tpr_reg, color='crimson', linewidth=2,
              label=f'CNN Regresión (AUC={roc_auc_reg:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC — CNN Regresión')
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    #guardar_figura("cnn_regresion_roc")
    plt.show()

    print(f"CNN Regresión AUC: {roc_auc_reg:.4f}")

    return err_reg_light, err_reg_dark, roc_auc_reg, fpr_reg, tpr_reg


# ===========================================================
# ----------------------- BLOQUE MAIN -----------------------
# ===========================================================
main()

"""División del conjunto de datos

Incialmente trabajaremos con los 4722 pulsos. Sin embargo, para reducir la carga computacional durante el entrenamiento
del modelo, se ha decidido seleccionar una ventana de muestras significativas. Para ello se ha calculado el pulso medio
del conjunto de datos light y dark. Se observa que la media en ambos casos se centra en el margen de muestra de 1000 a 
2000 muestras. Ajustando un poco más, se ha tomado como inicio 1300 como inicio. Se observa que los pulsos dark tienen
mayor dispersión, para que el modelo tenga más información de los pulsos dark, se ha decidido, tomar como limite final, 2500.
Además el usuario puede seleccionar con cuantas muestras quiere entrenar el modelo.

"""

"""## ----------- BLOQUE AUTOENCODERS ------------"""

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

"""## Modelo Mixto Gaussiano (GMM)

El GMM modela una distribución de probabilidad mediante combinación de gaussianas. La calidad de ese ajuste depende de la dimensionalidad de  
los pulsos del conjunto de entrenamiento. En este caso 1200.

Con 1200 dimensiones (muestras crudas), cada gaussiana necesita estimar una matriz de covarianza de 1200x1200 ≈ 720.000 parámetros. Con solo 1000 pulsos de entrenamiento, esto es un caso de "maldición de la dimensionalidad".

**Solución: extraer las características más relevantes.**

"""

"""### CNN de regresión

La otra alternativa que se sugiere en el artículo, es usar la CNN como modelo de regresión. En este caso, en lugar de calsificar el pulso como 
Light o Dark (salida binaria, 1 ó 0). El modelo tratará de predecir la engergía que tiene un pulso de Light (salida continua, un número). 
La detección de fondo se hace indirectamente, si el modelo predice bien la energía de un pulso de luz es porque se parece a la señal. Si la predicción
es mala, es porque la morfología del pulso es diferente, Fondo.
Para la regresión debemos elegir que característica es la que se tratará de rpedecir (el target). El artículo menciona directamente el Vmin y el FFT.

Sin embargo, como en el pipeline no se ha implementado el ajuste en frecuencia, la alternativa directa es usar el mínimo de voltaje de la traza (V_min), que 
es el valor del pico del pulso en dominio temporal y es proporcional a la energía absorbida.
"""
