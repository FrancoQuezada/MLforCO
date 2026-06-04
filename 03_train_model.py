# -*- coding: utf-8 -*-
"""
Fase 3: Entrenamiento de Modelos de Machine Learning
Entrena modelos de clasificación para predecir la variable binaria de producción.
Guarda modelos, scalers y métricas en archivos consolidados.
"""

import sys
import os
import pandas as pd
import time
import pickle

# Importar desde functions.py local
from functions import KNN, SVM_Classification, NB, DT, RF
from config import TRAIN_PARAMS, ML_CONFIG, PATHS

def load_training_data(params):
    """
    Carga datos de entrenamiento con variables óptimas generados en Fase 2.
    
    Args:
        params: Diccionario con parámetros TRAIN_PARAMS
    
    Returns:
        df: DataFrame con datos de entrenamiento
    """
    # Generar nombre de archivo
    filename = f"optimal_variables_train_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    filepath = os.path.join(PATHS['data_optimal'], filename)
    
    # Cargar datos
    df = pd.read_excel(filepath)
    
    # Crear columna 'binaria' desde y_pre (variable objetivo)
    df['binaria'] = df['y_pre'].round()
    
    return df

def prepare_features(df, feature_cols):
    """
    Prepara DataFrame con columnas de features y parámetros necesarios.
    
    Args:
        df: DataFrame con datos completos
        feature_cols: Lista de columnas de features a usar
    
    Returns:
        df_prepared: DataFrame preparado para entrenamiento
    """
    # Columnas de parámetros que siempre se incluyen
    cols_parameters = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
    
    # Columnas faltantes (parámetros no incluidos en features)
    columnas_faltantes = [col for col in cols_parameters if col not in feature_cols]
    
    # Crear DataFrame con columnas necesarias
    df_prepared = df[["Instancia", "Item", "Periodo", "binaria"] + columnas_faltantes + feature_cols].copy()
    
    return df_prepared

def train_model(df, model_name, feature_cols, n_iter):
    """
    Entrena un modelo específico con Grid Search.
    
    Args:
        df: DataFrame preparado para entrenamiento
        model_name: Nombre del modelo ('KNN', 'SVM', 'NB', 'DT', 'RF')
        feature_cols: Lista de columnas de features
        n_iter: Número de iteraciones de Grid Search
    
    Returns:
        model: Modelo entrenado
        scaler: Scaler entrenado
        metrics: Diccionario con métricas de entrenamiento
    """
    start_time = time.time()
    
    # Seleccionar función de entrenamiento según modelo
    if model_name == 'KNN':
        model, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, _, _, _, _, _, _, _, scaler = KNN(df, n_iter, feature_cols)
    elif model_name == 'SVM':
        model, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, _, _, _, _, _, _, _, scaler = SVM_Classification(df, n_iter, feature_cols)
    elif model_name == 'NB':
        model, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, _, _, _, _, _, _, _, scaler = NB(df, n_iter, feature_cols)
    elif model_name == 'DT':
        model, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, _, _, _, _, _, _, _, scaler = DT(df, n_iter, feature_cols)
    elif model_name == 'RF':
        model, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, _, _, _, _, _, _, _, scaler = RF(df, n_iter, feature_cols)
    
    end_time = time.time()
    
    # Consolidar métricas
    metrics = {
        'model': model_name,
        'features': str(feature_cols),
        'accuracy': final_score,
        'mean_score_gridsearch': best_score,
        'std_score_gridsearch': best_std,
        'training_time': end_time - start_time
    }
    
    return model, scaler, metrics

def save_model_and_scaler(model, scaler, model_name, feature_cols, params, combo_name=None):
    """
    Guarda modelo y scaler en archivos .pkl.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler entrenado
        model_name: Nombre del modelo
        feature_cols: Lista de columnas de features
        params: Diccionario con parámetros TRAIN_PARAMS
        combo_name: Nombre de la combinación de features (ej: 'A', 'B', 'A+B')
    """
    # Crear carpetas si no existen
    os.makedirs(PATHS['models'], exist_ok=True)
    os.makedirs(PATHS['scalers'], exist_ok=True)
    
    # Generar nombres de archivos
    feature_str = '_'.join(feature_cols)
    if combo_name:
        model_filename = f"model_{model_name}_combo_{combo_name}_features_{feature_str}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.pkl"
        scaler_filename = f"scaler_{model_name}_combo_{combo_name}_features_{feature_str}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.pkl"
    else:
        model_filename = f"model_{model_name}_features_{feature_str}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.pkl"
        scaler_filename = f"scaler_{model_name}_features_{feature_str}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.pkl"
    
    model_filepath = os.path.join(PATHS['models'], model_filename)
    scaler_filepath = os.path.join(PATHS['scalers'], scaler_filename)
    
    # Guardar archivos
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_filepath, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Modelo guardado en: {model_filepath}")
    print(f"Scaler guardado en: {scaler_filepath}")

def save_training_metrics(all_metrics, params, consolidated=True):
    """
    Guarda métricas de entrenamiento en archivo Excel.
    
    Args:
        all_metrics: Lista de diccionarios con métricas
        params: Diccionario con parámetros TRAIN_PARAMS
        consolidated: Si True, guarda archivo consolidado con todas las combinaciones
    """
    # Crear carpeta si no existe
    os.makedirs(PATHS['results'], exist_ok=True)
    
    # Crear DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    if consolidated:
        # Generar nombre de archivo consolidado
        metrics_filename = f"training_metrics_consolidated_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    else:
        # Generar nombre de archivo individual (comportamiento original)
        metrics_filename = f"training_metrics_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    
    metrics_filepath = os.path.join(PATHS['results'], metrics_filename)
    
    # Guardar
    metrics_df.to_excel(metrics_filepath, index=False)
    
    print(f"Métricas guardadas en: {metrics_filepath}")

def main():
    """Entrena modelos de ML con todas las combinaciones de features."""
    print("=== Fase 3: Entrenamiento de Modelos de Machine Learning ===")
    
    # Cargar datos de entrenamiento
    print("\nCargando datos de entrenamiento...")
    df = load_training_data(TRAIN_PARAMS)
    
    # Lista para acumular métricas de todas las combinaciones
    all_metrics = []
    
    # Iterar sobre todas las combinaciones de features
    for combo_tuple, combo_features in ML_CONFIG['feature_combinations']:
        combo_name = '_'.join(combo_tuple)
        print(f"\n{'='*60}")
        print(f"Combinación de features: {combo_name}")
        print(f"Features: {combo_features}")
        print(f"{'='*60}")
        
        # Preparar DataFrame con esta combinación
        df_prepared = prepare_features(df, combo_features)
        
        # Entrenar cada modelo con esta combinación
        for model_name in ML_CONFIG['models']:
            print(f"\nEntrenando modelo: {model_name} con combinación: {combo_name}")
            
            # Entrenar modelo
            model, scaler, metrics = train_model(df_prepared, model_name, combo_features, ML_CONFIG['N_iter'])
            
            # Agregar nombre de combinación a métricas
            metrics['combo_name'] = combo_name
            metrics['combo_tuple'] = str(combo_tuple)
            
            # Guardar modelo y scaler con nombre de combinación
            save_model_and_scaler(model, scaler, model_name, combo_features, TRAIN_PARAMS, combo_name)
            
            # Acumular métricas
            all_metrics.append(metrics)
            
            print(f"Métricas para {model_name} con {combo_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Mean CV Score: {metrics['mean_score_gridsearch']:.4f}")
            print(f"  Std CV Score: {metrics['std_score_gridsearch']:.4f}")
            print(f"  Training Time: {metrics['training_time']:.2f}s")
    
    # Guardar métricas consolidadas (todas las combinaciones en un archivo)
    save_training_metrics(all_metrics, TRAIN_PARAMS, consolidated=True)
    
    print("\n=== Fase 3 Completada ===")
    print(f"Total entrenamientos: {len(all_metrics)} ({len(ML_CONFIG['models'])} modelos × {len(ML_CONFIG['feature_combinations'])} combinaciones)")

if __name__ == '__main__':
    main()
