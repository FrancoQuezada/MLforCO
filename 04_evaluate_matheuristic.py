# -*- coding: utf-8 -*-
"""
Fase 4: Evaluación de la Mateurística (Gurobi + ML)
Evalúa el desempeño de la matheurística comparando modelos ML contra Ground Truth.
"""

import sys
import os
import pandas as pd
import pickle

# Importar desde functions.py local
from functions import optimal_predict_threshUP, predicting_values_new
from config import TEST_PARAMS, TRAIN_PARAMS, ML_CONFIG, PATHS, EVALUATION_CONFIG

def load_test_data(params):
    """
    Carga datos de test con variables óptimas y métricas generados en Fase 2.
    
    Args:
        params: Diccionario con parámetros TEST_PARAMS
    
    Returns:
        df_variables: DataFrame con variables óptimas
        df_metrics: DataFrame con métricas óptimas
    """
    # Generar nombre de archivo
    vars_filename = f"optimal_variables_test_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    metrics_filename = f"optimal_metrics_test_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    
    vars_filepath = os.path.join(PATHS['data_optimal'], vars_filename)
    metrics_filepath = os.path.join(PATHS['data_optimal'], metrics_filename)
    
    # Cargar datos
    df_variables = pd.read_excel(vars_filepath)
    df_metrics = pd.read_excel(metrics_filepath)
    df_metrics.set_index('instance_id', inplace=True)
    
    return df_variables, df_metrics

def load_model_and_scaler(model_name, feature_cols, params, combo_name=None):
    """
    Carga modelo y scaler entrenados en Fase 3.
    
    Args:
        model_name: Nombre del modelo
        feature_cols: Lista de columnas de features
        params: Diccionario con parámetros TRAIN_PARAMS
        combo_name: Nombre de la combinación de features (ej: 'A', 'B', 'A+B')
    
    Returns:
        model: Modelo entrenado
        scaler: Scaler entrenado
    """
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
    
    # Cargar archivos
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_filepath, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict_and_fix(df_variables, model, scaler, feature_cols, threshold):
    """
    Aplica modelo para predecir variables binarias y fijarlas en el DataFrame.
    
    Args:
        df_variables: DataFrame con variables óptimas
        model: Modelo entrenado
        scaler: Scaler entrenado
        feature_cols: Lista de columnas de features
        threshold: Threshold para clasificación binaria
    
    Returns:
        df_with_predictions: DataFrame con predicciones añadidas
    """
    # Copiar DataFrame
    df = df_variables.copy()
    
    # Extraer features
    X = df[feature_cols].copy()
    
    # Aplicar scaler
    X[feature_cols] = scaler.transform(X[feature_cols])
    
    # Predecir probabilidades
    df['proba'] = model.predict_proba(X)[:, 1]
    
    # Clasificar según threshold
    df['y_hat'] = (df['proba'] >= threshold).astype(int)
    
    # Usar predicting_values_new para inyectar predicciones
    df = predicting_values_new(X, df, model, len(df['Item'].unique()))
    
    return df

def evaluate_instance(df_variables, df_metrics, instance_id, params, threshold, zeroOrOne):
    """
    Evalúa una instancia con la matheurística.
    
    Args:
        df_variables: DataFrame con variables óptimas
        df_metrics: DataFrame con métricas óptimas
        instance_id: ID de la instancia
        params: Diccionario con parámetros TEST_PARAMS
        threshold: Threshold para clasificación
        zeroOrOne: Estrategia de fijación [0], [1], o [0,1]
    
    Returns:
        results: Diccionario con métricas de evaluación
    """
    # Obtener parámetros
    c = params['c']
    f = params['f']
    semilla = params['semilla']
    tl = EVALUATION_CONFIG['time_limit']
    
    # Obtener métricas óptimas (Ground Truth)
    obj_optimal = df_metrics.loc[instance_id, 'obj_optimal']
    obj_optimal_pre = df_metrics.loc[instance_id, 'obj_optimal_pre']
    gap_optimal = df_metrics.loc[instance_id, 'gap_optimal'] / 100
    time_optimal = df_metrics.loc[instance_id, 'time_optimal']
    work_optimal = df_metrics.loc[instance_id, 'work_optimal']
    nodes_opt = df_metrics.loc[instance_id, 'nodes']
    clb_work_opt = df_metrics.loc[instance_id, 'clb_work']
    clb_obj_opt = df_metrics.loc[instance_id, 'clb_obj']
    
    # Crear DataFrame con formato esperado por optimal_predict_threshUP
    df_num_periodos_items = pd.DataFrame({
        'Cantidad_Periodos': [params['max_periods']],
        'Numero_items': [params['num_items']],
        'obj optimal': [obj_optimal],
        'obj optimal pre': [obj_optimal_pre],
        'GAP optimal': [gap_optimal * 100],
        'time optimal': [time_optimal],
        'work optimal': [work_optimal],
        'Nodes': [nodes_opt],
        'CLB work': [clb_work_opt],
        'CLB obj': [clb_obj_opt]
    }, index=[instance_id])
    
    # Resolver con matheurística
    x_hat_values, s_hat_values, y_hat_values_pre, y_hat_values, obj_hat_value_pre, obj_hat_value, prediction_time, restricted_milp_time, total_time, total_work, gap, pred_level, num_fixed_0, num_fixed_1, iterations, node_count, clb_time, clb_obj, feasible_pred, repair_time = optimal_predict_threshUP(
        tl, df_variables, df_num_periodos_items, instance_id, threshold, zeroOrOne
    )
    
    # Consolidar resultados
    results = {
        'instance_id': instance_id,
        'obj_optimal': obj_optimal,
        'obj_hat': obj_hat_value,
        'gap_optimal': gap_optimal * 100,
        'gap_hat': gap * 100,
        'time_optimal': time_optimal,
        'time_hat': total_time,
        'prediction_time': prediction_time,
        'optimization_time': restricted_milp_time,
        'repair_time': repair_time,
        'work_optimal': work_optimal,
        'work_hat': total_work,
        'nodes_optimal': nodes_opt,
        'nodes_hat': node_count,
        'clb_work_optimal': clb_work_opt,
        'clb_work_hat': clb_time,
        'clb_obj_optimal': clb_obj_opt,
        'clb_obj_hat': clb_obj,
        'num_fixed_0': num_fixed_0,
        'num_fixed_1': num_fixed_1,
        'iterations': iterations,
        'feasible': feasible_pred
    }
    
    return results

def evaluate_all_models(df_variables, df_metrics, params):
    """
    Evalúa todos los modelos con todas las combinaciones de features en todas las instancias de test.
    
    Args:
        df_variables: DataFrame con variables óptimas
        df_metrics: DataFrame con métricas óptimas
        params: Diccionario con parámetros TEST_PARAMS
    
    Returns:
        all_results: Lista de diccionarios con resultados
    """
    # Obtener lista de instancias
    if 'instancias_pasan_prueba' in params:
        instance_ids = params['instancias_pasan_prueba']
    else:
        instance_ids = df_metrics.index.tolist()
    
    threshold = EVALUATION_CONFIG['prediction_threshold']
    zeroOrOne = EVALUATION_CONFIG['zeroOrOne']
    
    # Lista para acumular resultados
    all_results = []
    
    # Iterar sobre cada combinación de features (solo A, B, C para evaluación)
    for combo_tuple, combo_features in ML_CONFIG['evaluation_combinations']:
        combo_name = '_'.join(combo_tuple)
        print(f"\n{'='*60}")
        print(f"Combinación de features: {combo_name}")
        print(f"Features: {combo_features}")
        print(f"{'='*60}")
        
        # Iterar sobre cada modelo
        for model_name in ML_CONFIG['models']:
            print(f"\nEvaluando modelo: {model_name} con combinación: {combo_name}")
            
            # Cargar modelo y scaler con nombre de combinación
            model, scaler = load_model_and_scaler(model_name, combo_features, TRAIN_PARAMS, combo_name)
            
            # Iterar sobre cada instancia
            for instance_id in instance_ids:
                print(f"--- Instancia {instance_id} ---")
                
                # Filtrar datos de la instancia
                df_instance = df_variables[df_variables['Instancia'] == instance_id].copy()
                
                # Predecir y fijar variables
                df_with_predictions = predict_and_fix(df_instance, model, scaler, combo_features, threshold)
                
                # Evaluar instancia
                results = evaluate_instance(df_with_predictions, df_metrics, instance_id, params, threshold, zeroOrOne)
                
                # Agregar nombre del modelo y combinación
                results['model'] = model_name
                results['combo_name'] = combo_name
                results['features'] = str(combo_features)
                
                # Acumular resultados
                all_results.append(results)
                
                print(f"  Obj Optimal: {results['obj_optimal']:.2f}")
                print(f"  Obj Hat: {results['obj_hat']:.2f}")
                print(f"  Gap Hat: {results['gap_hat']:.2f}%")
                print(f"  Time Hat: {results['time_hat']:.2f}s")
    
    return all_results

def save_evaluation_results(all_results, params):
    """
    Guarda resultados de evaluación en archivo Excel.
    
    Args:
        all_results: Lista de diccionarios con resultados
        params: Diccionario con parámetros TEST_PARAMS
    """
    # Crear carpeta si no existe
    os.makedirs(PATHS['results'], exist_ok=True)
    
    # Generar nombre de archivo
    results_filename = f"final_evaluation_metrics_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    results_filepath = os.path.join(PATHS['results'], results_filename)
    
    # Crear DataFrame y guardar
    results_df = pd.DataFrame(all_results)
    results_df.to_excel(results_filepath, index=False)
    
    print(f"\nResultados guardados en: {results_filepath}")

def main():
    """Evalúa la matheurística con los 5 modelos de ML."""
    print("=== Fase 4: Evaluación de la Mateurística (Gurobi + ML) ===")
    
    # Cargar datos de test
    print("\nCargando datos de test...")
    df_variables, df_metrics = load_test_data(TEST_PARAMS)
    print(f"Variables cargadas: Shape {df_variables.shape}")
    print(f"Métricas cargadas: Shape {df_metrics.shape}")
    
    # Evaluar todos los modelos
    print("\nEvaluando modelos de ML...")
    all_results = evaluate_all_models(df_variables, df_metrics, TEST_PARAMS)
    
    # Guardar resultados
    save_evaluation_results(all_results, TEST_PARAMS)
    
    print("\n=== Fase 4 Completada ===")

if __name__ == '__main__':
    main()
