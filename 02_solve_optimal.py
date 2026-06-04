# -*- coding: utf-8 -*-
"""
Fase 2: Cálculo del Óptimo (Ground Truth) y Generación de Relajaciones
Calcula soluciones óptimas y relajaciones para TRAIN y TEST.
Guarda variables y métricas en archivos Excel consolidados.
"""

import sys
import os
import time
import pandas as pd

# Importar desde functions.py local
from functions import optimal, optimal_relaxed_binary, optimal_root_node
from config import TRAIN_PARAMS, TEST_PARAMS, PATHS

def load_instances(params, instance_type):
    """
    Carga instancias y metadatos generados en Fase 1.
    
    Args:
        params: Diccionario con parámetros (TRAIN_PARAMS o TEST_PARAMS)
        instance_type: 'train' o 'test'
    
    Returns:
        base_datos: DataFrame con instancias
        metadata: DataFrame con metadatos
    """
    # Generar nombre de archivo
    filename = f"instances_{instance_type}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    filepath = os.path.join(PATHS[f'data_{instance_type}'], filename)
    
    # Cargar instancias
    base_datos = pd.read_excel(filepath)
    
    # Cargar metadatos
    metadata_filename = f"metadata_{instance_type}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    metadata_filepath = os.path.join(PATHS[f'data_{instance_type}'], metadata_filename)
    metadata = pd.read_excel(metadata_filepath)
    metadata.set_index('Instancia', inplace=True)
    
    return base_datos, metadata

def solve_instance(base_datos, metadata, instance_id, params):
    """
    Resuelve una instancia con optimal, optimal_relaxed_binary y optimal_root_node.
    
    Args:
        base_datos: DataFrame con instancias
        metadata: DataFrame con metadatos
        instance_id: ID de la instancia a resolver
        params: Diccionario con parámetros
    
    Returns:
        results: Diccionario con variables y métricas
    """
    # Obtener parámetros
    c = params['c']
    f = params['f']
    semilla = params['semilla']
    tl = params['tl']
    
    # Solving Optimal
    x_values, s_values, y_values_pre, y_values, obj_value_pre, obj_value, obj_gap, solve_time, total_work, _, node_count, clb_time, clb_obj = optimal(base_datos, metadata, instance_id, c, f, semilla, tl)
    
    # Solving Relaxation (feature generation B)
    relaxed_start = time.time()
    relaxed_x_values, relaxed_s_values, relaxed_y_values_pre, relaxed_obj_value_pre, relaxed_obj_value, relaxed_solve_time, _ = optimal_relaxed_binary(base_datos, metadata, instance_id, c, f, semilla)
    relaxed_end = time.time()
    
    # Solving Root Node Relaxation (feature generation B_hat)
    rootnode_start = time.time()
    rootnode_x_values, rootnode_s_values, rootnode_y_values_pre, rootnode_obj_value_pre, rootnode_obj_value, rootnode_solve_time, _ = optimal_root_node(base_datos, metadata, instance_id, c, f, semilla, tl)
    rootnode_end = time.time()
    
    # Feature generation time (tiempo de generar relajaciones)
    feature_generation_time = (relaxed_end - relaxed_start) + (rootnode_end - rootnode_start)
    
    # Consolidar resultados
    results = {
        'variables': {
            'x': x_values,
            's': s_values,
            'y_pre': y_values_pre,
            'y': y_values,
            'x_relaxed': relaxed_x_values,
            's_relaxed': relaxed_s_values,
            'y_relaxed': relaxed_y_values_pre,
            'x_relaxed_rootnode': rootnode_x_values,
            's_relaxed_rootnode': rootnode_s_values,
            'y_relaxed_rootnode': rootnode_y_values_pre
        },
        'metrics': {
            'instance_id': instance_id,
            'obj_optimal_pre': obj_value_pre,
            'obj_optimal': obj_value,
            'gap_optimal': obj_gap,
            'time_optimal': solve_time,
            'work_optimal': total_work,
            'nodes': node_count,
            'clb_work': clb_time,
            'clb_obj': clb_obj,
            'obj_relaxed_pre': relaxed_obj_value_pre,
            'obj_relaxed': relaxed_obj_value,
            'time_relaxed': relaxed_solve_time,
            'gap_relaxed': round((relaxed_obj_value - obj_value) * 100 / obj_value, 3),
            'obj_rootnode_pre': rootnode_obj_value_pre,
            'obj_rootnode': rootnode_obj_value,
            'time_rootnode': rootnode_solve_time,
            'gap_rootnode': round((rootnode_obj_value - obj_value) * 100 / obj_value, 3),
            'feature_generation_time': feature_generation_time
        }
    }
    
    return results

def solve_all_instances(base_datos, metadata, params, instance_type):
    """
    Resuelve todas las instancias de un tipo (TRAIN o TEST).
    
    Args:
        base_datos: DataFrame con instancias
        metadata: DataFrame con metadatos
        params: Diccionario con parámetros
        instance_type: 'train' o 'test'
    
    Returns:
        base_datos_with_vars: DataFrame con variables añadidas
        metrics_df: DataFrame con métricas
    """
    # Obtener lista de instancias a resolver
    if 'instancias_pasan_prueba' in params:
        instance_ids = params['instancias_pasan_prueba']
    else:
        instance_ids = metadata.index.tolist()
    
    # Listas para acumular resultados
    all_metrics = []
    
    # Diccionarios para variables
    vars_x = []
    vars_s = []
    vars_y_pre = []
    vars_y = []
    vars_x_relaxed = []
    vars_s_relaxed = []
    vars_y_relaxed = []
    vars_x_rootnode = []
    vars_s_rootnode = []
    vars_y_rootnode = []
    
    for instance_id in instance_ids:
        print(f"---------------- RESOLVIENDO INSTANCIA {instance_id} ----------------")
        
        # Resolver instancia
        results = solve_instance(base_datos, metadata, instance_id, params)
        
        # Acumular métricas
        all_metrics.append(results['metrics'])
        
        # Acumular variables
        for (t, j), value in results['variables']['x'].items():
            vars_x.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
        for (t, j), value in results['variables']['s'].items():
            vars_s.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
        for (t, j), value in results['variables']['y_pre'].items():
            vars_y_pre.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
        for (t, j), value in results['variables']['y'].items():
            vars_y.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
        for (t, j), value in results['variables']['x_relaxed'].items():
            vars_x_relaxed.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
        for (t, j), value in results['variables']['s_relaxed'].items():
            vars_s_relaxed.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
        for (t, j), value in results['variables']['y_relaxed'].items():
            vars_y_relaxed.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
        for (t, j), value in results['variables']['x_relaxed_rootnode'].items():
            vars_x_rootnode.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
        for (t, j), value in results['variables']['s_relaxed_rootnode'].items():
            vars_s_rootnode.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
        for (t, j), value in results['variables']['y_relaxed_rootnode'].items():
            vars_y_rootnode.append({'Instancia': instance_id, 'Periodo': t, 'Item': j, 'Valor': value})
    
    # Crear DataFrames de variables
    df_x = pd.DataFrame(vars_x)
    df_s = pd.DataFrame(vars_s)
    df_y_pre = pd.DataFrame(vars_y_pre)
    df_y = pd.DataFrame(vars_y)
    df_x_relaxed = pd.DataFrame(vars_x_relaxed)
    df_s_relaxed = pd.DataFrame(vars_s_relaxed)
    df_y_relaxed = pd.DataFrame(vars_y_relaxed)
    df_x_rootnode = pd.DataFrame(vars_x_rootnode)
    df_s_rootnode = pd.DataFrame(vars_s_rootnode)
    df_y_rootnode = pd.DataFrame(vars_y_rootnode)
    
    # Añadir variables al DataFrame base
    base_datos_with_vars = base_datos.copy()
    base_datos_with_vars['x'] = df_x['Valor']
    base_datos_with_vars['s'] = df_s['Valor']
    base_datos_with_vars['y_pre'] = df_y_pre['Valor']
    base_datos_with_vars['y'] = df_y['Valor']
    base_datos_with_vars['x_relaxed'] = df_x_relaxed['Valor']
    base_datos_with_vars['s_relaxed'] = df_s_relaxed['Valor']
    base_datos_with_vars['y_relaxed'] = df_y_relaxed['Valor']
    base_datos_with_vars['x_relaxed_rootnode'] = df_x_rootnode['Valor']
    base_datos_with_vars['s_relaxed_rootnode'] = df_s_rootnode['Valor']
    base_datos_with_vars['y_relaxed_rootnode'] = df_y_rootnode['Valor']
    base_datos_with_vars['mu'] = base_datos_with_vars['x'] / base_datos_with_vars['Capacidad']
    
    # Crear DataFrame de métricas
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.set_index('instance_id', inplace=True)
    
    return base_datos_with_vars, metrics_df

def save_results(base_datos_with_vars, metrics_df, params, instance_type):
    """
    Guarda variables y métricas en archivos Excel consolidados.
    
    Args:
        base_datos_with_vars: DataFrame con variables añadidas
        metrics_df: DataFrame con métricas
        params: Diccionario con parámetros
        instance_type: 'train' o 'test'
    """
    # Crear carpeta de salida
    output_dir = PATHS['data_optimal']
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar nombres de archivos
    vars_filename = f"optimal_variables_{instance_type}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    metrics_filename = f"optimal_metrics_{instance_type}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    
    vars_filepath = os.path.join(output_dir, vars_filename)
    metrics_filepath = os.path.join(output_dir, metrics_filename)
    
    # Guardar archivos
    base_datos_with_vars.to_excel(vars_filepath, index=False)
    metrics_df.to_excel(metrics_filepath)
    
    print(f"Variables {instance_type.upper()} guardadas en: {vars_filepath}")
    print(f"Métricas {instance_type.upper()} guardadas en: {metrics_filepath}")

def main():
    """Resuelve instancias de TRAIN y TEST."""
    print("=== Fase 2: Cálculo del Óptimo y Relajaciones ===")
    
    # Procesar TRAIN
    print("\nProcesando instancias de TRAIN...")
    train_data, train_metadata = load_instances(TRAIN_PARAMS, 'train')
    train_vars, train_metrics = solve_all_instances(train_data, train_metadata, TRAIN_PARAMS, 'train')
    save_results(train_vars, train_metrics, TRAIN_PARAMS, 'train')
    
    # Procesar TEST
    print("\nProcesando instancias de TEST...")
    test_data, test_metadata = load_instances(TEST_PARAMS, 'test')
    test_vars, test_metrics = solve_all_instances(test_data, test_metadata, TEST_PARAMS, 'test')
    save_results(test_vars, test_metrics, TEST_PARAMS, 'test')
    
    print("\n=== Fase 2 Completada ===")

if __name__ == '__main__':
    main()
