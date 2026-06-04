# -*- coding: utf-8 -*-
"""
Fase 1: Generación de Instancias (TRAIN y TEST)
Genera bases de datos de instancias con identificadores únicos según configuración.
"""

import sys
import os
import pandas as pd

# Importar desde functions.py local
from functions import crear_base_datos
from config import TRAIN_PARAMS, TEST_PARAMS, PATHS

def generate_instances(params, instance_type):
    """
    Genera instancias según parámetros especificados.
    
    Args:
        params: Diccionario con parámetros (TRAIN_PARAMS o TEST_PARAMS)
        instance_type: 'train' o 'test'
    
    Returns:
        base_datos_completa: DataFrame con todas las instancias
        df_num_periodos_items: DataFrame con metadatos por instancia
    """
    # Llamar a crear_base_datos
    base_datos_completa, df_num_periodos_items = crear_base_datos(
        min_periodos=params['min_periods'],
        max_periodos=params['max_periods'],
        num_instancias=params['num_instancias'],
        num_items=params['num_items'],
        c=params['c'],
        f=params['f'],
        semilla=params['semilla']
    )
    
    # Filtrar instancias que pasan prueba (si está definido)
    if 'instancias_pasan_prueba' in params:
        base_datos_completa = base_datos_completa[
            base_datos_completa['Instancia'].isin(params['instancias_pasan_prueba'])
        ]
        df_num_periodos_items = df_num_periodos_items[
            df_num_periodos_items.index.isin(params['instancias_pasan_prueba'])
        ]
    
    return base_datos_completa, df_num_periodos_items

def save_instances(base_datos_completa, df_num_periodos_items, params, instance_type):
    """
    Guarda instancias con identificador único.
    
    Formato: instances_{type}_n{items}_p{periodos}_c{ratio}_f{setup}_s{semilla}.xlsx
    """
    # Crear carpeta de salida
    output_dir = PATHS[f'data_{instance_type}']
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar identificador único
    filename = f"instances_{instance_type}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    # Guardar Excel
    base_datos_completa.to_excel(filepath, index=False)
    
    # Guardar metadatos
    metadata_filename = f"metadata_{instance_type}_n{params['num_items']}_p{params['max_periods']}_c{params['c']}_f{params['f']}_s{params['semilla']}.xlsx"
    metadata_filepath = os.path.join(output_dir, metadata_filename)
    df_num_periodos_items.to_excel(metadata_filepath)
    
    print(f"Instancias {instance_type.upper()} guardadas en: {filepath}")
    print(f"Metadatos guardados en: {metadata_filepath}")

def main():
    """Genera instancias de TRAIN y TEST."""
    print("=== Fase 1: Generación de Instancias ===")
    
    # Generar TRAIN
    print("\nGenerando instancias de TRAIN...")
    train_data, train_metadata = generate_instances(TRAIN_PARAMS, 'train')
    save_instances(train_data, train_metadata, TRAIN_PARAMS, 'train')
    
    # Generar TEST
    print("\nGenerando instancias de TEST...")
    test_data, test_metadata = generate_instances(TEST_PARAMS, 'test')
    save_instances(test_data, test_metadata, TEST_PARAMS, 'test')
    
    print("\n=== Fase 1 Completada ===")

if __name__ == '__main__':
    main()
