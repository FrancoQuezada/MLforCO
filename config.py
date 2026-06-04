# -*- coding: utf-8 -*-
"""
Configuración centralizada para Experiment 0 - Reproducibilidad y Clean Pipeline
"""

# =============================================================================
# 1. PARÁMETROS DE GENERACIÓN DE INSTANCIAS
# =============================================================================

TRAIN_PARAMS = {
    'min_periods': 60,
    'max_periods': 60,
    'num_items': 8,
    'num_instancias': 1000,
    'semilla': 1,
    'c': 10,  # Ratio de capacidad (entre 10 y 14, más pequeño = más difícil)
    'f': 1000,  # Multiplicador de costo setup
    'instancias_pasan_prueba': [1, 21, 28, 90, 112, 116, 124, 145, 161, 197, 203, 221],
    'tl': 900  # Límite de tiempo para solver (segundos)
}

TEST_PARAMS = {
    'min_periods': 60,
    'max_periods': 60,
    'num_items': 32,
    'num_instancias': 33,
    'semilla': 1,
    'c': 40,  # Ratio de capacidad
    'f': 1000,  # Multiplicador de costo setup
    'instancias_pasan_prueba': [2,3,4,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33],
    'tl': 150  # Límite de tiempo para solver (segundos)
}

# =============================================================================
# 2. RANGOS DE PARÁMETROS (hardcodeados en crear_base_datos)
# =============================================================================

COST_RANGES = {
    'costo_prod': [1, 200],      # random.randint(1, 200)
    'costo_inv': [1, 100],       # random.randint(1, 100)
    'demanda': [500, 1500],      # random.randint(500, 1500)
    'capacidad_factor': [0.8, 1.2],  # random.randint(round(0.8 * c * d_barra), round(1.2 * c * d_barra))
    'setup_factor': [0.9, 1.1]   # random.randint(round(0.9 * f * h_barra), round(1.1 * f * h_barra))
}

# =============================================================================
# 3. RUTAS DE SALIDA
# =============================================================================

# ID del experimento actual (cambiar para exp1, exp2, etc.)
EXPERIMENT_ID = 'exp0'

PATHS = {
    'data_train': f'results_{EXPERIMENT_ID}/data/train',
    'data_test': f'results_{EXPERIMENT_ID}/data/test',
    'data_optimal': f'results_{EXPERIMENT_ID}/data/optimal',
    'models': f'results_{EXPERIMENT_ID}/models',
    'scalers': f'results_{EXPERIMENT_ID}/scalers',
    'params': f'results_{EXPERIMENT_ID}/params',
    'results': f'results_{EXPERIMENT_ID}/results',
    'functions': 'functions.py'  # Ruta local para importar funciones
}

# =============================================================================
# 4. CONFIGURACIÓN DE ENTRENAMIENTO ML
# =============================================================================

ML_CONFIG = {
    'models': ['DT', 'RF', 'KNN', 'SVM', 'NB'],  # Todos los modelos 
    'N_iter': 50,  # Iteraciones completas de Grid Search
    'feature_sets': {
        'A': ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup'],
        'B': ['x_relaxed', 's_relaxed', 'y_relaxed'],
        'B_hat': ['x_relaxed_rootnode', 's_relaxed_rootnode', 'y_relaxed_rootnode']
    },
    # Para entrenamiento (Tablas 4-5): todas las combinaciones
    'feature_combinations': [
        (('A',), ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup']),
        (('B',), ['x_relaxed', 's_relaxed', 'y_relaxed']),
        (('B_hat',), ['x_relaxed_rootnode', 's_relaxed_rootnode', 'y_relaxed_rootnode']),
        (('A', 'B'), ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'x_relaxed', 's_relaxed', 'y_relaxed']),
        (('A', 'B_hat'), ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'x_relaxed_rootnode', 's_relaxed_rootnode', 'y_relaxed_rootnode'])
    ],
    # Para evaluación matheuristic (Tablas 6+): solo A, B, C
    'evaluation_combinations': [
        (('A',), ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup']),
        (('B',), ['x_relaxed', 's_relaxed', 'y_relaxed']),
        (('B_hat',), ['x_relaxed_rootnode', 's_relaxed_rootnode', 'y_relaxed_rootnode'])
    ]
}

# =============================================================================
# 5. CONFIGURACIÓN DE EVALUACIÓN DE MATHEURÍSTICA
# =============================================================================

EVALUATION_CONFIG = {
    'prediction_threshold': 0.5,  # Threshold para clasificación binaria
    'zeroOrOne': [0, 1],  # Estrategia de fijación: fijar tanto ceros como unos
    'time_limit': 150  # Límite de tiempo para solver 
}
