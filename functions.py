# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:40:39 2024

@author: jcorn
"""
# %% LIBRERÍAS
import os
import shutil

# Importar PATHS desde config para guardar params en la ubicación correcta
from config import PATHS

# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.utils import resample
from gurobipy import GRB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import gurobipy as gp
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report


# import pyearth

# %% FUNCIONES

################################## FUNCIONES ##################################
def Kbest(X, y, K):
    selector = SelectKBest(score_func=f_classif, k=K)  # Selecciona las K mejores características
    X_selected = selector.fit_transform(X, y)
    # Obtener las puntuaciones de las características
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_,
        'P-value': selector.pvalues_
    })

    feature_scores = feature_scores.sort_values(by='Score', ascending=False).reset_index()

    # Visualizar los resultados
    plt.figure(figsize=(15, 18))
    sb.barplot(x='Score', y='Feature', data=feature_scores, palette='viridis')
    plt.title('Feature Scores')
    plt.xlabel('Score')
    plt.ylabel('Feature')
    plt.show()

    print("Feature Scores:\n", feature_scores)

    top_features = [feature_scores.loc[i, 'Feature'] for i in range(K)]

    X = X[top_features]

    return X, top_features


def crear_base_datos(min_periodos, max_periodos, num_instancias, num_items, c, f, semilla):
    random.seed(semilla)

    instancias = []
    num_periodos_por_instancia = []
    num_items_por_instancia = []

    for instancia in range(1, num_instancias + 1):
        num_periodos = random.randint(min_periodos, max_periodos)

        datos = {
            'Instancia': [],
            'Periodo': [],
            'Item': [],
            'Costo_Produccion': [],
            'Costo_Inventario': [],
            'Demanda': [],
            'Costo_Setup': [],
            'Capacidad': []
        }

        # Listas para almacenar todas las demandas y costos de inventario
        demandas_totales = []
        costos_inventario_totales = []

        # Generar datos aleatorios para p, h, d y acumular para calcular promedios globales
        for t in range(1, num_periodos + 1):
            for j in range(1, num_items + 1):
                cp = random.randint(1, 200)
                ci = random.randint(1, 100)
                d = random.randint(500, 1500)

                datos['Instancia'].append(instancia)
                datos['Periodo'].append(t)
                datos['Item'].append(j)
                datos['Costo_Produccion'].append(cp)
                datos['Costo_Inventario'].append(ci)
                datos['Demanda'].append(d)

                demandas_totales.append(d)
                costos_inventario_totales.append(ci)

        # Calcular demanda promedio global y costo de inventario promedio global
        d_barra = sum(demandas_totales) / len(demandas_totales)
        h_barra = sum(costos_inventario_totales) / len(costos_inventario_totales)

        # Generar capacidad compartida por período y costos de setup basados en promedios globales
        for t in range(1, num_periodos + 1):
            # Generar capacidad compartida para el período t
            capacidad_t = random.randint(
                round(0.8 * c * d_barra),
                round(1.2 * c * d_barra)
            )

            # Añadir capacidad compartida para cada artículo en el período t
            for j in range(1, num_items + 1):
                datos['Capacidad'].append(capacidad_t)

                # Generar costo de setup basado en h_barra
                q = random.randint(
                    round(0.9 * f * h_barra),
                    round(1.1 * f * h_barra)
                )
                datos['Costo_Setup'].append(q)

        # Crear DataFrame para esta instancia y añadirlo a la lista
        df = pd.DataFrame(datos)
        instancias.append(df)

        # Guardar la cantidad de períodos y artículos para esta instancia
        num_periodos_por_instancia.append(num_periodos)
        num_items_por_instancia.append(num_items)

    # Combinar todas las instancias en un solo DataFrame
    base_datos_completa = pd.concat(instancias, ignore_index=True)

    # Crear DataFrame con la cantidad de períodos por instancia
    df_num_periodos_items = pd.DataFrame({
        'Instancia': range(1, num_instancias + 1),
        'Cantidad_Periodos': num_periodos_por_instancia,
        'Numero_items': num_items_por_instancia
    })
    df_num_periodos_items.set_index('Instancia', inplace=True)

    return base_datos_completa, df_num_periodos_items


def crear_base_datos1(min_periodos, max_periodos, num_instancias, num_items, c, f, semilla, costo_prod, costo_inv, dem):
    random.seed(semilla)

    # Lista para almacenar los DataFrames de cada instancia
    instancias = []
    num_periodos_por_instancia = []
    num_items_por_instancia = []
    for instancia in range(1, num_instancias + 1):
        # Generar cantidad aleatoria de períodos para esta instancia
        num_periodos = random.randint(min_periodos, max_periodos)

        # Generar datos aleatorios para p, h, q, d
        datos = {'Instancia': [], 'Periodo': [], 'Item': [], 'Costo_Produccion': [], 'Costo_Inventario': [],
                 'Demanda': [], 'Costo_Setup': [], 'Capacidad': []}
        for t in range(1, num_periodos + 1):
            for j in range(1, num_items + 1):
                datos['Instancia'].append(instancia)
                datos['Periodo'].append(t)
                datos['Item'].append(j)
                datos['Costo_Produccion'].append(random.randint(costo_prod[0], costo_prod[1]))
                datos['Costo_Inventario'].append(random.randint(costo_inv[0], costo_inv[1]))
                datos['Demanda'].append(random.randint(dem[0], dem[1]))
            h_hat = sum(datos['Costo_Inventario'][-num_items:]) / num_items
            for j in range(1, num_items + 1):
                datos['Costo_Setup'].append(random.randint(round(0.9 * f * h_hat), round(1.1 * f * h_hat)))

            d_hat = sum(datos['Demanda'][-num_items:]) / num_items
            # print(f'Demanda promedio = {d_hat}')
            datos['Capacidad'].extend([random.randint(round(0.8 * c * d_hat), round(1.2 * c * d_hat))] * num_items)

        # print(f'Instancia: {datos}')

        # Crear DataFrame para esta instancia y añadirlo a la lista
        df = pd.DataFrame(datos)
        instancias.append(df)

        # Guardar la cantidad de períodos para esta instancia
        num_periodos_por_instancia.append(num_periodos)
        num_items_por_instancia.append(num_items)

    # Combinar todas las instancias en un solo DataFrame
    base_datos_completa = pd.concat(instancias)

    # Crear DataFrame con la cantidad de períodos por instancia
    df_num_periodos_items = pd.DataFrame(
        {'Instancia': range(1, num_instancias + 1), 'Cantidad_Periodos': num_periodos_por_instancia,
         'Numero_items': num_items_por_instancia})
    df_num_periodos_items.set_index('Instancia', inplace=True)

    return base_datos_completa, df_num_periodos_items


def replace_instance(c, f, semilla, base_datos_completa, df_num_periodos_items, i):
    random.seed(semilla)
    num_periodos = df_num_periodos_items.loc[i, 'Cantidad_Periodos']
    num_items = df_num_periodos_items.loc[i, 'Numero_items']

    # Filtrar las filas correspondientes a la instancia i
    indices_instancia = base_datos_completa['Instancia'] == i

    # Listas para acumular demandas y costos de inventario
    demandas_totales = []
    costos_inventario_totales = []

    # Generar datos aleatorios para p, h, d y acumular para promedios globales
    for t in range(1, num_periodos + 1):
        for j in range(1, num_items + 1):
            idx = indices_instancia & (base_datos_completa['Periodo'] == t) & (base_datos_completa['Item'] == j)
            cp = random.randint(1, 200)
            ci = random.randint(1, 100)
            d = random.randint(500, 1500)

            base_datos_completa.loc[idx, 'Costo_Produccion'] = cp
            base_datos_completa.loc[idx, 'Costo_Inventario'] = ci
            base_datos_completa.loc[idx, 'Demanda'] = d

            demandas_totales.append(d)
            costos_inventario_totales.append(ci)

    # Calcular promedios globales
    h_barra = sum(costos_inventario_totales) / len(costos_inventario_totales)
    d_barra = sum(demandas_totales) / len(demandas_totales)

    # Generar costos de setup y capacidades usando promedios globales
    for t in range(1, num_periodos + 1):
        # Generar capacidad compartida para el período t
        capacidad_t = random.randint(
            round(0.8 * c * d_barra),
            round(1.2 * c * d_barra)
        )
        idx_periodo = indices_instancia & (base_datos_completa['Periodo'] == t)
        base_datos_completa.loc[idx_periodo, 'Capacidad'] = capacidad_t

        # Generar costos de setup para cada artículo en el período t
        for j in range(1, num_items + 1):
            idx = indices_instancia & (base_datos_completa['Periodo'] == t) & (base_datos_completa['Item'] == j)
            q = random.randint(
                round(0.9 * f * h_barra),
                round(1.1 * f * h_barra)
            )
            base_datos_completa.loc[idx, 'Costo_Setup'] = q

    return base_datos_completa, df_num_periodos_items


def replace_instance1(c, f, semilla, base_datos_completa, df_num_periodos_items, i, costo_prod, costo_inv, dem):
    random.seed(semilla)
    # Generar cantidad aleatoria de períodos para esta instancia
    num_periodos = df_num_periodos_items.loc[i, 'Cantidad_Periodos']
    num_items = df_num_periodos_items.loc[i, 'Numero_items']

    # Generar datos aleatorios para p, h, q, d
    for t in range(1, num_periodos + 1):
        for j in range(1, num_items + 1):
            base_datos_completa.loc[(base_datos_completa['Instancia'] == i) & (base_datos_completa['Periodo'] == t) & (
                    base_datos_completa['Item'] == j), 'Costo_Produccion'] = random.randint(costo_prod[0],
                                                                                            costo_prod[
                                                                                                1])  # costo producción
            base_datos_completa.loc[(base_datos_completa['Instancia'] == i) & (base_datos_completa['Periodo'] == t) & (
                    base_datos_completa['Item'] == j), 'Costo_Inventario'] = random.randint(costo_inv[0], costo_inv[
                1])  # costo inventario
            base_datos_completa.loc[(base_datos_completa['Instancia'] == i) & (base_datos_completa['Periodo'] == t) & (
                    base_datos_completa['Item'] == j), 'Demanda'] = random.randint(dem[0], dem[1])  # demanda

        h_hat = base_datos_completa.loc[
            (base_datos_completa['Instancia'] == i) & (base_datos_completa['Periodo'] == t), 'Costo_Inventario'].mean()
        for j in range(1, num_items + 1):
            base_datos_completa.loc[(base_datos_completa['Instancia'] == i) & (base_datos_completa['Periodo'] == t) & (
                    base_datos_completa['Item'] == j), 'Costo_Setup'] = random.randint(round(0.9 * f * h_hat),
                                                                                       round(
                                                                                           1.1 * f * h_hat))  # costo setup

        d_hat = base_datos_completa.loc[
            (base_datos_completa['Instancia'] == i) & (base_datos_completa['Periodo'] == t), 'Demanda'].mean()
        base_datos_completa.loc[(base_datos_completa['Instancia'] == i) & (
                base_datos_completa['Periodo'] == t), 'Capacidad'] = random.randint(round(0.8 * c * d_hat), round(
            1.2 * c * d_hat))  # capacidad compartida

    return base_datos_completa, df_num_periodos_items


# Función modelo óptimo resuelto con gurobi
def optimal(base_datos_completa, df_num_periodos_items, i, c, f, semilla, tl):
    infactible = 1
    gurobi_results = {'objective_value': [], 'time': []}
    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t) &
                                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0])

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Definir el callback de Gurobi para almacenar resultados en un diccionario
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:  # Evento de solución incumbente encontrada
                # Obtener el valor de la función objetivo para la incumbente actual
                obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Obtener el tiempo transcurrido
                current_time = model.cbGet(GRB.Callback.WORK)
                # Almacenar el valor de la función objetivo y el tiempo
                gurobi_results['objective_value'].append(obj_value)
                gurobi_results['time'].append(current_time)

        # Resolución
        m.setParam('WorkLimit', tl)

        start_time = time.time()
        # Ejecutar la optimización
        m.optimize(gurobi_callback)
        end_time = time.time()
        total_time = end_time - start_time
        total_work = m.Work
        gap = m.MIPGap * 100
        # Obtener la cantidad de nodos después de la optimización
        node_count = m.getAttr(GRB.Attr.NodeCount)

        clb_obj = 0
        clb_time = 0

        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, reemplazando instancia ", i)
            base_datos_completa, df_num_periodos_items = replace_instance(c, f, semilla, base_datos_completa,
                                                                          df_num_periodos_items, i)

        else:
            try:
                # Obtener los valores de las variables
                x_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                y_values_pre = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_values[(t, j)] < 0.01:
                            y_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                        (base_datos_completa['Periodo'] == t) &
                                                        (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_values[(t, j)]
                                for t in Periodos for j in Items)
                # Obtener el valor de la función objetivo
                obj_value_pre = m.objVal
            except:
                x_values = {(t, j): 0for t in Periodos for j in Items}
                s_values = {(t, j): 0 for t in Periodos for j in Items}
                y_values = {(t, j): 0 for t in Periodos for j in Items}
                y_values_pre = {(t, j): 0 for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_values[(t, j)] < 0.01:
                            y_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_value = 0
                obj_value_pre = 0
                infactible = 0
            else:
                print("| | Modelo optimal resuelto")
                infactible = 0

    return x_values, s_values, y_values_pre, y_values, obj_value_pre, obj_value, gap, total_work, total_work, base_datos_completa, node_count, clb_time, clb_obj

def optimal_root_node(base_datos_completa, df_num_periodos_items, i, c, f, semilla, tl):
    def callback(model, where):
        if where == GRB.Callback.MIPNODE:
            # Check if the current node is the root node (node 0)
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                # Check if a relaxation solution is available
                if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                    print("\n VISITANDO NODO RAIZ \n")
                    # Get the relaxed values of the variables
                    x_values = {(t, j): model.cbGetNodeRel(x[t, j]) for t in Periodos for j in Items}
                    s_values = {(t, j): model.cbGetNodeRel(s[t, j]) for t in Periodos for j in Items}
                    y_values = {(t, j): model.cbGetNodeRel(y[t, j]) for t in Periodos for j in Items}

                    # Store or process the solution as needed
                    model._x_values = x_values
                    model._s_values = s_values
                    model._y_values = y_values
                    #model.terminate() AGREGAR ESTO HACE QUE VISITE EL NODO RAIZ UNA VEZ; EMPEORA LA SOLUCION, MEJORA EL TIEMPO

            else:
                model.terminate()

    infactible = 1
    gurobi_results = {'objective_value': [], 'time': []}
    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t) &
                                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0])

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Resolución
        gurobi_log_file = 'gurobi_optimization.log'
        m.setParam('LogFile', gurobi_log_file)
        #m.setParam('NodeLimit', 1)
        m._gurobi_results = gurobi_results

        # m.setParam('SolutionLimit', 1) #Permite devolver la primera solucion factible
        start_time = time.time()
        # Ejecutar la optimización
        m.optimize(callback)
        end_time = time.time()

        total_time = m.Work

        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, reemplazando instancia ", i)
            base_datos_completa, df_num_periodos_items = replace_instance(c, f, semilla, base_datos_completa,
                                                                          df_num_periodos_items, i)

        else:
            try:
                # Obtener los valores de las variables
                x_values = m._x_values
                s_values = m._s_values
                y_values_pre = m._y_values
                y_values = m._y_values
                for t in Periodos:
                    for j in Items:
                        if x_values[(t, j)] < 0.01:
                            y_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                        (base_datos_completa['Periodo'] == t) &
                                                        (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_values[(t, j)]
                                for t in Periodos for j in Items)
                # Obtener el valor de la función objetivo
                obj_value_pre = m.objVal
            except:
                print('| | Hubo un error, reemplazando instancia ', i)
                semilla = semilla + i
                base_datos_completa, df_num_periodos_items = replace_instance(c, f, semilla, base_datos_completa,
                                                                              df_num_periodos_items, i)
            else:
                print("| | Modelo optimal resuelto")
                infactible = 0

    return x_values, s_values, y_values_pre, obj_value_pre, obj_value, total_time, base_datos_completa


# Función modelo óptimo resuelto con gurobi
def optimal1(base_datos_completa, df_num_periodos_items, i, c, f, semilla, costo_prod, costo_inv, dem):
    infactible = 1
    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t) &
                                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0])

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Resolución
        m.Params.OutputFlag = 0  # Desactivar la salida en la consola de Gurobi
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        total_time = end_time - start_time

        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, reemplazando instancia ", i)
            base_datos_completa, df_num_periodos_items = replace_instance1(c, f, semilla, base_datos_completa,
                                                                           df_num_periodos_items, i, costo_prod,
                                                                           costo_inv, dem)

        else:
            try:
                # Obtener los valores de las variables
                x_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_values = {(t, j): y[t, j].X for t in Periodos for j in Items}

                for t in Periodos:
                    for j in Items:
                        if x_values[(t, j)] < 0.01:
                            y_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                        (base_datos_completa['Periodo'] == t) &
                                                        (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_values[(t, j)]
                                for t in Periodos for j in Items)
                # Obtener el valor de la función objetivo
                obj_value = m.objVal
            except:
                print('| | Hubo un error, reemplazando instancia ', i)
                semilla = semilla + i
                base_datos_completa, df_num_periodos_items = replace_instance1(c, f, semilla, base_datos_completa,
                                                                               df_num_periodos_items, i, costo_prod,
                                                                               costo_inv, dem)
            else:
                print("| | Modelo optimal resuelto")
                infactible = 0

    return x_values, s_values, y_values, obj_value, total_time, base_datos_completa


def optimal_relaxed_binary(base_datos_completa, df_num_periodos_items, i, c, f, semilla):
    infactible = 1
    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t) &
                                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0])

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Resolución
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        total_time = end_time - start_time

        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, reemplazando instancia ", i)
            base_datos_completa, df_num_periodos_items = replace_instance(c, f, semilla, base_datos_completa,
                                                                          df_num_periodos_items, i)

        else:
            try:
                # Obtener los valores de las variables
                x_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                y_values_pre = {(t, j): y[t, j].X for t in Periodos for j in Items}

                for t in Periodos:
                    for j in Items:
                        if x_values[(t, j)] < 0.01:
                            y_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                        (base_datos_completa['Periodo'] == t) &
                                                        (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_values[(t, j)]
                                for t in Periodos for j in Items)

                # Obtener el valor de la función objetivo
                obj_value_pre = m.objVal
            except:
                print('| | Hubo un error, reemplazando instancia ', i)
                semilla = semilla + i
                base_datos_completa, df_num_periodos_items = replace_instance(c, f, semilla, base_datos_completa,
                                                                              df_num_periodos_items, i)
            else:
                print("| | Modelo relajado resuelto")
                infactible = 0

    return x_values, s_values, y_values_pre, obj_value_pre, obj_value, m.Work, base_datos_completa


def optimal_relaxed_binary_round(base_datos_completa, df_num_periodos_items, i, c, f, semilla):
    infactible = 1
    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t) &
                                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0])

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Resolución
        m.Params.OutputFlag = 0  # Desactivar la salida en la consola de Gurobi
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        total_time = end_time - start_time

        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, reemplazando instancia ", i)
            base_datos_completa, df_num_periodos_items = replace_instance(c, f, semilla, base_datos_completa,
                                                                          df_num_periodos_items, i)

        else:
            try:
                # Obtener los valores de las variables
                x_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_values[(t, j)] < 0.01:
                            y_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                        (base_datos_completa['Periodo'] == t) &
                                                        (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_values[(t, j)]
                                for t in Periodos for j in Items)

                # Obtener el valor de la función objetivo
                obj_value = m.objVal
            except:
                print('| | Hubo un error, reemplazando instancia ', i)
                semilla = semilla + i
                base_datos_completa, df_num_periodos_items = replace_instance(c, f, semilla, base_datos_completa,
                                                                              df_num_periodos_items, i)
            else:
                print("| | Modelo relajado resuelto")

            # Conjuntos
            Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
            Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

            # Modelo
            m = gp.Model()

            # Producción
            x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
            # Stock
            s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
            # Activación
            y = m.addVars(Periodos, Items, vtype=GRB.BINARY, lb=0, ub=1, name="y")

            # Función Objetivo min costo_prod + costo_inv + costo_setup
            m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
            'Costo_Produccion'].iloc[0] * x[t, j]
                                       + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                 (base_datos_completa['Periodo'] == t) &
                                                                 (base_datos_completa['Item'] == j),
            'Costo_Inventario'].iloc[0] * s[t, j]
                                       + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                 (base_datos_completa['Periodo'] == t) &
                                                                 (base_datos_completa['Item'] == j),
            'Costo_Setup'].iloc[0] * y[t, j]
                                       for t in Periodos for j in Items), GRB.MINIMIZE)

            # Inventario
            for t in Periodos:
                for j in Items:
                    if t == max(Periodos):
                        m.addConstr(s[t, j] == 0)  # Supuesto del libro
                    if t == 1:
                        m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                       (base_datos_completa['Periodo'] == t) &
                                                                       (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])
                    else:
                        m.addConstr(
                            s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                             (base_datos_completa['Periodo'] == t) &
                                                                             (base_datos_completa['Item'] == j),
                            'Demanda'].iloc[0] + s[t, j])

            # Capacidad
            for t in Periodos:
                m.addConstr(
                    gp.quicksum(x[t, j] for j in Items) <=
                    base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                            (base_datos_completa['Periodo'] == t) &
                                            (base_datos_completa['Item'] == j),
                    'Capacidad'].iloc[0])

            # Relación binaria
            for t in Periodos:
                for j in Items:
                    m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Capacidad'].iloc[0] * y[t, j])

            # Siempre produce el primer período
            for j in Items:
                m.addConstr(y[1, j] == 1)
                for t in Periodos:
                    if t != 1:
                        m.addConstr(y[t, j] == round(y_values[(t, j)]))

            # Resolución
            m.Params.OutputFlag = 0  # Desactivar la salida en la consola de Gurobi
            start_time = time.time()
            m.optimize()
            end_time = time.time()
            total_time = end_time - start_time

            if m.status == GRB.INF_OR_UNBD:
                print("| | Infactible, reemplazando instancia ", i)

            else:
                try:
                    # Obtener los valores de las variables
                    x_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                    s_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                    y_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                    for t in Periodos:
                        for j in Items:
                            if x_values[(t, j)] < 0.01:
                                y_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                    obj_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                            (base_datos_completa['Periodo'] == t) &
                                                            (base_datos_completa['Item'] == j),
                    'Costo_Produccion'].iloc[0] * x_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                    'Costo_Inventario'].iloc[0] * s_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                    'Costo_Setup'].iloc[0] * y_values[(t, j)]
                                    for t in Periodos for j in Items)

                    # Obtener el valor de la función objetivo
                    obj_value = m.objVal
                except:
                    print('| | Hubo un error, reemplazando instancia ', i)
                    semilla = semilla + i
                    base_datos_completa, df_num_periodos_items = replace_instance(c, f, semilla, base_datos_completa,
                                                                                  df_num_periodos_items, i)
                else:
                    print("| | Modelo relajado resuelto")
                    infactible = 0

    return x_values, s_values, y_values, obj_value, total_time, base_datos_completa


# Función modelo óptimo resuelto con gurobi
def optimal_worst_case(base_datos_completa, df_num_periodos_items, i, c, f, semilla):
    infactible = 1
    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t) &
                                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0])

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce (todos los períodos, todos los items)
        for t in Periodos:
            for j in Items:
                m.addConstr(y[t, j] == 1)

        # Resolución
        m.Params.OutputFlag = 0  # Desactivar la salida en la consola de Gurobi
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        total_time = end_time - start_time

        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible instancia", i)
            x_values = {(t, j): 0 for t in Periodos for j in Items}
            s_values = {(t, j): 0 for t in Periodos for j in Items}
            y_values = {(t, j): 0 for t in Periodos for j in Items}
            obj_value = -12345

        else:
            try:
                # Intentar extraer resultados
                x_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_values[(t, j)] < 0.01:
                            y_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                        (base_datos_completa['Periodo'] == t) &
                                                        (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_values[(t, j)]
                                for t in Periodos for j in Items)
            except:
                print('| | Hubo un error en la instancia ', i)
                # semilla = semilla + i
                # base_datos_completa, df_num_periodos_items = replace_instance(c, f, semilla, base_datos_completa, df_num_periodos_items, i)
            else:
                print("| | Modelo worst resuelto")
                infactible = 0

    return x_values, s_values, y_values, obj_value, total_time, base_datos_completa


# Función modelo óptimo con fixing y tight resuelto con gurobi
def optimal_predict(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level, tight_level, reduce_level):
    infactible = 1
    last_time = 0
    w = 1
    gurobi_results = {'objective_value': [], 'time': []}

    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        filtro_binary1 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] >= pred_level)

        filtro_binary0 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] < pred_level)

        filtro_mu = (base_datos_completa['Instancia'] == i) & \
                    (base_datos_completa['pred_mu=1'] >= tight_level)

        df_mu = base_datos_completa.copy()

        filtered_df_binary1 = base_datos_completa[filtro_binary1]
        filtered_df_binary0 = base_datos_completa[filtro_binary0]

        filtered_df_mu = df_mu[filtro_mu]
        filtered_df_mu = filtered_df_mu.drop_duplicates(subset=['Periodo'])

        num_filas_binary1 = len(filtered_df_binary1)
        num_filas_binary0 = len(filtered_df_binary0)

        num_filas_mu = len(filtered_df_mu)

        print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary0} 0s AND {num_filas_binary1} 1s BINARY')
        if num_filas_mu == 0:
            tight_level = tight_level - reduce_level
            print("| | No se fijó ninguna restricción fuerte, se actualiza tight_level=", tight_level)
            continue

        # =============================================================================
        #         elif num_filas_binary1==0 or num_filas_binary0==0:
        #             pred_level = pred_level - reduce_level
        #             print("| | No se fijó ninguna variable, se actualiza pred_level=", pred_level)
        #             continue
        # =============================================================================

        # Fixing time
        start_time1 = time.time()
        # Fijar binaria 1
        for index, row in filtered_df_binary1.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            if t != 1:
                m.addConstr(y[t, j] == 1)

        # Fijar binaria 0
        for index, row in filtered_df_binary0.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            if t != 1:
                m.addConstr(y[t, j] == 0)
        end_time1 = time.time()

        # No tomo el tiempo, ya que estas restricciones 'existían' en el modelo
        # Fijar tight constraints
        # Capacidad
        t_added = []
        for index, row in filtered_df_mu.iterrows():
            t = int(row['Periodo'])
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t),
                'Capacidad'].iloc[0])
            t_added.append(t)

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])
        # Capacidad relajada
        M = gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                (base_datos_completa['Periodo'] == t) &
                                                (base_datos_completa['Item'] == j),
        'Demanda'].iloc[0] for t in Periodos for j in Items)
        # Relaxing time
        start_time2 = time.time()
        for t in Periodos:
            if t not in t_added:
                # print("ADDING RELAXED CONSTRAINT")
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <= M)

        end_time2 = time.time()

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Definir el callback de Gurobi para almacenar resultados en un diccionario
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:  # Evento de solución incumbente encontrada
                # Obtener el valor de la función objetivo para la incumbente actual
                obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Obtener el tiempo transcurrido
                current_time = model.cbGet(GRB.Callback.RUNTIME)
                # Almacenar el valor de la función objetivo y el tiempo
                gurobi_results['objective_value'].append(obj_value)
                gurobi_results['time'].append(current_time)

        # Resolución
        gurobi_log_file = 'gurobi_optimization.log'
        m.setParam('LogFile', gurobi_log_file)
        m.setParam('TimeLimit', MaxTime)
        m._gurobi_results = gurobi_results

        start_time = time.time()
        m.optimize(gurobi_callback)
        end_time = time.time()
        total_time = end_time - start_time + end_time1 - start_time1 + end_time2 - start_time2 + last_time
        last_time = total_time
        # Obtener la cantidad de nodos después de la optimización
        node_count = m.getAttr(GRB.Attr.NodeCount)

        clb_obj = gurobi_results['objective_value'][-1]
        clb_time = gurobi_results['time'][-1]

        # Leer el archivo de log de Gurobi para obtener información adicional
        def leer_log_gurobi(log_file):
            cortes_aplicados = {}
            with open(log_file, 'r') as log:
                cortar = False
                for line in log:
                    if 'Cutting planes:' in line:
                        cortar = True
                    elif cortar:
                        if line.strip() == '':
                            break  # Terminar la sección de cortes
                        partes = line.strip().split(':')
                        if len(partes) == 2:
                            corte_tipo = partes[0].strip()
                            cantidad = int(partes[1].strip())
                            cortes_aplicados[corte_tipo] = cantidad
            return cortes_aplicados

        # Leer el archivo de log para obtener los cortes aplicados
        cortes_aplicados = leer_log_gurobi(gurobi_log_file)

        # Calcular la suma total de los cortes aplicados
        cuts_count = sum(cortes_aplicados.values())
        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, se reducirá pred_level en ", reduce_level)
            pred_level = pred_level - reduce_level
            w += 1

        elif pred_level < 0:
            print('pred_level alcanzó un valor negativo, se ocupará pred_level=0')
            pred_level = 0

        else:
            try:
                # Intentar extraer resultados
                x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_hat_values[(t, j)] < 0.1:
                            y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                            (base_datos_completa['Periodo'] == t) &
                                                            (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                    for t in Periodos for j in Items)
            except:
                pred_level = pred_level - reduce_level
                print('| | ***Hubo un error, se actualizará pred_level = ', pred_level)
                w += 1

            else:
                infactible = 0  # Marcar como factible y salir del bucle
                print('| | Modelo factible')

    return x_hat_values, s_hat_values, y_hat_values, obj_hat_value, total_time, pred_level, num_filas_binary0, num_filas_binary1, num_filas_mu, w, node_count, cuts_count, clb_time, clb_obj


# Función modelo óptimo con fixing y tight resuelto con gurobi
def optimal_predict_hint_twoML(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level1, pred_level2,
                               tight_level, reduce_level):
    infactible = 1
    last_time = 0
    w = 1
    gurobi_results = {'objective_value': [], 'time': []}

    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Filtros para el modelo ML1
        filtro_binary1_ML1 = (base_datos_completa['Instancia'] == i) & \
                             (base_datos_completa['pred_binary=1_ML1'] >= pred_level1)

        filtro_binary0_ML1 = (base_datos_completa['Instancia'] == i) & \
                             (base_datos_completa['pred_binary=1_ML1'] < pred_level1)

        # Filtros para el modelo ML2
        filtro_binary1_ML2 = (base_datos_completa['Instancia'] == i) & \
                             (base_datos_completa['pred_binary=1_ML2'] >= pred_level2)

        filtro_binary0_ML2 = (base_datos_completa['Instancia'] == i) & \
                             (base_datos_completa['pred_binary=1_ML2'] < pred_level2)

        filtro_mu = (base_datos_completa['Instancia'] == i) & \
                    (base_datos_completa['pred_mu=1'] >= tight_level)

        # Copia del DataFrame para evitar modificar el original
        df_mu = base_datos_completa.copy()

        # Filtrado de datos para ML1
        filtered_df_binary1_ML1 = base_datos_completa[filtro_binary1_ML1]
        filtered_df_binary0_ML1 = base_datos_completa[filtro_binary0_ML1]

        # Filtrado de datos para ML2
        filtered_df_binary1_ML2 = base_datos_completa[filtro_binary1_ML2]
        filtered_df_binary0_ML2 = base_datos_completa[filtro_binary0_ML2]

        filtered_df_mu = df_mu[filtro_mu]
        filtered_df_mu = filtered_df_mu.drop_duplicates(subset=['Periodo'])

        # num_filas_binary1 = len(filtered_df_binary1)
        # num_filas_binary0 = len(filtered_df_binary)

        num_filas_mu = len(filtered_df_mu)

        # print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary0} 0s AND {num_filas_binary1} 1s BINARY')
        if num_filas_mu == 0:
            tight_level = tight_level - reduce_level
            print("| | No se fijó ninguna restricción fuerte, se actualiza tight_level=", tight_level)
            continue

        # =============================================================================
        #         elif num_filas_binary1==0 or num_filas_binary0==0:
        #             pred_level = pred_level - reduce_level
        #             print("| | No se fijó ninguna variable, se actualiza pred_level=", pred_level)
        #             continue
        # =============================================================================

        # Fixing time
        start_time1 = time.time()
        # Fijar binaria 1
        # Iterar sobre los datos filtrados para binaria 1 (ambos modelos predicen 1)
        num_filas_binary1 = 0
        for index, row in base_datos_completa.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            pred_ml1 = row['pred_binary=1_ML1']
            pred_ml2 = row['pred_binary=1_ML2']

            if pred_ml1 >= pred_level1 and pred_ml2 >= pred_level2:
                pri = int(max(pred_ml1, pred_ml2) * 1000)  # Priorizamos el menor score entre ambos
                val = 1
                num_filas_binary1 = num_filas_binary1 + 1
                if t != 1:
                    y[t, j].varhintval = val
                    y[t, j].varhintpri = pri

        num_filas_binary0 = 0

        # Iterar sobre los datos filtrados para binaria 0 (ambos modelos predicen 0)
        for index, row in base_datos_completa.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            pred_ml1 = row['pred_binary=0_ML1']
            pred_ml2 = row['pred_binary=0_ML2']

            if pred_ml1 < pred_level1 and pred_ml2 < pred_level2:
                pri = int((max(pred_ml1, pred_ml2)) * 1000)  # Priorizamos el mayor score invertido
                val = 0
                num_filas_binary0 = num_filas_binary0 + 1

                if t != 1:
                    y[t, j].varhintval = val
                    y[t, j].varhintpri = pri

        end_time1 = time.time()

        # No tomo el tiempo, ya que estas restricciones 'existían' en el modelo
        # Fijar tight constraints
        # Capacidad
        t_added = []
        for index, row in filtered_df_mu.iterrows():
            t = int(row['Periodo'])
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t),
                'Capacidad'].iloc[0])
            t_added.append(t)

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])
        # Capacidad relajada
        M = gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                (base_datos_completa['Periodo'] == t) &
                                                (base_datos_completa['Item'] == j),
        'Demanda'].iloc[0] for t in Periodos for j in Items)
        # Relaxing time
        start_time2 = time.time()
        for t in Periodos:
            if t not in t_added:
                # print("ADDING RELAXED CONSTRAINT")
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <= M)

        end_time2 = time.time()

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Resolución
        # Definir el callback de Gurobi para almacenar resultados en un diccionario
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:  # Evento de solución incumbente encontrada
                # Obtener el valor de la función objetivo para la incumbente actual
                obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Obtener el tiempo transcurrido
                current_time = model.cbGet(GRB.Callback.RUNTIME)
                # Almacenar el valor de la función objetivo y el tiempo
                gurobi_results['objective_value'].append(obj_value)
                gurobi_results['time'].append(current_time)

            # Resolución
            gurobi_log_file = 'gurobi_optimization.log'
            m.setParam('LogFile', gurobi_log_file)
            m.setParam('WorkLimit', MaxTime)
            # m.setParam('Cuts', 0)
            # m.setParam('Heuristics', 0)

            start_time = time.time()
            m.optimize(gurobi_callback)
            end_time = time.time()
            model_time = end_time1 - start_time1 + end_time2 - start_time2 + last_time
            optimize_time = end_time - start_time
            total_time = model_time + optimize_time
            last_time = total_time
            total_work = m.Work
            GAP = m.MIPGap
            # Obtener la cantidad de nodos después de la optimización
            node_count = m.getAttr(GRB.Attr.NodeCount)

            clb_obj = gurobi_results['objective_value'][-1]
            clb_time = gurobi_results['time'][-1]

            # Leer el archivo de log de Gurobi para obtener información adicional
            def leer_log_gurobi(log_file):
                cortes_aplicados = {}
                with open(log_file, 'r') as log:
                    cortar = False
                    for line in log:
                        if 'Cutting planes:' in line:
                            cortar = True
                        elif cortar:
                            if line.strip() == '':
                                break  # Terminar la sección de cortes
                            partes = line.strip().split(':')
                            if len(partes) == 2:
                                corte_tipo = partes[0].strip()
                                cantidad = int(partes[1].strip())
                                cortes_aplicados[corte_tipo] = cantidad
                return cortes_aplicados

            # Leer el archivo de log para obtener los cortes aplicados
            cortes_aplicados = leer_log_gurobi(gurobi_log_file)

            # Calcular la suma total de los cortes aplicados
            cuts_count = sum(cortes_aplicados.values())
            if m.status == GRB.INF_OR_UNBD:
                print("| | Infactible, se reducirá pred_level en ", reduce_level)
                pred_level = pred_level - reduce_level
                w += 1

            elif pred_level < 0:
                print('pred_level alcanzó un valor negativo, se ocupará pred_level=0')
                pred_level = 0

            else:
                try:
                    # Intentar extraer resultados
                    x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                    s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                    y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                    for t in Periodos:
                        for j in Items:
                            if x_hat_values[(t, j)] < 0.1:
                                y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                    obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                (base_datos_completa['Periodo'] == t) &
                                                                (base_datos_completa['Item'] == j),
                    'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                        + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                  (base_datos_completa['Periodo'] == t) &
                                                                  (base_datos_completa['Item'] == j),
                    'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                        + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                  (base_datos_completa['Periodo'] == t) &
                                                                  (base_datos_completa['Item'] == j),
                    'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                        for t in Periodos for j in Items)
                except:
                    pred_level = pred_level - reduce_level
                    print('| | ***Hubo un error, se actualizará pred_level = ', pred_level)
                    w += 1

                else:
                    infactible = 0  # Marcar como factible y salir del bucle
                    print('| | Modelo factible')

        return x_hat_values, s_hat_values, y_hat_values, obj_hat_value, model_time, optimize_time, total_time, total_work, GAP, pred_level, num_filas_binary0, num_filas_binary1, num_filas_mu, w, node_count, cuts_count, clb_time, clb_obj


# Función modelo óptimo con fixing y tight resuelto con gurobi
def optimal_predict_hint(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level, tight_level, reduce_level):
    infactible = 1
    last_time = 0
    w = 1
    gurobi_results = {'objective_value': [], 'time': []}

    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        filtro_binary1 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] >= pred_level)

        filtro_binary0 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] < pred_level)

        filtro_mu = (base_datos_completa['Instancia'] == i) & \
                    (base_datos_completa['pred_mu=1'] >= tight_level)

        df_mu = base_datos_completa.copy()

        filtered_df_binary1 = base_datos_completa[filtro_binary1]
        filtered_df_binary0 = base_datos_completa[filtro_binary0]

        filtered_df_mu = df_mu[filtro_mu]
        filtered_df_mu = filtered_df_mu.drop_duplicates(subset=['Periodo'])

        num_filas_binary1 = len(filtered_df_binary1)
        num_filas_binary0 = len(filtered_df_binary0)

        num_filas_mu = len(filtered_df_mu)

        print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary0} 0s AND {num_filas_binary1} 1s BINARY')
        if num_filas_mu == 0:
            tight_level = tight_level - reduce_level
            print("| | No se fijó ninguna restricción fuerte, se actualiza tight_level=", tight_level)
            continue

        # =============================================================================
        #         elif num_filas_binary1==0 or num_filas_binary0==0:
        #             pred_level = pred_level - reduce_level
        #             print("| | No se fijó ninguna variable, se actualiza pred_level=", pred_level)
        #             continue
        # =============================================================================

        # Fixing time
        start_time1 = time.time()
        # Fijar binaria 1
        for index, row in filtered_df_binary1.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            pri = int(row['pred_binary=1'] * 1000)
            val = 1
            if t != 1:
                y[t, j].varhintval = val
                y[t, j].varhintpri = pri

        # Fijar binaria 0
        for index, row in filtered_df_binary0.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            pri = int(row['pred_binary=0'] * 1000)
            val = 0
            if t != 1:
                y[t, j].varhintval = val
                y[t, j].varhintpri = pri

        end_time1 = time.time()

        # No tomo el tiempo, ya que estas restricciones 'existían' en el modelo
        # Fijar tight constraints
        # Capacidad
        t_added = []
        for index, row in filtered_df_mu.iterrows():
            t = int(row['Periodo'])
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t),
                'Capacidad'].iloc[0])
            t_added.append(t)

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])
        # Capacidad relajada
        M = gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                (base_datos_completa['Periodo'] == t) &
                                                (base_datos_completa['Item'] == j),
        'Demanda'].iloc[0] for t in Periodos for j in Items)
        # Relaxing time
        start_time2 = time.time()
        for t in Periodos:
            if t not in t_added:
                # print("ADDING RELAXED CONSTRAINT")
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <= M)

        end_time2 = time.time()

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Resolución
        # Definir el callback de Gurobi para almacenar resultados en un diccionario
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:  # Evento de solución incumbente encontrada
                # Obtener el valor de la función objetivo para la incumbente actual
                obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Obtener el tiempo transcurrido
                current_time = model.cbGet(GRB.Callback.RUNTIME)
                # Almacenar el valor de la función objetivo y el tiempo
                gurobi_results['objective_value'].append(obj_value)
                gurobi_results['time'].append(current_time)

        # Resolución
        gurobi_log_file = 'gurobi_optimization.log'
        m.setParam('LogFile', gurobi_log_file)
        m.setParam('TimeLimit', MaxTime)

        start_time = time.time()
        m.optimize(gurobi_callback)
        end_time = time.time()
        total_time = end_time - start_time + end_time1 - start_time1 + end_time2 - start_time2 + last_time
        last_time = total_time

        # Obtener la cantidad de nodos después de la optimización
        node_count = m.getAttr(GRB.Attr.NodeCount)

        clb_obj = gurobi_results['objective_value'][-1]
        clb_time = gurobi_results['time'][-1]

        # Leer el archivo de log de Gurobi para obtener información adicional
        def leer_log_gurobi(log_file):
            cortes_aplicados = {}
            with open(log_file, 'r') as log:
                cortar = False
                for line in log:
                    if 'Cutting planes:' in line:
                        cortar = True
                    elif cortar:
                        if line.strip() == '':
                            break  # Terminar la sección de cortes
                        partes = line.strip().split(':')
                        if len(partes) == 2:
                            corte_tipo = partes[0].strip()
                            cantidad = int(partes[1].strip())
                            cortes_aplicados[corte_tipo] = cantidad
            return cortes_aplicados

        # Leer el archivo de log para obtener los cortes aplicados
        cortes_aplicados = leer_log_gurobi(gurobi_log_file)

        # Calcular la suma total de los cortes aplicados
        cuts_count = sum(cortes_aplicados.values())
        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, se reducirá pred_level en ", reduce_level)
            pred_level = pred_level - reduce_level
            w += 1

        elif pred_level < 0:
            print('pred_level alcanzó un valor negativo, se ocupará pred_level=0')
            pred_level = 0

        else:
            try:
                # Intentar extraer resultados
                x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_hat_values[(t, j)] < 0.1:
                            y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                            (base_datos_completa['Periodo'] == t) &
                                                            (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                    for t in Periodos for j in Items)
            except:
                pred_level = pred_level - reduce_level
                print('| | ***Hubo un error, se actualizará pred_level = ', pred_level)
                w += 1

            else:
                infactible = 0  # Marcar como factible y salir del bucle
                print('| | Modelo factible')

    return x_hat_values, s_hat_values, y_hat_values, obj_hat_value, total_time, pred_level, num_filas_binary0, num_filas_binary1, num_filas_mu, w, node_count, cuts_count, clb_time, clb_obj


# Función modelo óptimo con fixing y tight resuelto con gurobi
def optimal_predict_warm(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level, tight_level, reduce_level):
    infactible = 1
    last_time = 0
    w = 1
    gurobi_results = {'objective_value': [], 'time': []}

    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        filtro_binary1 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] >= pred_level)

        filtro_binary0 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] < pred_level)

        filtro_mu = (base_datos_completa['Instancia'] == i) & \
                    (base_datos_completa['pred_mu=1'] >= tight_level)

        df_mu = base_datos_completa.copy()

        filtered_df_binary1 = base_datos_completa[filtro_binary1]
        filtered_df_binary0 = base_datos_completa[filtro_binary0]

        filtered_df_mu = df_mu[filtro_mu]
        filtered_df_mu = filtered_df_mu.drop_duplicates(subset=['Periodo'])

        num_filas_binary1 = len(filtered_df_binary1)
        num_filas_binary0 = len(filtered_df_binary0)

        num_filas_mu = len(filtered_df_mu)

        print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary0} 0s AND {num_filas_binary1} 1s BINARY')
        if num_filas_mu == 0:
            tight_level = tight_level - reduce_level
            print("| | No se fijó ninguna restricción fuerte, se actualiza tight_level=", tight_level)
            continue

        # =============================================================================
        #         elif num_filas_binary1==0 or num_filas_binary0==0:
        #             pred_level = pred_level - reduce_level
        #             print("| | No se fijó ninguna variable, se actualiza pred_level=", pred_level)
        #             continue
        # =============================================================================

        # Fixing time
        start_time1 = time.time()
        # Fijar binaria 1
        for index, row in filtered_df_binary1.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            pri = int(row['pred_binary=1'] * 1000)
            val = 1
            if t != 1:
                y[t, j].Start = val

        # Fijar binaria 0
        for index, row in filtered_df_binary0.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            pri = int(row['pred_binary=0'] * 1000)
            val = 0
            if t != 1:
                y[t, j].Start = val

        end_time1 = time.time()

        # No tomo el tiempo, ya que estas restricciones 'existían' en el modelo
        # Fijar tight constraints
        # Capacidad
        t_added = []
        for index, row in filtered_df_mu.iterrows():
            t = int(row['Periodo'])
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t),
                'Capacidad'].iloc[0])
            t_added.append(t)

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])
        # Capacidad relajada
        M = gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                (base_datos_completa['Periodo'] == t) &
                                                (base_datos_completa['Item'] == j),
        'Demanda'].iloc[0] for t in Periodos for j in Items)
        # Relaxing time
        start_time2 = time.time()
        for t in Periodos:
            if t not in t_added:
                # print("ADDING RELAXED CONSTRAINT")
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <= M)

        end_time2 = time.time()

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Resolución
        # Definir el callback de Gurobi para almacenar resultados en un diccionario
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:  # Evento de solución incumbente encontrada
                # Obtener el valor de la función objetivo para la incumbente actual
                obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Obtener el tiempo transcurrido
                current_time = model.cbGet(GRB.Callback.RUNTIME)
                # Almacenar el valor de la función objetivo y el tiempo
                gurobi_results['objective_value'].append(obj_value)
                gurobi_results['time'].append(current_time)

        # Resolución
        gurobi_log_file = 'gurobi_optimization.log'
        m.setParam('LogFile', gurobi_log_file)
        m.setParam('TimeLimit', MaxTime)
        start_time = time.time()
        m.optimize(gurobi_callback)
        end_time = time.time()
        total_time = end_time - start_time + end_time1 - start_time1 + end_time2 - start_time2 + last_time
        last_time = total_time
        # Obtener la cantidad de nodos después de la optimización
        node_count = m.getAttr(GRB.Attr.NodeCount)

        clb_obj = gurobi_results['objective_value'][-1]
        clb_time = gurobi_results['time'][-1]

        # Leer el archivo de log de Gurobi para obtener información adicional
        def leer_log_gurobi(log_file):
            cortes_aplicados = {}
            with open(log_file, 'r') as log:
                cortar = False
                for line in log:
                    if 'Cutting planes:' in line:
                        cortar = True
                    elif cortar:
                        if line.strip() == '':
                            break  # Terminar la sección de cortes
                        partes = line.strip().split(':')
                        if len(partes) == 2:
                            corte_tipo = partes[0].strip()
                            cantidad = int(partes[1].strip())
                            cortes_aplicados[corte_tipo] = cantidad
            return cortes_aplicados

        # Leer el archivo de log para obtener los cortes aplicados
        cortes_aplicados = leer_log_gurobi(gurobi_log_file)

        # Calcular la suma total de los cortes aplicados
        cuts_count = sum(cortes_aplicados.values())

        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, se reducirá pred_level en ", reduce_level)
            pred_level = pred_level - reduce_level
            w += 1

        elif pred_level < 0:
            print('pred_level alcanzó un valor negativo, se ocupará pred_level=0')
            pred_level = 0

        else:
            try:
                # Intentar extraer resultados
                x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_hat_values[(t, j)] < 0.1:
                            y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                            (base_datos_completa['Periodo'] == t) &
                                                            (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                    for t in Periodos for j in Items)
            except:
                pred_level = pred_level - reduce_level
                print('| | ***Hubo un error, se actualizará pred_level = ', pred_level)
                w += 1

            else:
                infactible = 0  # Marcar como factible y salir del bucle
                print('| | Modelo factible')

    return x_hat_values, s_hat_values, y_hat_values, obj_hat_value, total_time, pred_level, num_filas_binary0, num_filas_binary1, num_filas_mu, w, node_count, cuts_count, clb_time, clb_obj


# Función modelo óptimo con fixing y tight resuelto con gurobi
def optimal_relax_and_predict(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level, tight_level,
                              reduce_level, model_y, model_mu):
    # Conjuntos
    base_datos_completa_COPY = base_datos_completa.copy()
    Periodos_total = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 2)
    Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)
    Already_fixed = pd.DataFrame(index=Periodos_total)
    total_time_ALL = 0
    last_time = 0

    for T_AUX in Periodos_total:
        total_time_ALL += total_time_ALL
        base_datos_completa = base_datos_completa_COPY[base_datos_completa_COPY['Periodo'] <= T_AUX]
        print(T_AUX)
        if T_AUX == 61:
            Periodos = range(1, 61)
        else:
            Periodos = range(1, T_AUX + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t) &
                                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0])

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Fija las predicciones
        num_filas_binary1 = 0
        print(Already_fixed)

        for t in range(1, T_AUX):
            for j in Items:
                prediccion = Already_fixed.loc[t, j]
                print('FIJANDO PREDICCION ', prediccion)
                if prediccion >= 0.95:
                    num_filas_binary1 += 1
                    m.addConstr(y[t, j] == 1)
        print(num_filas_binary1)
        # Resolución
        m.Params.OutputFlag = 0  # Desactivar la salida en la consola de Gurobi
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        total_time_relax = end_time - start_time

        # Obtener los valores de las variables
        x_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
        s_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
        y_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
        for t in Periodos:
            for j in Items:
                # Filter the dataframe for the specific 'Periodo' and 'Item'
                mask = (base_datos_completa['Instancia'] == i) & (base_datos_completa['Periodo'] == t) & (
                        base_datos_completa['Item'] == j)

                # Update the columns with the corresponding values
                base_datos_completa.loc[mask, 'x_relaxed'] = x_values[(t, j)]
                base_datos_completa.loc[mask, 's_relaxed'] = s_values[(t, j)]
                base_datos_completa.loc[mask, 'y_relaxed'] = y_values[(t, j)]

        print('------ Predicting -------')
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        train_max_periods = 60  # Cantidad de períodos del modelo entrenado
        test_max_periodos = 60  # Cantidad de períodos de instancia a testear
        test_items = 8  # Cantidad de items de la instancia a testear

        # Estos son los nuevos en base a los mejores que salieron de las instancias de 60 periodos en el train
        y_features = ['Costo_Produccion', 'Demanda', 'Costo_Setup', 'Capacidad', 'x_relaxed', 's_relaxed', 'y_relaxed',
                      'x_worst']
        mu_features = ['Costo_Produccion', 'Capacidad', 's_relaxed', 'y_relaxed', 'x_worst', 's_worst']

        df = pd.read_csv(f'Datos_entrenamiento\Datos_entrenamiento_{train_max_periods}.csv')
        # Importar instancia de test y modelo de ML
        test_base_datos_completa = pd.read_csv(f'Datos_entrenamiento\Datos_test_{test_max_periodos}.csv')

        test_df_num_periodos_items = pd.read_excel(f'Informacion_instancias_Train_{train_max_periods}.xlsx',
                                                   sheet_name=f'Test_optimal_{test_max_periodos}')

        X_mu = test_base_datos_completa.copy()
        X_mu = X_mu.groupby(['Instancia', 'Periodo']).sum().reset_index()

        # PREDECIR PARA VARIOS PRED_LEVEL
        X = test_base_datos_completa.loc[:, y_features]
        X_mu = X_mu.loc[:, mu_features]

        test_base_datos_completa = predicting_values(X, X_mu, test_base_datos_completa, model_mu, model_y, test_items)
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        print('------ Predicting succesfuly-------')

        for j in Items:
            if T_AUX <= 60:
                print('Checking ', T_AUX)
                Valor_prediccion = base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == T_AUX) &
                                                           (base_datos_completa['Item'] == j),
                'pred_binary=1'].iloc[0]

                Already_fixed.loc[T_AUX, j] = int(Valor_prediccion >= pred_level)
                print('SE PREDICE QUE LA PREDICCION ', Valor_prediccion, ' TOMARA VALOR ', Already_fixed.loc[T_AUX, j])

        if T_AUX == 61:
            # Modelo
            m = gp.Model()

            # Producción
            x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
            # Stock
            s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
            # Activación
            y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

            # Función Objetivo min costo_prod + costo_inv + costo_setup
            m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
            'Costo_Produccion'].iloc[0] * x[t, j]
                                       + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                 (base_datos_completa['Periodo'] == t) &
                                                                 (base_datos_completa['Item'] == j),
            'Costo_Inventario'].iloc[0] * s[t, j]
                                       + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                 (base_datos_completa['Periodo'] == t) &
                                                                 (base_datos_completa['Item'] == j),
            'Costo_Setup'].iloc[0] * y[t, j]
                                       for t in Periodos for j in Items), GRB.MINIMIZE)

            # Inventario
            for t in Periodos:
                for j in Items:
                    if t == max(Periodos):
                        m.addConstr(s[t, j] == 0)  # Supuesto del libro
                    if t == 1:
                        m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                       (base_datos_completa['Periodo'] == t) &
                                                                       (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])
                    else:
                        m.addConstr(
                            s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                             (base_datos_completa['Periodo'] == t) &
                                                                             (base_datos_completa['Item'] == j),
                            'Demanda'].iloc[0] + s[t, j])

            # Capacidad
            for t in Periodos:
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <=
                            base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                    (base_datos_completa['Periodo'] == t) &
                                                    (base_datos_completa['Item'] == j),
                            'Capacidad'].iloc[0])

            # Relación binaria
            for t in Periodos:
                for j in Items:
                    m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Capacidad'].iloc[0] * y[t, j])

            # Siempre produce el primer período
            for j in Items:
                m.addConstr(y[1, j] == 1)

            for t in range(1, T_AUX):
                for j in Items:
                    prediccion = Already_fixed.loc[t, j]
                    print('FIJANDO PREDICCION ', prediccion)
                    if prediccion >= 0.95:
                        num_filas_binary1 += 1
                        m.addConstr(y[t, j] == 1)

            # Resolución
            m.Params.OutputFlag = 0  # Desactivar la salida en la consola de Gurobi
            m.setParam('TimeLimit', MaxTime)  # Se lo puse porque a veces sucede que se demora más que el optimal
            start_time = time.time()
            m.optimize()
            end_time = time.time()
            total_time = end_time - start_time
            last_time = total_time + last_time
            # Intentar extraer resultados
            x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
            s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
            y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
            total_time_ALL = total_time_relax + total_time
            for j in Items:
                if T_AUX in range(1, 61):
                    Already_fixed.loc[T_AUX, j] = y_hat_values[(T_AUX, j)]
            for t in Periodos:
                for j in Items:
                    if x_hat_values[(t, j)] < 0.1:
                        y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
            obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                        (base_datos_completa['Periodo'] == t) &
                                                        (base_datos_completa['Item'] == j),
            'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
            'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
            'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                for t in Periodos for j in Items)
            print('FINAL ITERACIÓN')

        print(Already_fixed)

    return x_hat_values, s_hat_values, y_hat_values, obj_hat_value, total_time, pred_level, num_filas_binary1


def optimal_relax_and_predict_ALLPERIOD(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level, tight_level,
                                        reduce_level, model_y, model_mu):
    w = 1
    # Conjuntos
    base_datos_completa_COPY = base_datos_completa.copy()
    Periodos_total = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 2)
    Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)
    fix = {}

    total_time_ALL = 0
    for T_AUX in Periodos_total:
        base_datos_completa = base_datos_completa_COPY[base_datos_completa_COPY['Periodo'] <= 60]
        print('Empezó T_AUX ', T_AUX)
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)
        infactible = 1
        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")
        print('PASO')

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t) &
                                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0])

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Fija las predicciones
        num_filas_binary1 = 0
        for t in range(1, T_AUX):
            for j in Items:
                prediccion = fix[(t, j)]
                print('FIJANDO PREDICCION ', prediccion)
                if prediccion >= 0.95:
                    num_filas_binary1 += 1
                    m.addConstr(y[t, j] == 1)
        print(num_filas_binary1)
        # Resolución
        m.Params.OutputFlag = 0  # Desactivar la salida en la consola de Gurobi
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        total_time_relax = end_time - start_time

        # Obtener los valores de las variables
        x_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
        s_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
        y_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
        for t in Periodos:
            for j in Items:
                # Filter the dataframe for the specific 'Periodo' and 'Item'
                mask = (base_datos_completa['Instancia'] == i) & (base_datos_completa['Periodo'] == t) & (
                        base_datos_completa['Item'] == j)

                # Update the columns with the corresponding values
                base_datos_completa.loc[mask, 'x_relaxed'] = x_values[(t, j)]
                base_datos_completa.loc[mask, 's_relaxed'] = s_values[(t, j)]
                base_datos_completa.loc[mask, 'y_relaxed'] = y_values[(t, j)]

                if x_values[(t, j)] < 0.1:
                    y_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
        obj_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                (base_datos_completa['Periodo'] == t) &
                                                (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x_values[(t, j)]
                        + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                  (base_datos_completa['Periodo'] == t) &
                                                  (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s_values[(t, j)]
                        + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                  (base_datos_completa['Periodo'] == t) &
                                                  (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y_values[(t, j)]
                        for t in Periodos for j in Items)

        print('------ Predicting -------')
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        train_max_periods = 60  # Cantidad de períodos del modelo entrenado
        test_max_periodos = 60  # Cantidad de períodos de instancia a testear
        test_items = 8  # Cantidad de items de la instancia a testear

        # Estos son los nuevos en base a los mejores que salieron de las instancias de 60 periodos en el train
        y_features = ['Costo_Produccion', 'Demanda', 'Costo_Setup', 'Capacidad', 'x_relaxed', 's_relaxed', 'y_relaxed',
                      'x_worst']
        mu_features = ['Costo_Produccion', 'Capacidad', 's_relaxed', 'y_relaxed', 'x_worst', 's_worst']

        df = pd.read_csv(f'Datos_entrenamiento\Datos_entrenamiento_{train_max_periods}.csv')
        # Importar instancia de test y modelo de ML
        test_base_datos_completa = pd.read_csv(f'Datos_entrenamiento\Datos_test_{test_max_periodos}.csv')

        test_df_num_periodos_items = pd.read_excel(f'Informacion_instancias_Train_{train_max_periods}.xlsx',
                                                   sheet_name=f'Test_optimal_{test_max_periodos}')

        X_mu = test_base_datos_completa.copy()
        X_mu = X_mu.groupby(['Instancia', 'Periodo']).sum().reset_index()

        # PREDECIR PARA VARIOS PRED_LEVEL
        X = test_base_datos_completa.loc[:, y_features]
        X_mu = X_mu.loc[:, mu_features]

        test_base_datos_completa = predicting_values(X, X_mu, test_base_datos_completa, model_mu, model_y, test_items)
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        ######################### AQUI SE ACTUALIZAN LAS PREDICCIONES ######################################################
        print('------ Predicting succesfuly-------')

        for j in Items:
            if T_AUX <= 60:
                print('Checking ', T_AUX)
                Valor_prediccion = base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == T_AUX) &
                                                           (base_datos_completa['Item'] == j),
                'pred_binary=1'].iloc[0]

                print('SE PREDICE QUE LA PREDICCION ', Valor_prediccion, ' TOMARA VALOR ',
                      int(Valor_prediccion >= pred_level))
                fix[(T_AUX, j)] = int(Valor_prediccion >= pred_level)

        total_time_ALL = total_time_ALL + total_time_relax

        if T_AUX == 61:
            print('RESOLVIENDO MODELO NORMAL')
            # Modelo
            m = gp.Model()

            # Producción
            x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
            # Stock
            s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
            # Activación
            y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

            # Función Objetivo min costo_prod + costo_inv + costo_setup
            m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
            'Costo_Produccion'].iloc[0] * x[t, j]
                                       + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                 (base_datos_completa['Periodo'] == t) &
                                                                 (base_datos_completa['Item'] == j),
            'Costo_Inventario'].iloc[0] * s[t, j]
                                       + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                 (base_datos_completa['Periodo'] == t) &
                                                                 (base_datos_completa['Item'] == j),
            'Costo_Setup'].iloc[0] * y[t, j]
                                       for t in Periodos for j in Items), GRB.MINIMIZE)

            # Inventario
            for t in Periodos:
                for j in Items:
                    if t == max(Periodos):
                        m.addConstr(s[t, j] == 0)  # Supuesto del libro
                    if t == 1:
                        m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                       (base_datos_completa['Periodo'] == t) &
                                                                       (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])
                    else:
                        m.addConstr(
                            s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                             (base_datos_completa['Periodo'] == t) &
                                                                             (base_datos_completa['Item'] == j),
                            'Demanda'].iloc[0] + s[t, j])

            # Capacidad
            for t in Periodos:
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <=
                            base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                    (base_datos_completa['Periodo'] == t) &
                                                    (base_datos_completa['Item'] == j),
                            'Capacidad'].iloc[0])

            # Relación binaria
            for t in Periodos:
                for j in Items:
                    m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Capacidad'].iloc[0] * y[t, j])

            # Siempre produce el primer período
            for j in Items:
                m.addConstr(y[1, j] == 1)

            num_filas_binary1 = 0
            for t in range(1, T_AUX):
                for j in Items:
                    prediccion = fix[(t, j)]
                    print('FIJANDO PREDICCION ', prediccion)
                    if prediccion >= 0.95:
                        num_filas_binary1 += 1
                        m.addConstr(y[t, j] == 1)
            print('En el modelo normal se fijaron ', num_filas_binary1)
            # Resolución
            m.Params.OutputFlag = 0  # Desactivar la salida en la consola de Gurobi
            m.setParam('TimeLimit', MaxTime)  # Se lo puse porque a veces sucede que se demora más que el optimal
            start_time = time.time()
            m.optimize()
            end_time = time.time()
            total_time = end_time - start_time
            # Intentar extraer resultados
            x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
            s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
            y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
            total_time_ALL = total_time_relax + total_time
            for j in Items:
                if T_AUX in range(1, 61):
                    fix[(T_AUX, j)] = y_hat_values[(T_AUX, j)]
            for t in Periodos:
                for j in Items:
                    if x_hat_values[(t, j)] < 0.1:
                        y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
            obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                        (base_datos_completa['Periodo'] == t) &
                                                        (base_datos_completa['Item'] == j),
            'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
            'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                          (base_datos_completa['Periodo'] == t) &
                                                          (base_datos_completa['Item'] == j),
            'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                for t in Periodos for j in Items)
            print('FINAL ITERACIÓN')
    return x_values, s_values, y_values, obj_hat_value, total_time_ALL, num_filas_binary1


# Función modelo óptimo con fixing y tight resuelto con gurobi con dos pred_levels
def optimal_predict_bi_pred(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level0, pred_level1,
                            tight_level, reduce_level):
    infactible = 1
    last_time = 0
    w = 1
    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        # Filtrados de las variables binarias basados en dos niveles de predicción
        filtro_binary1 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] >= pred_level1)

        filtro_binary0 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] < pred_level0)

        filtro_mu = (base_datos_completa['Instancia'] == i) & \
                    (base_datos_completa['pred_mu=1'] >= tight_level)

        df_mu = base_datos_completa.copy()

        filtered_df_binary1 = base_datos_completa[filtro_binary1]
        filtered_df_binary0 = base_datos_completa[filtro_binary0]

        filtered_df_mu = df_mu[filtro_mu]
        filtered_df_mu = filtered_df_mu.drop_duplicates(subset=['Periodo'])

        num_filas_binary1 = len(filtered_df_binary1)
        num_filas_binary0 = len(filtered_df_binary0)

        num_filas_mu = len(filtered_df_mu)

        print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary0} 0s AND {num_filas_binary1} 1s BINARY')
        if num_filas_mu == 0:
            tight_level = tight_level - reduce_level
            print("| | No se fijó ninguna restricción fuerte, se actualiza tight_level=", tight_level)
            continue

        # Fixing time
        start_time1 = time.time()
        # Fijar binaria 1
        for index, row in filtered_df_binary1.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            if t != 1:
                m.addConstr(y[t, j] == 1)

        # Fijar binaria 0
        for index, row in filtered_df_binary0.iterrows():
            t, j = int(row['Periodo']), int(row['Item'])
            if t != 1:
                m.addConstr(y[t, j] == 0)
        end_time1 = time.time()

        # No tomo el tiempo, ya que estas restricciones 'existían' en el modelo
        # Fijar tight constraints
        # Capacidad
        t_added = []
        for index, row in filtered_df_mu.iterrows():
            t = int(row['Periodo'])
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t),
                'Capacidad'].iloc[0])
            t_added.append(t)

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])
        # Capacidad relajada
        M = gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                (base_datos_completa['Periodo'] == t) &
                                                (base_datos_completa['Item'] == j),
        'Demanda'].iloc[0] for t in Periodos for j in Items)
        # Relaxing time
        start_time2 = time.time()
        for t in Periodos:
            if t not in t_added:
                # print("ADDING RELAXED CONSTRAINT")
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <= M)

        end_time2 = time.time()

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Resolución
        m.Params.OutputFlag = 0  # Desactivar la salida en la consola de Gurobi
        m.setParam('TimeLimit', MaxTime)  # Se lo puse porque a veces sucede que se demora más que el optimal
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        total_time = end_time - start_time + end_time1 - start_time1 + end_time2 - start_time2 + last_time
        last_time = total_time
        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, se reducirá pred_level0 y pred_level1 en ", reduce_level)
            pred_level0 = pred_level0 - reduce_level
            # pred_level1 = pred_level1 + reduce_level
            w += 1

        elif pred_level0 < 0 or pred_level1 > 1:
            print('pred_level0 o pred_level1 alcanzaron valores no adecuados, se ocuparán pred_levels mínimos')
            pred_level0 = max(0, pred_level0)
            pred_level1 = min(1, pred_level1)

        else:
            try:
                # Intentar extraer resultados
                x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_hat_values[(t, j)] < 0.1:
                            y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                            (base_datos_completa['Periodo'] == t) &
                                                            (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                    for t in Periodos for j in Items)
            except:
                pred_level0 = pred_level0 - reduce_level
                # pred_level1 = pred_level1 + reduce_level
                print('| | ***Hubo un error, se actualizarán pred_level0 y pred_level1 = ', pred_level0, pred_level1)
                w += 1

            else:
                infactible = 0  # Marcar como factible y salir del bucle
                print('| | Modelo factible')

    return x_hat_values, s_hat_values, y_hat_values, obj_hat_value, total_time, pred_level0, pred_level1, num_filas_binary0, num_filas_binary1, num_filas_mu, w


# Función modelo óptimo con fixing y tight resuelto con gurobi
def optimal_predict_zeroOrOne(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level, tight_level,
                              reduce_level, zeroOrOne):
    infactible = 1
    last_time = 0
    w = 1
    gurobi_results = {'objective_value': [], 'time': []}

    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        filtro_binary1 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] >= pred_level)

        filtro_binary0 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] < pred_level)

        filtro_mu = (base_datos_completa['Instancia'] == i) & \
                    (base_datos_completa['pred_mu=1'] >= tight_level)

        filtered_df_binary1 = base_datos_completa.copy()
        filtered_df_binary1 = filtered_df_binary1[filtro_binary1]

        filtered_df_binary0 = base_datos_completa.copy()
        filtered_df_binary0 = filtered_df_binary0[filtro_binary0]

        df_mu = base_datos_completa.copy()
        filtered_df_mu = df_mu[filtro_mu]
        filtered_df_mu = filtered_df_mu.drop_duplicates(subset=['Periodo'])

        num_filas_binary1 = len(filtered_df_binary1)
        num_filas_binary0 = len(filtered_df_binary0)

        num_filas_mu = len(filtered_df_mu)

        if num_filas_mu == 0:
            tight_level = tight_level - reduce_level
            print("| | No se fijó ninguna restricción fuerte, se actualiza tight_level=", tight_level)
            continue

        # Fixing time
        start_time1 = time.time()
        # Fijar binaria 1
        if zeroOrOne == 1:
            print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary1} 1s')
            num_filas_binary0 = 0
            for index, row in filtered_df_binary1.iterrows():
                t, j = int(row['Periodo']), int(row['Item'])
                if t != 1:
                    m.addConstr(y[t, j] == 1)

        # Fijar binaria 0
        if zeroOrOne == 0:
            print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary0} 0s')
            num_filas_binary1 = 0
            for index, row in filtered_df_binary0.iterrows():
                t, j = int(row['Periodo']), int(row['Item'])
                if t != 1:
                    m.addConstr(y[t, j] == 0)
        end_time1 = time.time()

        # No tomo el tiempo, ya que estas restricciones 'existían' en el modelo
        # Fijar tight constraints
        t_added = []
        for index, row in filtered_df_mu.iterrows():
            t = int(row['Periodo'])
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t),
                'Capacidad'].iloc[0])
            t_added.append(t)

        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Capacidad relajada
        M = gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                (base_datos_completa['Periodo'] == t) &
                                                (base_datos_completa['Item'] == j),
        'Demanda'].iloc[0] for t in Periodos for j in Items)
        # Relaxing time
        start_time2 = time.time()
        for t in Periodos:
            if t not in t_added:
                # print("ADDING RELAXED CONSTRAINT")
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <= M)

        end_time2 = time.time()

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        end_time2 = time.time()

        # Resolución
        # Definir el callback de Gurobi para almacenar resultados en un diccionario
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:  # Evento de solución incumbente encontrada
                # Obtener el valor de la función objetivo para la incumbente actual
                obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Obtener el tiempo transcurrido
                current_time = model.cbGet(GRB.Callback.RUNTIME)
                # Almacenar el valor de la función objetivo y el tiempo
                gurobi_results['objective_value'].append(obj_value)
                gurobi_results['time'].append(current_time)

        # Resolución
        gurobi_log_file = 'gurobi_optimization.log'
        m.setParam('LogFile', gurobi_log_file)
        m.setParam('TimeLimit', MaxTime)
        start_time = time.time()
        m.optimize(gurobi_callback)
        end_time = time.time()
        total_time = end_time - start_time + end_time1 - start_time1 + end_time2 - start_time2 + last_time
        last_time = total_time
        # Obtener la cantidad de nodos después de la optimización
        node_count = m.getAttr(GRB.Attr.NodeCount)

        clb_obj = gurobi_results['objective_value'][-1]
        clb_time = gurobi_results['time'][-1]

        # Leer el archivo de log de Gurobi para obtener información adicional
        def leer_log_gurobi(log_file):
            cortes_aplicados = {}
            with open(log_file, 'r') as log:
                cortar = False
                for line in log:
                    if 'Cutting planes:' in line:
                        cortar = True
                    elif cortar:
                        if line.strip() == '':
                            break  # Terminar la sección de cortes
                        partes = line.strip().split(':')
                        if len(partes) == 2:
                            corte_tipo = partes[0].strip()
                            cantidad = int(partes[1].strip())
                            cortes_aplicados[corte_tipo] = cantidad
            return cortes_aplicados

        # Leer el archivo de log para obtener los cortes aplicados
        cortes_aplicados = leer_log_gurobi(gurobi_log_file)

        # Calcular la suma total de los cortes aplicados
        cuts_count = sum(cortes_aplicados.values())
        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, se reducirá pred_level en ", reduce_level)
            pred_level = pred_level - reduce_level
            w += 1

        elif pred_level < 0:
            print('pred_level alcanzó un valor negativo, se ocupará pred_level=0')
            pred_level = 0

        else:
            try:
                # Intentar extraer resultados
                x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_hat_values[(t, j)] < 0.1:
                            y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                            (base_datos_completa['Periodo'] == t) &
                                                            (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                    for t in Periodos for j in Items)
            except:
                pred_level = pred_level - reduce_level
                print('| | ***Hubo un error, se actualizará pred_level = ', pred_level)
                w += 1

            else:
                infactible = 0  # Marcar como factible y salir del bucle
                print('| | Modelo factible')

    return x_hat_values, s_hat_values, y_hat_values, obj_hat_value, total_time, pred_level, num_filas_binary0, num_filas_binary1, num_filas_mu, w, node_count, cuts_count, clb_time, clb_obj


# Función modelo óptimo con fixing y tight resuelto con gurobi
def optimal_predict_zeroOrOne_warm(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level, tight_level,
                                   reduce_level, zeroOrOne):
    infactible = 1
    last_time = 0
    w = 1
    gurobi_results = {'objective_value': [], 'time': []}

    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        filtro_binary1 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] >= pred_level)

        filtro_binary0 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] < pred_level)

        filtro_mu = (base_datos_completa['Instancia'] == i) & \
                    (base_datos_completa['pred_mu=1'] >= tight_level)

        filtered_df_binary1 = base_datos_completa.copy()
        filtered_df_binary1 = filtered_df_binary1[filtro_binary1]

        filtered_df_binary0 = base_datos_completa.copy()
        filtered_df_binary0 = filtered_df_binary0[filtro_binary0]

        df_mu = base_datos_completa.copy()
        filtered_df_mu = df_mu[filtro_mu]
        filtered_df_mu = filtered_df_mu.drop_duplicates(subset=['Periodo'])

        num_filas_binary1 = len(filtered_df_binary1)
        num_filas_binary0 = len(filtered_df_binary0)

        num_filas_mu = len(filtered_df_mu)

        if num_filas_mu == 0:
            tight_level = tight_level - reduce_level
            print("| | No se fijó ninguna restricción fuerte, se actualiza tight_level=", tight_level)
            continue

        # Fixing time
        start_time1 = time.time()
        # Fijar binaria 1
        if zeroOrOne == 1:
            print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary1} 1s')
            num_filas_binary0 = 0
            for index, row in filtered_df_binary1.iterrows():
                t, j = int(row['Periodo']), int(row['Item'])
                if t != 1:
                    y[t, j].Start = 1

        # Fijar binaria 0
        if zeroOrOne == 0:
            print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary0} 0s')
            num_filas_binary1 = 0
            for index, row in filtered_df_binary0.iterrows():
                t, j = int(row['Periodo']), int(row['Item'])
                if t != 1:
                    y[t, j].Start = 0
        end_time1 = time.time()

        # No tomo el tiempo, ya que estas restricciones 'existían' en el modelo
        # Fijar tight constraints
        t_added = []
        for index, row in filtered_df_mu.iterrows():
            t = int(row['Periodo'])
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t),
                'Capacidad'].iloc[0])
            t_added.append(t)

        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Capacidad relajada
        M = gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                (base_datos_completa['Periodo'] == t) &
                                                (base_datos_completa['Item'] == j),
        'Demanda'].iloc[0] for t in Periodos for j in Items)
        # Relaxing time
        start_time2 = time.time()
        for t in Periodos:
            if t not in t_added:
                # print("ADDING RELAXED CONSTRAINT")
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <= M)

        end_time2 = time.time()

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        end_time2 = time.time()

        # Resolución
        # Definir el callback de Gurobi para almacenar resultados en un diccionario
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:  # Evento de solución incumbente encontrada
                # Obtener el valor de la función objetivo para la incumbente actual
                obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Obtener el tiempo transcurrido
                current_time = model.cbGet(GRB.Callback.RUNTIME)
                # Almacenar el valor de la función objetivo y el tiempo
                gurobi_results['objective_value'].append(obj_value)
                gurobi_results['time'].append(current_time)

        # Resolución
        gurobi_log_file = 'gurobi_optimization.log'
        m.setParam('LogFile', gurobi_log_file)
        m.setParam('TimeLimit', MaxTime)
        start_time = time.time()
        m.optimize(gurobi_callback)
        end_time = time.time()
        total_time = end_time - start_time + end_time1 - start_time1 + end_time2 - start_time2 + last_time
        last_time = total_time
        # Obtener la cantidad de nodos después de la optimización
        node_count = m.getAttr(GRB.Attr.NodeCount)

        clb_obj = gurobi_results['objective_value'][-1]
        clb_time = gurobi_results['time'][-1]

        # Leer el archivo de log de Gurobi para obtener información adicional
        def leer_log_gurobi(log_file):
            cortes_aplicados = {}
            with open(log_file, 'r') as log:
                cortar = False
                for line in log:
                    if 'Cutting planes:' in line:
                        cortar = True
                    elif cortar:
                        if line.strip() == '':
                            break  # Terminar la sección de cortes
                        partes = line.strip().split(':')
                        if len(partes) == 2:
                            corte_tipo = partes[0].strip()
                            cantidad = int(partes[1].strip())
                            cortes_aplicados[corte_tipo] = cantidad
            return cortes_aplicados

        # Leer el archivo de log para obtener los cortes aplicados
        cortes_aplicados = leer_log_gurobi(gurobi_log_file)

        # Calcular la suma total de los cortes aplicados
        cuts_count = sum(cortes_aplicados.values())
        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, se reducirá pred_level en ", reduce_level)
            pred_level = pred_level - reduce_level
            w += 1

        elif pred_level < 0:
            print('pred_level alcanzó un valor negativo, se ocupará pred_level=0')
            pred_level = 0

        else:
            try:
                # Intentar extraer resultados
                x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_hat_values[(t, j)] < 0.1:
                            y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                            (base_datos_completa['Periodo'] == t) &
                                                            (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                    for t in Periodos for j in Items)
            except:
                pred_level = pred_level - reduce_level
                print('| | ***Hubo un error, se actualizará pred_level = ', pred_level)
                w += 1

            else:
                infactible = 0  # Marcar como factible y salir del bucle
                print('| | Modelo factible')

    return x_hat_values, s_hat_values, y_hat_values, obj_hat_value, total_time, pred_level, num_filas_binary0, num_filas_binary1, num_filas_mu, w, node_count, cuts_count, clb_time, clb_obj


# Función modelo óptimo con fixing y tight resuelto con gurobi
def optimal_predict_zeroOrOne_hint(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level, tight_level,
                                   reduce_level, zeroOrOne):
    infactible = 1
    last_time = 0
    w = 1
    gurobi_results = {'objective_value': [], 'time': []}

    while infactible == 1:
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        filtro_binary1 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] >= pred_level)

        filtro_binary0 = (base_datos_completa['Instancia'] == i) & \
                         (base_datos_completa['pred_binary=1'] < pred_level)

        filtro_mu = (base_datos_completa['Instancia'] == i) & \
                    (base_datos_completa['pred_mu=1'] >= tight_level)

        filtered_df_binary1 = base_datos_completa.copy()
        filtered_df_binary1 = filtered_df_binary1[filtro_binary1]

        filtered_df_binary0 = base_datos_completa.copy()
        filtered_df_binary0 = filtered_df_binary0[filtro_binary0]

        df_mu = base_datos_completa.copy()
        filtered_df_mu = df_mu[filtro_mu]
        filtered_df_mu = filtered_df_mu.drop_duplicates(subset=['Periodo'])

        num_filas_binary1 = len(filtered_df_binary1)
        num_filas_binary0 = len(filtered_df_binary0)

        num_filas_mu = len(filtered_df_mu)

        if num_filas_mu == 0:
            tight_level = tight_level - reduce_level
            print("| | No se fijó ninguna restricción fuerte, se actualiza tight_level=", tight_level)
            continue

        # Fixing time
        start_time1 = time.time()
        # Fijar binaria 1
        if zeroOrOne == 1:
            print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary1} 1s')
            num_filas_binary0 = 0
            for index, row in filtered_df_binary1.iterrows():
                t, j = int(row['Periodo']), int(row['Item'])
                pri = int(row['pred_binary=1'] * 1000)
                val = 1
                if t != 1:
                    y[t, j].varhintval = val
                    y[t, j].varhintpri = pri

        # Fijar binaria 0
        if zeroOrOne == 0:
            print(f'| | TIGHT {num_filas_mu} CONSTRAINTS, FIXING {num_filas_binary0} 0s')
            num_filas_binary1 = 0
            for index, row in filtered_df_binary0.iterrows():
                t, j = int(row['Periodo']), int(row['Item'])
                pri = int(row['pred_binary=0'] * 1000)
                val = 0
                if t != 1:
                    y[t, j].varhintval = val
                    y[t, j].varhintpri = pri
        end_time1 = time.time()

        # No tomo el tiempo, ya que estas restricciones 'existían' en el modelo
        # Fijar tight constraints
        t_added = []
        for index, row in filtered_df_mu.iterrows():
            t = int(row['Periodo'])
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t),
                'Capacidad'].iloc[0])
            t_added.append(t)

        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Capacidad relajada
        M = gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                (base_datos_completa['Periodo'] == t) &
                                                (base_datos_completa['Item'] == j),
        'Demanda'].iloc[0] for t in Periodos for j in Items)
        # Relaxing time
        start_time2 = time.time()
        for t in Periodos:
            if t not in t_added:
                # print("ADDING RELAXED CONSTRAINT")
                m.addConstr(gp.quicksum(x[t, j] for j in Items) <= M)

        end_time2 = time.time()

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        end_time2 = time.time()

        # Resolución
        # Definir el callback de Gurobi para almacenar resultados en un diccionario
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:  # Evento de solución incumbente encontrada
                # Obtener el valor de la función objetivo para la incumbente actual
                obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Obtener el tiempo transcurrido
                current_time = model.cbGet(GRB.Callback.RUNTIME)
                # Almacenar el valor de la función objetivo y el tiempo
                gurobi_results['objective_value'].append(obj_value)
                gurobi_results['time'].append(current_time)

        # Resolución
        gurobi_log_file = 'gurobi_optimization.log'
        m.setParam('LogFile', gurobi_log_file)
        m.setParam('TimeLimit', MaxTime)

        start_time = time.time()
        m.optimize(gurobi_callback)
        end_time = time.time()
        total_time = end_time - start_time + end_time1 - start_time1 + end_time2 - start_time2 + last_time
        last_time = total_time
        # Obtener la cantidad de nodos después de la optimización
        node_count = m.getAttr(GRB.Attr.NodeCount)

        clb_obj = gurobi_results['objective_value'][-1]
        clb_time = gurobi_results['time'][-1]

        # Leer el archivo de log de Gurobi para obtener información adicional
        def leer_log_gurobi(log_file):
            cortes_aplicados = {}
            with open(log_file, 'r') as log:
                cortar = False
                for line in log:
                    if 'Cutting planes:' in line:
                        cortar = True
                    elif cortar:
                        if line.strip() == '':
                            break  # Terminar la sección de cortes
                        partes = line.strip().split(':')
                        if len(partes) == 2:
                            corte_tipo = partes[0].strip()
                            cantidad = int(partes[1].strip())
                            cortes_aplicados[corte_tipo] = cantidad
            return cortes_aplicados

        # Leer el archivo de log para obtener los cortes aplicados
        cortes_aplicados = leer_log_gurobi(gurobi_log_file)

        # Calcular la suma total de los cortes aplicados
        cuts_count = sum(cortes_aplicados.values())
        if m.status == GRB.INF_OR_UNBD:
            print("| | Infactible, se reducirá pred_level en ", reduce_level)
            pred_level = pred_level - reduce_level
            w += 1

        elif pred_level < 0:
            print('pred_level alcanzó un valor negativo, se ocupará pred_level=0')
            pred_level = 0

        else:
            try:
                # Intentar extraer resultados
                x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_hat_values[(t, j)] < 0.1:
                            y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                            (base_datos_completa['Periodo'] == t) &
                                                            (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                    for t in Periodos for j in Items)
            except:
                pred_level = pred_level - reduce_level
                print('| | ***Hubo un error, se actualizará pred_level = ', pred_level)
                w += 1

            else:
                infactible = 0  # Marcar como factible y salir del bucle
                print('| | Modelo factible')

    return x_hat_values, s_hat_values, y_hat_values, obj_hat_value, total_time, pred_level, num_filas_binary0, num_filas_binary1, num_filas_mu, w, node_count, cuts_count, clb_time, clb_obj


def resampling(df, option, variable):
    if option == 2:
        print(df[variable].value_counts())
        return df
    # option = 1 ; UPSAMPLE
    # option = 0 ; DOWNSAMPLE
    # Contar las ocurrencias de cada valor en la columna 'binaria'
    counts = df[variable].value_counts()

    # Determinar cuál grupo es el mayoritario y cuál el minoritario
    if counts[0] > counts[1]:
        df_majority = df[df[variable] == 0]
        df_minority = df[df[variable] == 1]
    else:
        df_majority = df[df[variable] == 1]
        df_minority = df[df[variable] == 0]
    if option == 1:
        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=len(df_majority),  # to match majority class
                                         random_state=123)  # reproducible results

        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])

        # Display new class counts
        print(df_upsampled[variable].value_counts())

        return df_upsampled
    else:
        # Downsample majority class
        df_majority_downsampled = resample(df_majority,
                                           replace=False,  # sample without replacement
                                           n_samples=len(df_minority),  # to match minority class
                                           random_state=123)  # reproducible results

        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])

        # Display new class counts
        print(df_downsampled[variable].value_counts())

        return df_downsampled


def predicting_values(X, X_mu, base_datos_completa, model_mu, model_y, test_items):
    base_datos_completa['pred_mu=1'] = np.repeat(model_mu.predict_proba(X_mu)[:, 1], test_items)
    base_datos_completa['pred_mu=0'] = np.repeat(model_mu.predict_proba(X_mu)[:, 0], test_items)

    base_datos_completa['pred_binary=1'] = model_y.predict_proba(X)[:, 1]
    base_datos_completa['pred_binary=0'] = model_y.predict_proba(X)[:, 0]

    return base_datos_completa

def predicting_values_new(X, base_datos_completa, model_y, test_items):
    base_datos_completa['pred_binary=1'] = model_y.predict_proba(X)[:, 1]
    base_datos_completa['pred_binary=0'] = model_y.predict_proba(X)[:, 0]

    return base_datos_completa

def optimal_predict_threshUP(MaxTime, base_datos_completa, df_num_periodos_items, i, pred_level, zeroOrOne):
    infactible = 1
    last_time = 0
    last_work = 0
    w = 1
    gurobi_results = {'objective_value': [], 'time': []}
    feasible_pred = {}
    firstloop=1
    repair_time = 0  # Tiempo total de la regla de factibilidad
    while infactible == 1:
        #print(not_include)
        # Conjuntos
        Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)
        Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)
        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                             (base_datos_completa['Periodo'] == t) &
                                                             (base_datos_completa['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                   (base_datos_completa['Periodo'] == t) &
                                                                   (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                         (base_datos_completa['Periodo'] == t) &
                                                                         (base_datos_completa['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        num_filas_binary0, num_filas_binary1 = 0, 0
        start_time1 = time.time()
        # Fijar binarias
        for t in Periodos:
            for j in Items:
                prediccion = base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                     (base_datos_completa['Periodo'] == t) &
                                                     (base_datos_completa['Item'] == j), 'pred_binary=1'].iloc[0]
                # Fijar 1s
                if prediccion >= pred_level and 1 in zeroOrOne:
                    if t != 1:
                        m.addConstr(y[t, j] == 1)
                        num_filas_binary1 += 1

                # Fijar 0s
                if prediccion < pred_level and 0 in zeroOrOne:
                    if t != 1:
                        # Caso 1: Si not_include está vacío, agrega todas las variables como 0
                        if not feasible_pred:  # verifica si dict está vacío
                            m.addConstr(y[t, j] == 0)
                            num_filas_binary0 += 1
                        else:
                            # Caso 2: Si not_include tiene valores, sólo agrega restricciones cuando not_include[(t,j)]==0
                            if feasible_pred[(t, j)] == 0:
                                m.addConstr(y[t, j] == 0)
                                num_filas_binary0 += 1
                            if feasible_pred[(t, j)] == 1:
                                m.addConstr(y[t, j] == 1)
                                num_filas_binary0 += 1
        end_time1 = time.time()

        # Restricción  de capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                               (base_datos_completa['Periodo'] == t),
                'Capacidad'].iloc[0])

        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        # Resolución
        # Definir el callback de Gurobi para almacenar resultados en un diccionario
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:  # Evento de solución incumbente encontrada
                # Obtener el valor de la función objetivo para la incumbente actual
                obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Obtener el tiempo transcurrido
                current_time = model.cbGet(GRB.Callback.WORK)
                # Almacenar el valor de la función objetivo y el tiempo
                gurobi_results['objective_value'].append(obj_value)
                gurobi_results['time'].append(current_time)

        if MaxTime - last_work < 0:
            break
        m.setParam('WorkLimit', MaxTime - last_work)  # m.setParam('Cuts', 0)
        m.setParam('OutputFlag', 1)  # Desactiva la salida en consola

        # m.setParam('Heuristics', 0)
        print(f"| | Trying to optimize with {num_filas_binary0} 0s and {num_filas_binary1} 1s")

        start_time = time.time()
        if firstloop==1:
            print("| | Solving first loop")
            m.setParam('SolutionLimit', 1)
            m.optimize()
            print("WORK USED", m.Work)
            if m.status == GRB.SOLUTION_LIMIT:
                #  Entrará aqui solo si para el solution limit no era óptimo
                print("| | FEASIBLE PREDICTIONS")
                last_work = m.Work
                firstloop = 0
                continue
        if firstloop==0:
            m.optimize(gurobi_callback)
            print("WORK USED", m.Work)

        end_time = time.time()
        model_time = end_time1 - start_time1 + last_time
        optimize_time = end_time - start_time
        total_time = model_time + optimize_time
        last_time = total_time
        total_work = m.Work + last_work
        GAP = m.MIPGap
        # Obtener la cantidad de nodos después de la optimización
        node_count = m.getAttr(GRB.Attr.NodeCount)

        try:
            clb_obj = gurobi_results['objective_value'][-1]
            clb_time = gurobi_results['time'][-1]
        except:
            clb_obj = 0
            clb_time = 0

        if m.status == GRB.INF_OR_UNBD:
            print(f"| | Infeasible, updating {num_filas_binary0} 0s ")
            repair_start = time.time()
            feasible_pred, work_levantar = vars_a_levantar(base_datos_completa, df_num_periodos_items, i, pred_level)
            repair_end = time.time()
            repair_time += (repair_end - repair_start)
            last_work = total_work + work_levantar
            print(f"| | It takes {work_levantar}")
            firstloop=0
            w += 1

        else:
            try:
                # Intentar extraer resultados
                x_hat_values = {(t, j): x[t, j].X for t in Periodos for j in Items}
                s_hat_values = {(t, j): s[t, j].X for t in Periodos for j in Items}
                y_hat_values_pre = {(t, j): y[t, j].X for t in Periodos for j in Items}
                y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}
                for t in Periodos:
                    for j in Items:
                        if x_hat_values[(t, j)] < 0.1:
                            y_hat_values[(t, j)] = 0  # Establecer y a 0 si x es menor que 0
                obj_hat_value = sum(base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                            (base_datos_completa['Periodo'] == t) &
                                                            (base_datos_completa['Item'] == j),
                'Costo_Produccion'].iloc[0] * x_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Inventario'].iloc[0] * s_hat_values[(t, j)]
                                    + base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                              (base_datos_completa['Periodo'] == t) &
                                                              (base_datos_completa['Item'] == j),
                'Costo_Setup'].iloc[0] * y_hat_values[(t, j)]
                                    for t in Periodos for j in Items)
                obj_value_pre = m.objVal

            except:
                print(f"| | Infeasible, updating {num_filas_binary0} 0s ")
                repair_start = time.time()
                feasible_pred, work_levantar = vars_a_levantar(base_datos_completa, df_num_periodos_items, i, pred_level)
                repair_end = time.time()
                repair_time += (repair_end - repair_start)
                last_work = total_work + work_levantar
                print(f"| | It takes {work_levantar}")
                w += 1
                firstloop = 0

            else:
                infactible = 0  # Marcar como factible y salir del bucle
                repair_start = time.time()
                feasible_pred, work_levantar = vars_a_levantar(base_datos_completa, df_num_periodos_items, i,
                                                               pred_level)
                repair_end = time.time()
                repair_time += (repair_end - repair_start)
                print(f'| | Solved using scheme. Fixed 0s: {num_filas_binary0} 1s: {num_filas_binary1}')

    return x_hat_values, s_hat_values, y_hat_values_pre, y_hat_values, obj_value_pre, obj_hat_value, model_time, optimize_time, total_time, total_work, GAP, pred_level, num_filas_binary0, num_filas_binary1, w, node_count, clb_time, clb_obj, feasible_pred, repair_time


def vars_a_levantar(base_datos_completa, df_num_periodos_items, i, pred_level):
    Periodos = range(1, df_num_periodos_items.loc[i, 'Cantidad_Periodos'] + 1)

    Items = range(1, df_num_periodos_items.loc[i, 'Numero_items'] + 1)

    # Modelo
    m = gp.Model()

    # Producción
    x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")

    # Stock
    s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
    # Activación
    y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

    levantar = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, name="levantar")
    data = []
    Z = {}
    for t in Periodos:
        for j in Items:
            # Filtrar la base de datos completa para obtener la demanda para la instancia, periodo, y item específicos
            filtrado = base_datos_completa[(base_datos_completa['Instancia'] == i) &
                                           (base_datos_completa['Periodo'] == t) &
                                           (base_datos_completa['Item'] == j)]
            # Si el filtrado no está vacío, guarda el primer valor encontrado, si no, asigna NaN o alguna señal de no disponibilidad
            if not filtrado.empty:
                prediccion = filtrado['pred_binary=1'].iloc[0]
                optimal = filtrado['y'].iloc[0]
                costo = filtrado["Costo_Setup"].iloc[0]
                demanda = filtrado["Demanda"].iloc[0]
            else:
                prediccion = "ERROR"  # O puedes usar None

            # Añade los datos a la lista
            data.append({'Periodo': t, 'Item': j, 'Prediccion': prediccion, 'Optimal': optimal, 'Costo': costo,
                         'Demanda': demanda})
            Z[(t, j)] = prediccion

            # Modifica la función objetivo original integrando penalización
    m.setObjective(gp.quicksum(levantar[t, j] * (1 - Z[(t, j)]) + levantar[t, j] for t in Periodos for j in Items),GRB.MINIMIZE)
    #m.setObjective(gp.quicksum(levantar[t, j] for t in Periodos for j in Items if Z[(t,j)]<pred_level),GRB.MINIMIZE)

    # Inventario
    for t in Periodos:
        for j in Items:
            if t == max(Periodos):
                m.addConstr(s[t, j] == 0)  # Supuesto del libro
            if t == 1:
                m.addConstr(x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                               (base_datos_completa['Periodo'] == t) &
                                                               (base_datos_completa['Item'] == j),
                'Demanda'].iloc[0] + s[t, j])
            else:
                m.addConstr(
                    s[t - 1, j] + x[t, j] == base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                     (base_datos_completa['Periodo'] == t) &
                                                                     (base_datos_completa['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])

    # Capacidad
    for t in Periodos:
        m.addConstr(
            gp.quicksum(x[t, j] for j in Items) <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                                           (base_datos_completa['Periodo'] == t) &
                                                                           (base_datos_completa['Item'] == j),
            'Capacidad'].iloc[0])

    # Relación binaria
    for t in Periodos:
        for j in Items:
            m.addConstr(x[t, j] <= base_datos_completa.loc[(base_datos_completa['Instancia'] == i) &
                                                           (base_datos_completa['Periodo'] == t) &
                                                           (base_datos_completa['Item'] == j),
            'Capacidad'].iloc[0] * y[t, j])

    # Siempre produce el primer período
    for j in Items:
        m.addConstr(y[1, j] == 1)

    for t in Periodos:
        for j in Items:
                if int(Z[(t,j)]>=pred_level)==1:
                    if t != 1:
                        m.addConstr(y[t, j] == 1)
                        m.addConstr(levantar[t,j] == 0)
                else:
                    #m.addConstr(y[t, j] <= 1 + (Z[(t, j)] - 0.5) + levantar[t, j])
                    m.addConstr(y[t, j] <= levantar[t, j])
    print("Numeros sobre umbral ", sum(1 for t in Periodos for j in Items if Z[(t, j)] >= pred_level))
    # Ejecutar la optimización
    m.setParam('OutputFlag', 1)  # Desactiva la salida en consola
    m.setParam('SolutionLimit', 1)
    #m.setParam('WorkLimit', 1)

    m.optimize()

    data = []
    not_include = {}
    for t in Periodos:
        for j in Items:
            # Filtrar la base de datos completa para obtener la demanda para la instancia, periodo, y item específicos
            filtrado = base_datos_completa[(base_datos_completa['Instancia'] == i) &
                                           (base_datos_completa['Periodo'] == t) &
                                           (base_datos_completa['Item'] == j)]
            # Si el filtrado no está vacío, guarda el primer valor encontrado, si no, asigna NaN o alguna señal de no disponibilidad
            if not filtrado.empty:
                prediccion = filtrado['pred_binary=1'].iloc[0]
                optimal = filtrado['y'].iloc[0]
                demanda = filtrado["Demanda"].iloc[0]
                costo1 = filtrado["Costo_Setup"].iloc[0]
                costo2 = filtrado["Costo_Inventario"].iloc[0]
                costo3 = filtrado["Costo_Produccion"].iloc[0]
                demanda = filtrado["Demanda"].iloc[0]
                #x_rn = filtrado["x_relaxed_rootnode"].iloc[0]
                #y_rn = filtrado["y_relaxed_rootnode"].iloc[0]
                #s_rn = filtrado["s_relaxed_rootnode"].iloc[0]
            else:
                prediccion = "ERROR"  # O puedes usar None

            # Añade los datos a la lista
            data.append({'Periodo': t, 'Item': j, 'Prediccion': prediccion, 'Optimal': optimal, 'Costo_setup': costo1,
                         'Costo_produccion': costo2, 'Costo_inventario': costo3,
                         'Demanda': demanda, "corregir": y[t, j].X})
            # Aquellas en que la lista y la prediccion fue 0, las mantiene
            not_include[(t, j)] = int(y[t, j].X)
    # Crea el DataFrame a partir de la lista de diccionarios
    df = pd.DataFrame(data)

    # Guarda el DataFrame a Excel (comentado - archivo de debugging)
    # df.to_excel(f"Comprobanding{i}.xlsx", index=False)  # index=False evita guardar el índice en el Excel

    # Guarda los resultados en Excel para análisis posterior
    return not_include, m.Work


def vars_a_levantar_train(df, pred_level=0.5):
    instancias_unique = df["Instancia"].unique()
    Periodos = range(1, len(df["Periodo"].unique())+1)
    Items = range(1, len(df["Item"].unique())+1)
    acc_total = 0
    acc_total_pre = 0
    for i in instancias_unique:
        aciertos=0
        aciertos_pre_feas=0
        num_predicciones=0
        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")

        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        levantar = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="levantar")
        Z = {}
        for t in Periodos:
            for j in Items:
                prediccion = df.loc[(df['Instancia'] == i) & (df['Periodo'] == t) & (df['Item'] == j),'proba'].iloc[0]
                Z[(t, j)] = prediccion

                # Modifica la función objetivo original integrando penalización
        m.setObjective(gp.quicksum(levantar[t, j] * (1 - Z[(t, j)]) + levantar[t, j] for t in Periodos for j in Items),
                       GRB.MINIMIZE)

        cantidad_supera_umbral = sum(1 for valor in Z.values() if valor >= pred_level)
        cantidad_bajo_umbral = sum(1 for valor in Z.values() if valor < pred_level)
        print("cantidad_supera_umbral ", cantidad_supera_umbral)
        print("cantidad_bajo_umbral", cantidad_bajo_umbral)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == df.loc[(df['Instancia'] == i) & (df['Periodo'] == t) & (df['Item'] == j),'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == df.loc[(df['Instancia'] == i) & (df['Periodo'] == t) & (df['Item'] == j),'Demanda'].iloc[0] + s[t, j])

        # Capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= df.loc[(df['Instancia'] == i) & (df['Periodo'] == t) & (df['Item'] == j),'Capacidad'].iloc[0])

        # Relación binaria
        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= df.loc[(df['Instancia'] == i) & (df['Periodo'] == t) & (df['Item'] == j),'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        for t in Periodos:
            for j in Items:
                m.addConstr(y[t, j] <= 1 + (Z[(t, j)] - 0.5) + levantar[t, j])

        # Ejecutar la optimización
        m.setParam('OutputFlag', 1)  # Desactiva la salida en consola
        m.setParam('WorkLimit', 1)  # Desactiva la salida en consola
        m.optimize()

        feasibility_pred ={}
        data=[]
        for t in Periodos:
            for j in Items:
                # Filtrar la base de datos completa para obtener la demanda para la instancia, periodo, y item específicos
                filtrado = df[(df['Instancia'] == i) &
                                               (df['Periodo'] == t) &
                                               (df['Item'] == j)]
                # Si el filtrado no está vacío, guarda el primer valor encontrado, si no, asigna NaN o alguna señal de no disponibilidad
                if not filtrado.empty:
                    prediccion = filtrado['proba'].iloc[0]
                    optimal = filtrado['binaria'].iloc[0]
                    costo1 = filtrado["Costo_Setup"].iloc[0]
                    costo2 = filtrado["Costo_Inventario"].iloc[0]
                    costo3 = filtrado["Costo_Produccion"].iloc[0]
                    demanda = filtrado["Demanda"].iloc[0]


                else:
                    prediccion = "ERROR"  # O puedes usar None

                # Añade los datos a la lista
                data.append({'Periodo': t, 'Item': j, 'Prediccion': prediccion, 'Optimal': optimal, 'Costo_setup': costo1,
                             'Costo_produccion':costo2, 'Costo_inventario': costo3,
                             'Demanda': demanda, "corregir": y[t, j].X})
                # Aquellas en que la lista y la prediccion fue 0, las mantiene
                feasibility_pred[(t, j)] = int(y[t, j].X)
            # Crea el DataFrame a partir de la lista de diccionarios
        df_1 = pd.DataFrame(data)

        # Guarda el DataFrame a Excel (comentado - archivo de debugging)
        # df_1.to_excel("Comprobanding2.xlsx", index=False)  # index=False evita guardar el índice en el Excel

        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################

        # Modelo
        m = gp.Model()

        # Producción
        x = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="x")
        # Stock
        s = m.addVars(Periodos, Items, vtype=GRB.CONTINUOUS, lb=0, name="s")
        # Activación
        y = m.addVars(Periodos, Items, vtype=GRB.BINARY, name="y")

        # Función Objetivo min costo_prod + costo_inv + costo_setup
        m.setObjective(gp.quicksum(df.loc[(df['Instancia'] == i) &
                                                           (df['Periodo'] == t) &
                                                           (df['Item'] == j),
        'Costo_Produccion'].iloc[0] * x[t, j]
                                   + df.loc[(df['Instancia'] == i) &
                                                             (df['Periodo'] == t) &
                                                             (df['Item'] == j),
        'Costo_Inventario'].iloc[0] * s[t, j]
                                   + df.loc[(df['Instancia'] == i) &
                                                             (df['Periodo'] == t) &
                                                             (df['Item'] == j),
        'Costo_Setup'].iloc[0] * y[t, j]
                                   for t in Periodos for j in Items), GRB.MINIMIZE)

        # Inventario
        for t in Periodos:
            for j in Items:
                if t == max(Periodos):
                    m.addConstr(s[t, j] == 0)  # Supuesto del libro
                if t == 1:
                    m.addConstr(x[t, j] == df.loc[(df['Instancia'] == i) &
                                                                   (df['Periodo'] == t) &
                                                                   (df['Item'] == j),
                    'Demanda'].iloc[0] + s[t, j])
                else:
                    m.addConstr(
                        s[t - 1, j] + x[t, j] == df.loc[(df['Instancia'] == i) &
                                                                         (df['Periodo'] == t) &
                                                                         (df['Item'] == j),
                        'Demanda'].iloc[0] + s[t, j])

        num_filas_binary0, num_filas_binary1 = 0, 0
        # Fijar binarias
        for t in Periodos:
            for j in Items:
                prediccion = df.loc[(df['Instancia'] == i) &
                                                     (df['Periodo'] == t) &
                                                     (df['Item'] == j), 'proba'].iloc[0]
                # Fijar 1s
                if prediccion >= pred_level:
                    if t != 1:
                        m.addConstr(y[t, j] == 1)
                        num_filas_binary1 += 1

                # Fijar 0s
                if prediccion < pred_level:
                    if t != 1:
                        # Caso 1: Si not_include está vacío, agrega todas las variables como 0
                        if not feasibility_pred:  # verifica si dict está vacío
                            m.addConstr(y[t, j] == 0)
                            num_filas_binary0 += 1
                        else:
                            # Caso 2: Si not_include tiene valores, sólo agrega restricciones cuando not_include[(t,j)]==0
                            if feasibility_pred[(t, j)] == 0:
                                m.addConstr(y[t, j] == 0)
                                num_filas_binary0 += 1
                            if feasibility_pred[(t, j)] == 1:
                                m.addConstr(y[t, j] == 1)
                                num_filas_binary0 += 1

        # Restricción  de capacidad
        for t in Periodos:
            m.addConstr(
                gp.quicksum(x[t, j] for j in Items) <= df.loc[(df['Instancia'] == i) &
                                                                               (df['Periodo'] == t),
                'Capacidad'].iloc[0])

        for t in Periodos:
            for j in Items:
                m.addConstr(x[t, j] <= df.loc[(df['Instancia'] == i) &
                                                               (df['Periodo'] == t) &
                                                               (df['Item'] == j),
                'Capacidad'].iloc[0] * y[t, j])

        # Siempre produce el primer período
        for j in Items:
            m.addConstr(y[1, j] == 1)

        m.setParam('WorkLimit', 1)  # m.setParam('Cuts', 0)
        m.setParam('OutputFlag', 1)  # Desactiva la salida en consola

        # m.setParam('Heuristics', 0)
        print(f"| | Trying to optimize with {num_filas_binary0} 0s and {num_filas_binary1} 1s")

        m.optimize()
        print("WORK USED", m.Work)

        y_hat_values = {(t, j): y[t, j].X for t in Periodos for j in Items}

        for (t, j), value in y_hat_values.items():
            valor_optimal = int(df.loc[(df['Instancia'] == i) & (df['Periodo'] == t) & (df['Item'] == j), 'binaria'].iloc[0])
            prediccion = df.loc[(df['Instancia'] == i) &
                                (df['Periodo'] == t) &
                                (df['Item'] == j), 'proba'].iloc[0]
            if t != 1:
                if value == valor_optimal:
                    aciertos += 1

                if int(valor_optimal) == int(prediccion>=pred_level):
                    aciertos_pre_feas +=1
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################
        ###############################################

        for t in Periodos:
            for j in Items:
                prediccion = int(Z[(t,j)]>=pred_level)
                optimal = int(df.loc[(df['Instancia'] == i) & (df['Periodo'] == t) & (df['Item'] == j),'binaria'].iloc[0])
                feasibility_pred = int(y[t,j].X)
                if t!=1:
                    num_predicciones += 1
                    '''
                    if prediccion==1 and optimal==1:
                        aciertos += 1
                    if prediccion==0 and feasibility_pred==0:
                        if optimal==0:
                            aciertos += 1
                    if prediccion==0 and feasibility_pred==1:
                        if optimal == 1:
                            aciertos += 1
                    '''
        print(f'| | Solved using scheme. Fixed 0s: {num_filas_binary0} 1s: {num_filas_binary1}')
        print("Accuracy for instance ", i, " ", aciertos/num_predicciones)
        acc_total += aciertos/num_predicciones
        acc_total_pre += aciertos_pre_feas/num_predicciones
        print("Accuracy acumulado ", acc_total)
    acc_promedio = acc_total/len(instancias_unique)
    acc_promedio_pre_feas = acc_total_pre/len(instancias_unique)
    return acc_promedio, acc_promedio_pre_feas


# TRAIN FUNCTIONS

# from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, ParameterSampler, LeaveOneGroupOut
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, roc_curve, make_scorer
from sklearn.preprocessing import StandardScaler

def SVM_Classification(df, N_iter, cols_features):
    print('--- Training SVM ---')

    # Obtener todas las instancias únicas
    unique_instances = df['Instancia'].unique()
    train_instances, test_instances = train_test_split(unique_instances, test_size=3 / 12, random_state=42)
    # Crear df_train y df_test a partir de las instancias seleccionadas
    df_train = df[df['Instancia'].isin(train_instances)].copy()
    df_test = df[df['Instancia'].isin(test_instances)].copy()

    print(f"Instances {train_instances} will be used to train")
    # --- Configurar GroupKFold usando "Instancia" para que cada fold sea una instancia distinta ---
    glo = LeaveOneGroupOut()

    # --- Bucle de validación cruzada manual usando GroupKFold ---
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [10, 1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly'],  # linear demora mucho
        'degree': [0, 1, 2, 3]  # Aplicable solo si se usa el kernel 'poly'
    }

    random_params = list(ParameterSampler(param_grid, n_iter=N_iter, random_state=42))
    print(random_params)
    # Lista para almacenar los resultados
    resultados = []
    # Bucle sobre cada combinación de hiperparámetros seleccionada aleatoriamente
    for hiperparams in range(len(random_params)):
        scores = []  # Para almacenar los scores de cada fold
        fold = 1
        print(f"\n=== Probando hiperparámetros: {random_params[hiperparams]} ===")
        for train_idx, test_idx in glo.split(
                df_train.drop(columns=["binaria", "Periodo", "Item", "Instancia", "Capacidad"]), df_train['binaria'],
                groups=df_train['Instancia']):
            print(f"\n-- Fold {fold} --")
            df_train_copy = df_train.copy()
            # Extraer datos de entrenamiento y prueba
            train_data = df_train_copy.iloc[train_idx].copy()
            print("USING INSTANCES ", train_data["Instancia"].unique(), " TO TRAIN")
            test_data = df_train_copy.iloc[test_idx].copy()
            print("USING INSTANCES ", test_data["Instancia"].unique(), " TO TEST")

            # Crear el modelo SVC con probability=True y la configuración actual
            model = SVC(probability=True, **random_params[hiperparams])

            scaler = StandardScaler()
            # Ajustar el scaler solo con los datos de entrenamiento
            train_data.loc[:, cols_features] = scaler.fit_transform(train_data.loc[:, cols_features])

            model.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

            # Antes de transformar el test, guardar las columnas que se quieren restaurar
            cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
            #cols_to_restore = list(dict.fromkeys(cols_original + cols_features))
            original_test_values = test_data.loc[:, cols_original].copy()

            # Obtener predict_proba para el conjunto de prueba (probabilidad de la clase 1)
            test_data.loc[:, cols_features] = scaler.transform(test_data.loc[:, cols_features])

            test_data.loc[:, "proba"] = model.predict_proba(test_data.loc[:, cols_features])[:, 1]

            # Restaurar las columnas originales en test_data
            test_data.loc[:, cols_original] = original_test_values

            # Llamar a la función custom de evaluación (por ejemplo, vars_a_levantar_train)
            score, score_pre_feas = vars_a_levantar_train(test_data)
            scores.append(score)

            print(f"Score del fold {fold}: {score:.3f}")
            fold += 1

        # Calcular media y desviación estándar de los scores para esta combinación
        media = np.mean(scores)
        std = np.std(scores)
        resultados.append({'hiperparametros': random_params[hiperparams], 'media': media, 'std': std})
        print(f"\nResultado para hiperparámetros {hiperparams}: media = {media:.3f}, std = {std:.3f}")

    # Convertimos las medias y desviaciones a arreglos NumPy para facilitar los cálculos
    mean_scores = np.array([res['media'] for res in resultados])
    std_scores = np.array([res['std'] for res in resultados])

    # Calculamos el coeficiente de variación para cada combinación: std / media
    # (Nos aseguramos de evitar la división por cero)
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff_variation = np.where(mean_scores != 0, std_scores / mean_scores, np.inf)

    print("Coeficiente de variación:", coeff_variation)

    # Seleccionamos el índice con el menor coeficiente de variación, ignorando los que sean cero o inf
    nonzero_indices = np.where(coeff_variation != 0)[0]
    if nonzero_indices.size > 0:
        best_index = nonzero_indices[np.nanargmin(coeff_variation[nonzero_indices])]
    else:
        best_index = 0

    print("MEJOR INDEX:", best_index)

    # Extraemos los mejores parámetros y sus métricas
    best_params = resultados[best_index]['hiperparametros']
    best_score = resultados[best_index]['media']
    best_std = resultados[best_index]['std']

    # Crear un diccionario con los resultados
    data = {
        'best_params': [str(best_params)],  # Convertimos a string si el tipo es diccionario
        'best_score': [best_score],
        'best_std': [best_std]
    }
    # Convertirlo a un DataFrame
    df_best_params = pd.DataFrame(data)
    # Exportarlo a un archivo Excel
    os.makedirs(PATHS['params'], exist_ok=True)
    output_path = os.path.join(PATHS['params'], f'SVM_{cols_features}_params.xlsx')
    df_best_params.to_excel(output_path, index=False)

    print("Mejores hiperparámetros:", best_params)
    print("Mejor score:", best_score)
    print("Mejor std:", best_std)

    # Entrenamos un modelo final con los mejores hiperparámetros
    SVC2 = SVC(probability=True, **best_params)

    # Escalamos el df_train completo (todas las instancias de entrenamiento)
    final_scaler = StandardScaler()
    train_data = df[df['Instancia'].isin(train_instances)].copy()
    train_data.loc[:, cols_features] = final_scaler.fit_transform(train_data.loc[:, cols_features])

    # Entrenamos el modelo definitivo
    SVC2.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

    # Ahora aplicamos el modelo al conjunto de test (en vez de leer un CSV, usamos test_instances)
    df_test_copy = df_test.copy()
    cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
    #cols_to_restore = list(dict.fromkeys(cols_original + cols_features))

    original_test_values = df_test_copy.loc[:, cols_original].copy()

    # Escalado del test
    df_test_copy.loc[:, cols_features] = final_scaler.transform(df_test_copy.loc[:, cols_features])

    # Predicción de probabilidades
    df_test_copy.loc[:, "proba"] = SVC2.predict_proba(df_test_copy.loc[:, cols_features])[:, 1]

    # Restauramos valores originales
    df_test_copy.loc[:, cols_original] = original_test_values

    # Calculamos score final en el test
    final_score, final_score_pre_feas = vars_a_levantar_train(df_test_copy)

    df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr = 0, 0, 0, 0, 0, 0, 0, 0
    return SVC2, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr, final_scaler


def KNN(df, N_iter, cols_features):
    print('--- Training KNN ---')

    # Obtener todas las instancias únicas
    unique_instances = df['Instancia'].unique()
    train_instances, test_instances = train_test_split(unique_instances, test_size=3 / 12, random_state=42)
    # Crear df_train y df_test a partir de las instancias seleccionadas
    df_train = df[df['Instancia'].isin(train_instances)].copy()
    df_test = df[df['Instancia'].isin(test_instances)].copy()

    print(f"Instances {train_instances} will be used to train")
    # --- Configurar GroupKFold usando "Instancia" para que cada fold sea una instancia distinta ---
    glo = LeaveOneGroupOut()

    # --- Bucle de validación cruzada manual usando GroupKFold ---
    param_grid = {
        'algorithm': ['auto', 'ball_tree'],
        'leaf_size': [20, 30, 40, 50],
        'metric': ['minkowski', 'chebyshev'],
        'n_neighbors': np.arange(1, 50, 2),  # Reducido para una búsqueda más rápida
        'p': [1, 2],  # Reducido inicialmente para simplificar
        'weights': ['uniform', 'distance'],
    }

    random_params = list(ParameterSampler(param_grid, n_iter=N_iter, random_state=42))
    print(random_params)
    # Lista para almacenar los resultados
    resultados = []
    # Bucle sobre cada combinación de hiperparámetros seleccionada aleatoriamente
    for hiperparams in range(len(random_params)):
        scores = []  # Para almacenar los scores de cada fold
        fold = 1
        print(f"\n=== Probando hiperparámetros: {random_params[hiperparams]} ===")
        for train_idx, test_idx in glo.split(
                df_train.drop(columns=["binaria", "Periodo", "Item", "Instancia", "Capacidad"]), df_train['binaria'],
                groups=df_train['Instancia']):
            print(f"\n-- Fold {fold} --")
            df_train_copy = df_train.copy()
            # Extraer datos de entrenamiento y prueba
            train_data = df_train_copy.iloc[train_idx].copy()
            print("USING INSTANCES ", train_data["Instancia"].unique(), " TO TRAIN")
            test_data = df_train_copy.iloc[test_idx].copy()
            print("USING INSTANCES ", test_data["Instancia"].unique(), " TO TEST")

            # Crear el modelo SVC con probability=True y la configuración actual
            model = KNeighborsClassifier(**random_params[hiperparams])

            scaler = StandardScaler()
            # Ajustar el scaler solo con los datos de entrenamiento
            train_data.loc[:, cols_features] = scaler.fit_transform(train_data.loc[:, cols_features])

            model.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

            # Antes de transformar el test, guardar las columnas que se quieren restaurar
            cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
            #cols_to_restore = list(dict.fromkeys(cols_original + cols_features))
            original_test_values = test_data.loc[:, cols_original].copy()

            # Obtener predict_proba para el conjunto de prueba (probabilidad de la clase 1)
            test_data.loc[:, cols_features] = scaler.transform(test_data.loc[:, cols_features])

            test_data.loc[:, "proba"] = model.predict_proba(test_data.loc[:, cols_features])[:, 1]

            # Restaurar las columnas originales en test_data
            test_data.loc[:, cols_original] = original_test_values

            # Llamar a la función custom de evaluación (por ejemplo, vars_a_levantar_train)
            score, score_pre_feas = vars_a_levantar_train(test_data)
            scores.append(score)

            print(f"Score del fold {fold}: {score:.3f}")
            fold += 1

        # Calcular media y desviación estándar de los scores para esta combinación
        media = np.mean(scores)
        std = np.std(scores)
        resultados.append({'hiperparametros': random_params[hiperparams], 'media': media, 'std': std})
        print(f"\nResultado para hiperparámetros {hiperparams}: media = {media:.3f}, std = {std:.3f}")

    # Convertimos las medias y desviaciones a arreglos NumPy para facilitar los cálculos
    mean_scores = np.array([res['media'] for res in resultados])
    std_scores = np.array([res['std'] for res in resultados])

    # Calculamos el coeficiente de variación para cada combinación: std / media
    # (Nos aseguramos de evitar la división por cero)
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff_variation = np.where(mean_scores != 0, std_scores / mean_scores, np.inf)

    print("Coeficiente de variación:", coeff_variation)

    # Seleccionamos el índice con el menor coeficiente de variación, ignorando los que sean cero o inf
    nonzero_indices = np.where(coeff_variation != 0)[0]
    if nonzero_indices.size > 0:
        best_index = nonzero_indices[np.nanargmin(coeff_variation[nonzero_indices])]
    else:
        best_index = 0

    print("MEJOR INDEX:", best_index)

    # Extraemos los mejores parámetros y sus métricas
    best_params = resultados[best_index]['hiperparametros']
    best_score = resultados[best_index]['media']
    best_std = resultados[best_index]['std']

    # Crear un diccionario con los resultados
    data = {
        'best_params': [str(best_params)],  # Convertimos a string si el tipo es diccionario
        'best_score': [best_score],
        'best_std': [best_std]
    }
    # Convertirlo a un DataFrame
    df_best_params = pd.DataFrame(data)
    # Exportarlo a un archivo Excel
    os.makedirs(PATHS['params'], exist_ok=True)
    output_path = os.path.join(PATHS['params'], f'KNN_{cols_features}_params.xlsx')
    df_best_params.to_excel(output_path, index=False)

    print("Mejores hiperparámetros:", best_params)
    print("Mejor score:", best_score)
    print("Mejor std:", best_std)

    # Entrenamos un modelo final con los mejores hiperparámetros
    KNN2 = KNeighborsClassifier(**best_params)

    # Escalamos el df_train completo (todas las instancias de entrenamiento)
    final_scaler = StandardScaler()
    train_data = df[df['Instancia'].isin(train_instances)].copy()
    train_data.loc[:, cols_features] = final_scaler.fit_transform(train_data.loc[:, cols_features])

    # Entrenamos el modelo definitivo
    KNN2.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

    # Ahora aplicamos el modelo al conjunto de test (en vez de leer un CSV, usamos test_instances)
    df_test_copy = df_test.copy()
    cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
    #cols_to_restore = list(dict.fromkeys(cols_original+cols_features))
    original_test_values = df_test_copy.loc[:, cols_original].copy()

    # Escalado del test
    df_test_copy.loc[:, cols_features] = final_scaler.transform(df_test_copy.loc[:, cols_features])

    # Predicción de probabilidades
    df_test_copy.loc[:, "proba"] = KNN2.predict_proba(df_test_copy.loc[:, cols_features])[:, 1]

    # Restauramos valores originales
    df_test_copy.loc[:, cols_original] = original_test_values

    # Calculamos score final en el test
    final_score, final_score_pre_feas = vars_a_levantar_train(df_test_copy)

    df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr = 0, 0, 0, 0, 0, 0, 0, 0
    return KNN2, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr, final_scaler


from sklearn.ensemble import RandomForestClassifier


def RF(df, N_iter, cols_features):
    print('--- Training Random Forest ---')

    # Obtener todas las instancias únicas
    unique_instances = df['Instancia'].unique()
    train_instances, test_instances = train_test_split(unique_instances, test_size=3 / 12, random_state=42)
    # Crear df_train y df_test a partir de las instancias seleccionadas
    df_train = df[df['Instancia'].isin(train_instances)].copy()
    df_test = df[df['Instancia'].isin(test_instances)].copy()

    print(f"Instances {train_instances} will be used to train")
    # --- Configurar GroupKFold usando "Instancia" para que cada fold sea una instancia distinta ---
    glo = LeaveOneGroupOut()

    # --- Bucle de validación cruzada manual usando GroupKFold ---
    param_grid = [{
        'n_estimators': [5, 10, 12],  # Número de árboles en el bosque
        'max_features': [None, 'sqrt', 'log2'],  # Número de características a considerar en cada división
        'max_depth': [2, 3, 5, 10],  # Profundidad máxima del árbol
        'min_samples_split': [3],  # Número mínimo de muestras requeridas para dividir un nodo interno
        'min_samples_leaf': [10, 20, 30, 40],  # Número mínimo de muestras requeridas para estar en un nodo hoja
        'bootstrap': [True, False]  # Metodo para seleccionar muestras para entrenar cada árbol
    }]

    random_params = list(ParameterSampler(param_grid, n_iter=N_iter, random_state=42))
    print(random_params)
    # Lista para almacenar los resultados
    resultados = []
    # Bucle sobre cada combinación de hiperparámetros seleccionada aleatoriamente
    for hiperparams in range(len(random_params)):
        scores = []  # Para almacenar los scores de cada fold
        fold = 1
        print(f"\n=== Probando hiperparámetros: {random_params[hiperparams]} ===")
        for train_idx, test_idx in glo.split(df_train.drop(columns=["binaria", "Periodo", "Item", "Instancia", "Capacidad"]), df_train['binaria'],groups=df_train['Instancia']):
            print(f"\n-- Fold {fold} --")
            df_train_copy = df_train.copy()
            # Extraer datos de entrenamiento y prueba
            train_data = df_train_copy.iloc[train_idx].copy()
            print("USING INSTANCES ", train_data["Instancia"].unique(), " TO TRAIN")
            test_data = df_train_copy.iloc[test_idx].copy()
            print("USING INSTANCES ", test_data["Instancia"].unique(), " TO TEST")

            # Crear el modelo SVC con probability=True y la configuración actual
            model = RandomForestClassifier(random_state=42, **random_params[hiperparams])

            scaler = StandardScaler()
            # Ajustar el scaler solo con los datos de entrenamiento
            train_data.loc[:, cols_features] = scaler.fit_transform(train_data.loc[:, cols_features])

            model.fit(train_data.loc[:,cols_features], train_data.loc[:,"binaria"])

            # Antes de transformar el test, guardar las columnas que se quieren restaurar
            cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
            #cols_to_restore = list(dict.fromkeys(cols_original + cols_features))
            original_test_values = test_data.loc[:,cols_original].copy()

            # Obtener predict_proba para el conjunto de prueba (probabilidad de la clase 1)
            test_data.loc[:, cols_features] = scaler.transform(test_data.loc[:,cols_features])

            test_data.loc[:, "proba"] = model.predict_proba(test_data.loc[:,cols_features])[:, 1]

            # Restaurar las columnas originales en test_data
            test_data.loc[:,cols_original] = original_test_values

            # Llamar a la función custom de evaluación (por ejemplo, vars_a_levantar_train)
            score, score_pre_feas = vars_a_levantar_train(test_data)
            scores.append(score)

            print(f"Score del fold {fold}: {score:.3f}")
            fold += 1

        # Calcular media y desviación estándar de los scores para esta combinación
        media = np.mean(scores)
        std = np.std(scores)
        resultados.append({'hiperparametros': random_params[hiperparams], 'media': media, 'std': std})
        print(f"\nResultado para hiperparámetros {hiperparams}: media = {media:.3f}, std = {std:.3f}")

    # Convertimos las medias y desviaciones a arreglos NumPy para facilitar los cálculos
    mean_scores = np.array([res['media'] for res in resultados])
    std_scores = np.array([res['std'] for res in resultados])

    # Calculamos el coeficiente de variación para cada combinación: std / media
    # (Nos aseguramos de evitar la división por cero)
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff_variation = np.where(mean_scores != 0, std_scores / mean_scores, np.inf)

    print("Coeficiente de variación:", coeff_variation)

    # Seleccionamos el índice con el menor coeficiente de variación, ignorando los que sean cero o inf
    nonzero_indices = np.where(coeff_variation != 0)[0]
    if nonzero_indices.size > 0:
        best_index = nonzero_indices[np.nanargmin(coeff_variation[nonzero_indices])]
    else:
        best_index = 0

    print("MEJOR INDEX:", best_index)

    # Extraemos los mejores parámetros y sus métricas
    best_params = resultados[best_index]['hiperparametros']
    best_score = resultados[best_index]['media']
    best_std = resultados[best_index]['std']

    # Crear un diccionario con los resultados
    data = {
        'best_params': [str(best_params)],  # Convertimos a string si el tipo es diccionario
        'best_score': [best_score],
        'best_std': [best_std]
    }
    # Convertirlo a un DataFrame
    df_best_params = pd.DataFrame(data)
    # Exportarlo a un archivo Excel
    os.makedirs(PATHS['params'], exist_ok=True)
    output_path = os.path.join(PATHS['params'], f'RF_{cols_features}_params.xlsx')
    df_best_params.to_excel(output_path, index=False)

    print("Mejores hiperparámetros:", best_params)
    print("Mejor score:", best_score)
    print("Mejor std:", best_std)

    # Entrenamos un modelo final con los mejores hiperparámetros
    RF2 = RandomForestClassifier(random_state=42, **best_params)

    # Escalamos el df_train completo (todas las instancias de entrenamiento)
    final_scaler = StandardScaler()
    train_data = df[df['Instancia'].isin(train_instances)].copy()
    train_data.loc[:,cols_features] = final_scaler.fit_transform(train_data.loc[:,cols_features])

    # Entrenamos el modelo definitivo
    RF2.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

    # Ahora aplicamos el modelo al conjunto de test (en vez de leer un CSV, usamos test_instances)
    df_test_copy = df_test.copy()
    cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
    original_test_values = df_test_copy.loc[:, cols_original].copy()

    # Escalado del test
    df_test_copy.loc[:, cols_features] = final_scaler.transform(df_test_copy.loc[:, cols_features])

    # Predicción de probabilidades
    df_test_copy.loc[:, "proba"] = RF2.predict_proba(df_test_copy.loc[:, cols_features])[:, 1]

    # Restauramos valores originales
    df_test_copy.loc[:, cols_original] = original_test_values

    # Calculamos score final en el test
    final_score, final_score_pre_feas = vars_a_levantar_train(df_test_copy)

    df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr = 0, 0, 0, 0, 0, 0, 0, 0
    return RF2, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr, final_scaler


from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def XGB(df, N_iter, cols_features):
    print('--- Training XGB ---')

    # Obtener todas las instancias únicas
    unique_instances = df['Instancia'].unique()
    train_instances, test_instances = train_test_split(unique_instances, test_size=3 / 12, random_state=42)
    # Crear df_train y df_test a partir de las instancias seleccionadas
    df_train = df[df['Instancia'].isin(train_instances)].copy()
    df_test = df[df['Instancia'].isin(test_instances)].copy()

    print(f"Instances {train_instances} will be used to train")
    # --- Configurar GroupKFold usando "Instancia" para que cada fold sea una instancia distinta ---
    glo = LeaveOneGroupOut()

    # --- Bucle de validación cruzada manual usando GroupKFold ---
    param_grid = [{
        'n_estimators': [5, 10, 12],
        'max_depth': [2, 3, 5, 10],  # Reducir el número de valores para probar una gama representativa
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Eliminar el valor de 0.3 para evitar el riesgo de sobreajuste
        'subsample': [0.7, 0.8, 0.9, 1.0],  # Mantener el rango, pero reducir un valor
        'colsample_bytree': [0.7, 0.8, 1.0],  # Reducir el número de valores para reducir combinaciones
        'gamma': [0, 0.1, 0.2, 0.3, 0.5],
        # Mantener valores representativos, eliminar valores intermedios para acotar la búsqueda
        'min_child_weight': [1, 5, 10]  # Reducir los valores intermedios para simplificar la búsqueda
    }]

    random_params = list(ParameterSampler(param_grid, n_iter=N_iter, random_state=42))
    print(random_params)
    # Lista para almacenar los resultados
    resultados = []
    # Bucle sobre cada combinación de hiperparámetros seleccionada aleatoriamente
    for hiperparams in range(len(random_params)):
        scores = []  # Para almacenar los scores de cada fold
        fold = 1
        print(f"\n=== Probando hiperparámetros: {random_params[hiperparams]} ===")
        for train_idx, test_idx in glo.split(
                df_train.drop(columns=["binaria", "Periodo", "Item", "Instancia", "Capacidad"]), df_train['binaria'],
                groups=df_train['Instancia']):
            print(f"\n-- Fold {fold} --")
            df_train_copy = df_train.copy()
            # Extraer datos de entrenamiento y prueba
            train_data = df_train_copy.iloc[train_idx].copy()
            print("USING INSTANCES ", train_data["Instancia"].unique(), " TO TRAIN")
            test_data = df_train_copy.iloc[test_idx].copy()
            print("USING INSTANCES ", test_data["Instancia"].unique(), " TO TEST")

            # Crear el modelo SVC con probability=True y la configuración actual
            model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, **random_params[hiperparams])

            scaler = StandardScaler()
            # Ajustar el scaler solo con los datos de entrenamiento
            train_data.loc[:, cols_features] = scaler.fit_transform(train_data.loc[:, cols_features])

            model.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

            # Antes de transformar el test, guardar las columnas que se quieren restaurar
            cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
            original_test_values = test_data.loc[:, cols_original].copy()

            # Obtener predict_proba para el conjunto de prueba (probabilidad de la clase 1)
            test_data.loc[:, cols_features] = scaler.transform(test_data.loc[:, cols_features])

            test_data.loc[:, "proba"] = model.predict_proba(test_data.loc[:, cols_features])[:, 1]

            # Restaurar las columnas originales en test_data
            test_data.loc[:, cols_original] = original_test_values

            # Llamar a la función custom de evaluación (por ejemplo, vars_a_levantar_train)
            score, score_pre_feas = vars_a_levantar_train(test_data)
            scores.append(score)

            print(f"Score del fold {fold}: {score:.3f}")
            fold += 1

        # Calcular media y desviación estándar de los scores para esta combinación
        media = np.mean(scores)
        std = np.std(scores)
        resultados.append({'hiperparametros': random_params[hiperparams], 'media': media, 'std': std})
        print(f"\nResultado para hiperparámetros {hiperparams}: media = {media:.3f}, std = {std:.3f}")

    # Convertimos las medias y desviaciones a arreglos NumPy para facilitar los cálculos
    mean_scores = np.array([res['media'] for res in resultados])
    std_scores = np.array([res['std'] for res in resultados])

    # Calculamos el coeficiente de variación para cada combinación: std / media
    # (Nos aseguramos de evitar la división por cero)
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff_variation = np.where(mean_scores != 0, std_scores / mean_scores, np.inf)

    print("Coeficiente de variación:", coeff_variation)

    # Seleccionamos el índice con el menor coeficiente de variación, ignorando los que sean cero o inf
    nonzero_indices = np.where(coeff_variation != 0)[0]
    if nonzero_indices.size > 0:
        best_index = nonzero_indices[np.nanargmin(coeff_variation[nonzero_indices])]
    else:
        best_index = 0

    print("MEJOR INDEX:", best_index)

    # Extraemos los mejores parámetros y sus métricas
    best_params = resultados[best_index]['hiperparametros']
    best_score = resultados[best_index]['media']
    best_std = resultados[best_index]['std']

    # Crear un diccionario con los resultados
    data = {
        'best_params': [str(best_params)],  # Convertimos a string si el tipo es diccionario
        'best_score': [best_score],
        'best_std': [best_std]
    }
    # Convertirlo a un DataFrame
    df_best_params = pd.DataFrame(data)
    # Exportarlo a un archivo Excel
    os.makedirs(PATHS['params'], exist_ok=True)
    output_path = os.path.join(PATHS['params'], f'XGB_{cols_features}_params.xlsx')
    df_best_params.to_excel(output_path, index=False)

    print("Mejores hiperparámetros:", best_params)
    print("Mejor score:", best_score)
    print("Mejor std:", best_std)

    # Entrenamos un modelo final con los mejores hiperparámetros
    XGB2 = xgb.XGBClassifier(eval_metric='logloss', random_state=42, **best_params)

    # Escalamos el df_train completo (todas las instancias de entrenamiento)
    final_scaler = StandardScaler()
    train_data = df[df['Instancia'].isin(train_instances)].copy()
    train_data.loc[:, cols_features] = final_scaler.fit_transform(train_data.loc[:, cols_features])

    # Entrenamos el modelo definitivo
    XGB2.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

    # Ahora aplicamos el modelo al conjunto de test (en vez de leer un CSV, usamos test_instances)
    df_test_copy = df_test.copy()
    cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
    original_test_values = df_test_copy.loc[:, cols_original].copy()

    # Escalado del test
    df_test_copy.loc[:, cols_features] = final_scaler.transform(df_test_copy.loc[:, cols_features])

    # Predicción de probabilidades
    df_test_copy.loc[:, "proba"] = XGB2.predict_proba(df_test_copy.loc[:, cols_features])[:, 1]

    # Restauramos valores originales
    df_test_copy.loc[:, cols_original] = original_test_values

    # Calculamos score final en el test
    final_score, final_score_pre_feas = vars_a_levantar_train(df_test_copy)

    df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr = 0, 0, 0, 0, 0, 0, 0, 0
    return XGB2, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr, final_scaler


from sklearn.linear_model import LogisticRegression
def LogisticR(X, y, N_iter, metric):
    print('--- Training Logistic Regression ---')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Crear el modelo de regresión logística
    model = LogisticRegression(random_state=42)

    # Parámetros para GridSearchCV
    param_grid = [{
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced'],
        'fit_intercept': [True, False],
        'multi_class': ['auto', 'ovr', 'multinomial'],
        # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        # 'l1_ratio': [0.0, 0.15, 0.5, 0.7, 1.0]
    },
        {'penalty': ['l1'],
         'C': [0.001, 0.01, 0.1, 1, 10, 100],
         'class_weight': [None, 'balanced'],
         'fit_intercept': [True, False],
         'multi_class': ['auto', 'ovr'],
         'solver': ['liblinear']
         # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
         # 'l1_ratio': [0.0, 0.15, 0.5, 0.7, 1.0]
         },
        {'penalty': ['elasticnet'],
         'C': [0.001, 0.01, 0.1, 1, 10, 100],
         'max_iter': [1000],
         'class_weight': [None, 'balanced'],
         'fit_intercept': [True, False],
         'multi_class': ['auto', 'ovr', 'multinomial'],
         'solver': ['saga'],
         'l1_ratio': [0.0, 0.15, 0.5, 0.7, 1.0]
         # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
         # 'l1_ratio': [0.0, 0.15, 0.5, 0.7, 1.0]
         }, {
            'penalty': [None],
            # 'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'fit_intercept': [True, False],
            'multi_class': ['auto', 'ovr', 'multinomial'],
            # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            # 'l1_ratio': [0.0, 0.15, 0.5, 0.7, 1.0]
        }]

    # Definir las funciones personalizadas con average='binary' para precision y recall
    scorers = {
        'precision': make_scorer(precision_score, average='binary', zero_division=0),
        'recall': make_scorer(recall_score, average='binary', zero_division=0),
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average='binary', zero_division=0)
    }

    # Configuración de GridSearchCV
    group_kfold = GroupKFold(n_splits=len(train_instances))
    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=group_kfold.split(X_train, y_train, groups=X.loc[X.index.isin(train_instances), 'Instancia']),
        scoring=scorers[metric],
        verbose=0,
        n_jobs=14,
        random_state=42,
        n_iter=N_iter
    )

    grid_search.fit(X_train, y_train)
    df_gridsearch = pd.DataFrame(grid_search.cv_results_)

    # Accedemos a cv_results_ para obtener las medias y desviaciones estándar de cada combinación de hiperparámetros
    mean_scores = grid_search.cv_results_['mean_test_score']
    std_scores = grid_search.cv_results_['std_test_score']

    # Calculamos el coeficiente de variación para cada combinación de hiperparámetros
    coeff_variation = std_scores / mean_scores

    # Seleccionamos el índice del menor coeficiente de variación
    best_index = np.where(coeff_variation != 0)[0][np.nanargmin(coeff_variation[coeff_variation != 0])]

    # Mostramos el mejor modelo
    best_params = grid_search.cv_results_['params'][best_index]
    best_score = grid_search.cv_results_['mean_test_score'][best_index]
    best_std = grid_search.cv_results_['std_test_score'][best_index]

    print("Mejores hiperparámetros según coeficiente de variación:", best_params)
    print("Mejor puntuación de validación cruzada (accuracy):", best_score)

    LR2 = LogisticRegression(random_state=42, **best_params)
    LR2.fit(X_train, y_train)
    y_pred = LR2.predict(X_test)

    # Calcular las métricas dependiendo del valor de 'metric'
    if metric == 'precision':
        finalscore = precision_score(y_test, y_pred, average='binary', zero_division=0)  # Añadir zero_division=0
    elif metric == 'accuracy':
        finalscore = accuracy_score(y_test, y_pred)  # Accuracy no requiere zero_division
    elif metric == 'recall':
        finalscore = recall_score(y_test, y_pred, average='binary', zero_division=0)  # Añadir zero_division=0
    elif metric == 'f1':
        finalscore = f1_score(y_test, y_pred, average='binary', zero_division=0)  # Calcular la matriz de confusión
    # Calcular la matriz de confusión
    matriz_confusion = confusion_matrix(y_test, y_pred)

    # Imprimir la matriz de confusión
    print("Matriz de confusión:")
    print(matriz_confusion)

    # Imprimir el reporte de clasificación
    print(classification_report(y_test, y_pred, zero_division=0))  # Añadir zero_division aquí también

    # Devolver el mejor modelo entrenado
    return LR2, finalscore, best_score, best_std, df_gridsearch


from sklearn.tree import DecisionTreeClassifier


def DT(df, N_iter, cols_features):
    print('--- Training DT ---')

    # Obtener todas las instancias únicas
    unique_instances = df['Instancia'].unique()
    train_instances, test_instances = train_test_split(unique_instances, test_size=3 / 12, random_state=42)
    # Crear df_train y df_test a partir de las instancias seleccionadas
    df_train = df[df['Instancia'].isin(train_instances)].copy()
    df_test = df[df['Instancia'].isin(test_instances)].copy()

    print(f"Instances {train_instances} will be used to train")
    # --- Configurar GroupKFold usando "Instancia" para que cada fold sea una instancia distinta ---
    glo = LeaveOneGroupOut()

    # --- Bucle de validación cruzada manual usando GroupKFold ---
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [2, 3, 5, 10],  # Reducido para enfocarse en un rango representativo
        'min_samples_split': [3],  # Reducido para disminuir combinaciones innecesarias
        'min_samples_leaf': [10, 20, 30, 40],  # Reducido para simplificar la búsqueda
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2],  # Reducido para evitar valores excesivamente restrictivos
        'max_features': ['sqrt', 'log2', 0.7],
        # Reducido para mantener las opciones más comunes y una opción adicional de fracción
        'max_leaf_nodes': [20, 50, 100],
        # Reducido para limitar el número de combinaciones y enfocarse en un rango adecuado
        'min_impurity_decrease': [0.0, 0.05, 0.1],  # Reducido para mantener los valores más relevantes
        'ccp_alpha': [0.0, 0.001, 0.01, 0.1]  # Reducido para mantener un rango útil y evitar podas excesivas
    }

    random_params = list(ParameterSampler(param_grid, n_iter=N_iter, random_state=42))
    print(random_params)
    # Lista para almacenar los resultados
    resultados = []
    # Bucle sobre cada combinación de hiperparámetros seleccionada aleatoriamente
    for hiperparams in range(len(random_params)):
        scores = []  # Para almacenar los scores de cada fold
        fold = 1
        print(f"\n=== Probando hiperparámetros: {random_params[hiperparams]} ===")
        for train_idx, test_idx in glo.split(
                df_train.drop(columns=["binaria", "Periodo", "Item", "Instancia", "Capacidad"]), df_train['binaria'],
                groups=df_train['Instancia']):
            print(f"\n-- Fold {fold} --")
            df_train_copy = df_train.copy()
            # Extraer datos de entrenamiento y prueba
            train_data = df_train_copy.iloc[train_idx].copy()
            print("USING INSTANCES ", train_data["Instancia"].unique(), " TO TRAIN")
            test_data = df_train_copy.iloc[test_idx].copy()
            print("USING INSTANCES ", test_data["Instancia"].unique(), " TO TEST")

            # Crear el modelo SVC con probability=True y la configuración actual
            model = DecisionTreeClassifier(**random_params[hiperparams])

            scaler = StandardScaler()
            # Ajustar el scaler solo con los datos de entrenamiento
            train_data.loc[:, cols_features] = scaler.fit_transform(train_data.loc[:, cols_features])

            model.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

            # Antes de transformar el test, guardar las columnas que se quieren restaurar
            cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
            original_test_values = test_data.loc[:, cols_original].copy()

            # Obtener predict_proba para el conjunto de prueba (probabilidad de la clase 1)
            test_data.loc[:, cols_features] = scaler.transform(test_data.loc[:, cols_features])

            test_data.loc[:, "proba"] = model.predict_proba(test_data.loc[:, cols_features])[:, 1]

            # Restaurar las columnas originales en test_data
            test_data.loc[:, cols_original] = original_test_values

            # Llamar a la función custom de evaluación (por ejemplo, vars_a_levantar_train)
            score, score_pre_feas = vars_a_levantar_train(test_data)
            scores.append(score)

            print(f"Score del fold {fold}: {score:.3f}")
            fold += 1

        # Calcular media y desviación estándar de los scores para esta combinación
        media = np.mean(scores)
        std = np.std(scores)
        resultados.append({'hiperparametros': random_params[hiperparams], 'media': media, 'std': std})
        print(f"\nResultado para hiperparámetros {hiperparams}: media = {media:.3f}, std = {std:.3f}")

    # Convertimos las medias y desviaciones a arreglos NumPy para facilitar los cálculos
    mean_scores = np.array([res['media'] for res in resultados])
    std_scores = np.array([res['std'] for res in resultados])

    # Calculamos el coeficiente de variación para cada combinación: std / media
    # (Nos aseguramos de evitar la división por cero)
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff_variation = np.where(mean_scores != 0, std_scores / mean_scores, np.inf)

    print("Coeficiente de variación:", coeff_variation)

    # Seleccionamos el índice con el menor coeficiente de variación, ignorando los que sean cero o inf
    nonzero_indices = np.where(coeff_variation != 0)[0]
    if nonzero_indices.size > 0:
        best_index = nonzero_indices[np.nanargmin(coeff_variation[nonzero_indices])]
    else:
        best_index = 0

    print("MEJOR INDEX:", best_index)

    # Extraemos los mejores parámetros y sus métricas
    best_params = resultados[best_index]['hiperparametros']
    best_score = resultados[best_index]['media']
    best_std = resultados[best_index]['std']

    # Crear un diccionario con los resultados
    data = {
        'best_params': [str(best_params)],  # Convertimos a string si el tipo es diccionario
        'best_score': [best_score],
        'best_std': [best_std]
    }
    # Convertirlo a un DataFrame
    df_best_params = pd.DataFrame(data)
    # Exportarlo a un archivo Excel
    os.makedirs(PATHS['params'], exist_ok=True)
    output_path = os.path.join(PATHS['params'], f'DT_{cols_features}_params.xlsx')
    df_best_params.to_excel(output_path, index=False)

    print("Mejores hiperparámetros:", best_params)
    print("Mejor score:", best_score)
    print("Mejor std:", best_std)

    # Entrenamos un modelo final con los mejores hiperparámetros
    DT2 = DecisionTreeClassifier(**best_params)

    # Escalamos el df_train completo (todas las instancias de entrenamiento)
    final_scaler = StandardScaler()
    train_data = df[df['Instancia'].isin(train_instances)].copy()
    train_data.loc[:, cols_features] = final_scaler.fit_transform(train_data.loc[:, cols_features])

    # Entrenamos el modelo definitivo
    DT2.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

    # Ahora aplicamos el modelo al conjunto de test (en vez de leer un CSV, usamos test_instances)
    df_test_copy = df_test.copy()
    cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
    original_test_values = df_test_copy.loc[:, cols_original].copy()

    # Escalado del test
    df_test_copy.loc[:, cols_features] = final_scaler.transform(df_test_copy.loc[:, cols_features])

    # Predicción de probabilidades
    df_test_copy.loc[:, "proba"] = DT2.predict_proba(df_test_copy.loc[:, cols_features])[:, 1]

    # Restauramos valores originales
    df_test_copy.loc[:, cols_original] = original_test_values

    # Calculamos score final en el test
    final_score, final_score_pre_feas = vars_a_levantar_train(df_test_copy)

    df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr = 0, 0, 0, 0, 0, 0, 0, 0
    return DT2, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr, final_scaler

from sklearn.ensemble import GradientBoostingClassifier


def GradientBoosting(df, N_iter, cols_features):
    print('--- Training GB ---')

    # Obtener todas las instancias únicas
    unique_instances = df['Instancia'].unique()
    train_instances, test_instances = train_test_split(unique_instances, test_size=3 / 12, random_state=42)
    # Crear df_train y df_test a partir de las instancias seleccionadas
    df_train = df[df['Instancia'].isin(train_instances)].copy()
    df_test = df[df['Instancia'].isin(test_instances)].copy()

    print(f"Instances {train_instances} will be used to train")
    # --- Configurar GroupKFold usando "Instancia" para que cada fold sea una instancia distinta ---
    glo = LeaveOneGroupOut()

    # --- Bucle de validación cruzada manual usando GroupKFold ---
    param_grid = {
        'n_estimators': [5, 10, 12],
        # Reducir los valores superiores para reducir el costo computacional
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        # Reducir valores muy bajos para limitar el número de iteraciones necesarias
        'max_depth': [2, 3, 5, 10],
        # Reducir la cantidad de valores y enfocarse en un rango adecuado para capturar patrones
        'min_samples_split': [3],
        # Reducir combinaciones sin perder la capacidad de exploración significativa
        'min_samples_leaf': [10, 20, 30, 40],
        # Reducir valores para simplificar el número de combinaciones posibles
        'subsample': [0.7, 0.8, 1.0]
        # Mantener un rango reducido para encontrar un buen balance entre sesgo y varianza
    }

    random_params = list(ParameterSampler(param_grid, n_iter=N_iter, random_state=42))
    print(random_params)
    # Lista para almacenar los resultados
    resultados = []
    # Bucle sobre cada combinación de hiperparámetros seleccionada aleatoriamente
    for hiperparams in range(len(random_params)):
        scores = []  # Para almacenar los scores de cada fold
        fold = 1
        print(f"\n=== Probando hiperparámetros: {random_params[hiperparams]} ===")
        for train_idx, test_idx in glo.split(
                df_train.drop(columns=["binaria", "Periodo", "Item", "Instancia", "Capacidad"]), df_train['binaria'],
                groups=df_train['Instancia']):
            print(f"\n-- Fold {fold} --")
            df_train_copy = df_train.copy()
            # Extraer datos de entrenamiento y prueba
            train_data = df_train_copy.iloc[train_idx].copy()
            print("USING INSTANCES ", train_data["Instancia"].unique(), " TO TRAIN")
            test_data = df_train_copy.iloc[test_idx].copy()
            print("USING INSTANCES ", test_data["Instancia"].unique(), " TO TEST")

            # Crear el modelo SVC con probability=True y la configuración actual
            model = GradientBoostingClassifier(random_state=42, **random_params[hiperparams])

            scaler = StandardScaler()
            # Ajustar el scaler solo con los datos de entrenamiento
            train_data.loc[:, cols_features] = scaler.fit_transform(train_data.loc[:, cols_features])

            model.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

            # Antes de transformar el test, guardar las columnas que se quieren restaurar
            cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
            original_test_values = test_data.loc[:, cols_original].copy()

            # Obtener predict_proba para el conjunto de prueba (probabilidad de la clase 1)
            test_data.loc[:, cols_features] = scaler.transform(test_data.loc[:, cols_features])

            test_data.loc[:, "proba"] = model.predict_proba(test_data.loc[:, cols_features])[:, 1]

            # Restaurar las columnas originales en test_data
            test_data.loc[:, cols_original] = original_test_values

            # Llamar a la función custom de evaluación (por ejemplo, vars_a_levantar_train)
            score, score_pre_feas = vars_a_levantar_train(test_data)
            scores.append(score)

            print(f"Score del fold {fold}: {score:.3f}")
            fold += 1

        # Calcular media y desviación estándar de los scores para esta combinación
        media = np.mean(scores)
        std = np.std(scores)
        resultados.append({'hiperparametros': random_params[hiperparams], 'media': media, 'std': std})
        print(f"\nResultado para hiperparámetros {hiperparams}: media = {media:.3f}, std = {std:.3f}")

    # Convertimos las medias y desviaciones a arreglos NumPy para facilitar los cálculos
    mean_scores = np.array([res['media'] for res in resultados])
    std_scores = np.array([res['std'] for res in resultados])

    # Calculamos el coeficiente de variación para cada combinación: std / media
    # (Nos aseguramos de evitar la división por cero)
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff_variation = np.where(mean_scores != 0, std_scores / mean_scores, np.inf)

    print("Coeficiente de variación:", coeff_variation)

    # Seleccionamos el índice con el menor coeficiente de variación, ignorando los que sean cero o inf
    nonzero_indices = np.where(coeff_variation != 0)[0]
    if nonzero_indices.size > 0:
        best_index = nonzero_indices[np.nanargmin(coeff_variation[nonzero_indices])]
    else:
        best_index = 0

    print("MEJOR INDEX:", best_index)

    # Extraemos los mejores parámetros y sus métricas
    best_params = resultados[best_index]['hiperparametros']
    best_score = resultados[best_index]['media']
    best_std = resultados[best_index]['std']

    print("Mejores hiperparámetros:", best_params)
    print("Mejor score:", best_score)
    print("Mejor std:", best_std)

    # Entrenamos un modelo final con los mejores hiperparámetros
    GB2 = GradientBoostingClassifier(random_state=42, **best_params)

    # Escalamos el df_train completo (todas las instancias de entrenamiento)
    final_scaler = StandardScaler()
    train_data = df[df['Instancia'].isin(train_instances)].copy()
    train_data.loc[:, cols_features] = final_scaler.fit_transform(train_data.loc[:, cols_features])

    # Entrenamos el modelo definitivo
    GB2.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

    # Ahora aplicamos el modelo al conjunto de test (en vez de leer un CSV, usamos test_instances)
    df_test_copy = df_test.copy()
    cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
    original_test_values = df_test_copy.loc[:, cols_original].copy()

    # Escalado del test
    df_test_copy.loc[:, cols_features] = final_scaler.transform(df_test_copy.loc[:, cols_features])

    # Predicción de probabilidades
    df_test_copy.loc[:, "proba"] = GB2.predict_proba(df_test_copy.loc[:, cols_features])[:, 1]

    # Restauramos valores originales
    df_test_copy.loc[:, cols_original] = original_test_values

    # Calculamos score final en el test
    final_score, final_score_pre_feas = vars_a_levantar_train(df_test_copy)

    df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr = 0, 0, 0, 0, 0, 0, 0, 0
    return GB2, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr, final_scaler


from sklearn.naive_bayes import GaussianNB


def NB(df, N_iter, cols_features):
    print('--- Training NB ---')

    # Obtener todas las instancias únicas
    unique_instances = df['Instancia'].unique()
    train_instances, test_instances = train_test_split(unique_instances, test_size=3 / 12, random_state=42)
    # Crear df_train y df_test a partir de las instancias seleccionadas
    df_train = df[df['Instancia'].isin(train_instances)].copy()
    df_test = df[df['Instancia'].isin(test_instances)].copy()

    print(f"Instances {train_instances} will be used to train")
    # --- Configurar GroupKFold usando "Instancia" para que cada fold sea una instancia distinta ---
    glo = LeaveOneGroupOut()

    # --- Bucle de validación cruzada manual usando GroupKFold ---
    param_grid = {
        'var_smoothing': np.logspace(-12, 1, 150)
    }

    random_params = list(ParameterSampler(param_grid, n_iter=N_iter, random_state=42))
    print(random_params)
    # Lista para almacenar los resultados
    resultados = []
    # Bucle sobre cada combinación de hiperparámetros seleccionada aleatoriamente
    for hiperparams in range(len(random_params)):
        scores = []  # Para almacenar los scores de cada fold
        fold = 1
        print(f"\n=== Probando hiperparámetros: {random_params[hiperparams]} ===")
        for train_idx, test_idx in glo.split(
                df_train.drop(columns=["binaria", "Periodo", "Item", "Instancia", "Capacidad"]), df_train['binaria'],
                groups=df_train['Instancia']):
            print(f"\n-- Fold {fold} --")
            df_train_copy = df_train.copy()
            # Extraer datos de entrenamiento y prueba
            train_data = df_train_copy.iloc[train_idx].copy()
            print("USING INSTANCES ", train_data["Instancia"].unique(), " TO TRAIN")
            test_data = df_train_copy.iloc[test_idx].copy()
            print("USING INSTANCES ", test_data["Instancia"].unique(), " TO TEST")

            # Crear el modelo SVC con probability=True y la configuración actual
            model = GaussianNB(**random_params[hiperparams])

            scaler = StandardScaler()
            # Ajustar el scaler solo con los datos de entrenamiento
            train_data.loc[:, cols_features] = scaler.fit_transform(train_data.loc[:, cols_features])

            model.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

            # Antes de transformar el test, guardar las columnas que se quieren restaurar
            cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
            original_test_values = test_data.loc[:, cols_original].copy()

            # Obtener predict_proba para el conjunto de prueba (probabilidad de la clase 1)
            test_data.loc[:, cols_features] = scaler.transform(test_data.loc[:, cols_features])

            test_data.loc[:, "proba"] = model.predict_proba(test_data.loc[:, cols_features])[:, 1]

            # Restaurar las columnas originales en test_data
            test_data.loc[:, cols_original] = original_test_values

            # Llamar a la función custom de evaluación (por ejemplo, vars_a_levantar_train)
            score, score_pre_feas = vars_a_levantar_train(test_data)
            scores.append(score)

            print(f"Score del fold {fold}: {score:.3f}")
            fold += 1

        # Calcular media y desviación estándar de los scores para esta combinación
        media = np.mean(scores)
        std = np.std(scores)
        resultados.append({'hiperparametros': random_params[hiperparams], 'media': media, 'std': std})
        print(f"\nResultado para hiperparámetros {hiperparams}: media = {media:.3f}, std = {std:.3f}")

    # Convertimos las medias y desviaciones a arreglos NumPy para facilitar los cálculos
    mean_scores = np.array([res['media'] for res in resultados])
    std_scores = np.array([res['std'] for res in resultados])

    # Calculamos el coeficiente de variación para cada combinación: std / media
    # (Nos aseguramos de evitar la división por cero)
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff_variation = np.where(mean_scores != 0, std_scores / mean_scores, np.inf)

    print("Coeficiente de variación:", coeff_variation)

    # Seleccionamos el índice con el menor coeficiente de variación, ignorando los que sean cero o inf
    nonzero_indices = np.where(coeff_variation != 0)[0]
    if nonzero_indices.size > 0:
        best_index = nonzero_indices[np.nanargmin(coeff_variation[nonzero_indices])]
    else:
        best_index = 0

    print("MEJOR INDEX:", best_index)

    # Extraemos los mejores parámetros y sus métricas
    best_params = resultados[best_index]['hiperparametros']
    best_score = resultados[best_index]['media']
    best_std = resultados[best_index]['std']

    # Crear un diccionario con los resultados
    data = {
        'best_params': [str(best_params)],  # Convertimos a string si el tipo es diccionario
        'best_score': [best_score],
        'best_std': [best_std]
    }
    # Convertirlo a un DataFrame
    df_best_params = pd.DataFrame(data)
    # Exportarlo a un archivo Excel
    os.makedirs(PATHS['params'], exist_ok=True)
    output_path = os.path.join(PATHS['params'], f'NB_{cols_features}_params.xlsx')
    df_best_params.to_excel(output_path, index=False)

    print("Mejores hiperparámetros:", best_params)
    print("Mejor score:", best_score)
    print("Mejor std:", best_std)

    # Entrenamos un modelo final con los mejores hiperparámetros
    NB2 = GaussianNB(**best_params)

    # Escalamos el df_train completo (todas las instancias de entrenamiento)
    final_scaler = StandardScaler()
    train_data = df[df['Instancia'].isin(train_instances)].copy()
    train_data.loc[:, cols_features] = final_scaler.fit_transform(train_data.loc[:, cols_features])

    # Entrenamos el modelo definitivo
    NB2.fit(train_data.loc[:, cols_features], train_data.loc[:, "binaria"])

    # Ahora aplicamos el modelo al conjunto de test (en vez de leer un CSV, usamos test_instances)
    df_test_copy = df_test.copy()
    cols_original = ['Costo_Produccion', 'Costo_Inventario', 'Demanda', 'Costo_Setup', 'Capacidad']
    original_test_values = df_test_copy.loc[:, cols_original].copy()

    # Escalado del test
    df_test_copy.loc[:, cols_features] = final_scaler.transform(df_test_copy.loc[:, cols_features])

    # Predicción de probabilidades
    df_test_copy.loc[:, "proba"] = NB2.predict_proba(df_test_copy.loc[:, cols_features])[:, 1]

    # Restauramos valores originales
    df_test_copy.loc[:, cols_original] = original_test_values

    # Calculamos score final en el test
    final_score, final_score_pre_feas = vars_a_levantar_train(df_test_copy)

    df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr = 0, 0, 0, 0, 0, 0, 0, 0
    return NB2, final_score, final_score_pre_feas, best_score, best_std, df_gridsearch, auc_score, max_youden, optimal_threshold, max_youden_fpr, optimal_threshold_fpr, max_youden_fnr, optimal_threshold_fnr, final_scaler


def graficar_datos_por_instancia(datos_grafico):
    # Convertir el diccionario en un DataFrame
    df_datos = pd.DataFrame(datos_grafico)

    # Obtener la lista de instancias únicas
    instancias = df_datos['Instancia'].unique()

    # Crear la carpeta 'Graphs' si no existe
    if not os.path.exists('Graphs'):
        os.makedirs('Graphs')

    # Crear un gráfico por instancia
    for instancia in instancias:
        df_instancia = df_datos[df_datos['Instancia'] == instancia]

        # Crear subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Graficar Pred_level vs Tiempo y añadir etiquetas
        ax1.plot(df_instancia['Pred_level'], df_instancia['Tiempo'], color='blue', marker='o')
        for index, row in df_instancia.iterrows():
            ax1.annotate(f"inicial= {row['Pred_level_inicial']}", (row['Pred_level'], row['Tiempo']), xytext=(5, -5),
                         textcoords='offset points')
        ax1.set_title(f'Pred_level vs Tiempo - Instancia {instancia}')
        ax1.set_xlabel('Pred_level')
        ax1.set_ylabel('Tiempo')

        # Establecer los ticks en el eje X
        ax1.set_xticks(np.arange(df_instancia['Pred_level'].min(), df_instancia['Pred_level'].max() + 0.05, 0.05))

        # Graficar GAP vs Tiempo
        ax2.plot(df_instancia['GAP'], df_instancia['Tiempo'], color='red', marker='s')
        for index, row in df_instancia.iterrows():
            ax2.annotate(f"inicial= {row['Pred_level_inicial']}", (row['GAP'], row['Tiempo']), xytext=(5, -5),
                         textcoords='offset points')
        ax2.set_title(f'GAP vs Tiempo - Instancia {instancia}')
        ax2.set_xlabel('GAP')
        ax2.set_ylabel('Tiempo')

        # Guardar el gráfico en la carpeta 'Graphs'
        plt.tight_layout()
        nombre_grafico = f'Graphs/Instancia_{instancia}_grafico.png'
        plt.savefig(nombre_grafico)
        plt.close()  # Cerrar la figura para liberar memoria

    print("Los gráficos se han guardado en la carpeta 'Graphs'.")


def graficar_datos_por_instancia_con_fixing(datos_grafico):
    # Convertir el diccionario en un DataFrame
    df_datos = pd.DataFrame(datos_grafico)

    # Obtener la lista de instancias únicas
    instancias = df_datos['Instancia'].unique()

    # Crear la carpeta 'Graphs' si no existe
    if not os.path.exists('Graphs'):
        os.makedirs('Graphs')

    # Crear un gráfico por instancia y por valor de zeroOrOne
    for instancia in instancias:
        for zeroOrOne in df_datos['zeroOrOne'].unique():
            # Filtrar datos por instancia y zeroOrOne
            df_instancia = df_datos[(df_datos['Instancia'] == instancia) & (df_datos['zeroOrOne'] == zeroOrOne)]

            # Crear subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Graficar Pred_level vs Tiempo y añadir etiquetas
            ax1.plot(df_instancia['Pred_level'], df_instancia['Tiempo'], color='blue', marker='o')
            for index, row in df_instancia.iterrows():
                ax1.annotate(f"inicial= {row['Pred_level_inicial']}", (row['Pred_level'], row['Tiempo']),
                             xytext=(5, -5), textcoords='offset points')
            ax1.set_title(f'Pred_level vs Tiempo - Instancia {instancia}')
            ax1.set_xlabel('Pred_level')
            ax1.set_ylabel('Tiempo')
            # Establecer los ticks en el eje X
            ax1.set_xticks(np.arange(df_instancia['Pred_level'].min(), df_instancia['Pred_level'].max() + 0.05, 0.05))

            # Graficar GAP vs Tiempo
            ax2.plot(df_instancia['GAP'], df_instancia['Tiempo'], color='red', marker='s')
            for index, row in df_instancia.iterrows():
                ax2.annotate(f"inicial= {row['Pred_level_inicial']}", (row['GAP'], row['Tiempo']), xytext=(5, -5),
                             textcoords='offset points')
            ax2.set_title(f'GAP vs Tiempo - Instancia {instancia}')
            ax2.set_xlabel('GAP')
            ax2.set_ylabel('Tiempo')

            # Guardar el gráfico en la carpeta 'Graphs'
            plt.tight_layout()
            nombre_grafico = f'Graphs/Instancia_{instancia}_grafico_fixing_{zeroOrOne}.png'
            plt.savefig(nombre_grafico)
            plt.close()  # Cerrar la figura para liberar memoria

    print("Los gráficos se han guardado en la carpeta 'Graphs'.")


def generate_features(df):
    # Crear un nuevo DataFrame para almacenar las características generadas
    df_features = pd.DataFrame()

    # Obtener la lista de columnas
    columns = df.columns

    # Realizar operaciones entre columnas
    for i in range(len(columns)):
        col1 = columns[i]

        # Operaciones unarias sobre una sola columna
        df_features[f'{col1}_sqrt'] = np.where(df[col1] >= 0, np.sqrt(df[col1]), np.nan)
        df_features[f'{col1}_log'] = np.where(df[col1] > 0, np.log(df[col1]), np.nan)
        df_features[f'{col1}_inv'] = np.where(df[col1] != 0, 1 / df[col1], np.nan)

        for j in range(i + 1, len(columns)):  # Cambiado para evitar operaciones redundantes como col1 + col1
            col2 = columns[j]

            # Operaciones binarias entre dos columnas diferentes
            df_features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            df_features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
            df_features[f'{col1}_times_{col2}'] = df[col1] * df[col2]
            df_features[f'{col1}_div_{col2}'] = np.where(df[col2] != 0, df[col1] / df[col2], np.nan)
            df_features[f'{col1}_mod_{col2}'] = np.where(df[col2] != 0, df[col1] % df[col2], np.nan)
            df_features[f'{col1}_pow_{col2}'] = np.where((df[col1] >= 0) | (df[col2] % 1 == 0), df[col1] ** df[col2],
                                                         np.nan)
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features = df_features.dropna(axis=1, how='any')  # Limpiar columnas con NaN

    # Concatenar el DataFrame original con las nuevas características generadas
    df_combined = pd.concat([df, df_features], axis=1)

    # Eliminar columnas idénticas
    df_combined = df_combined.T.drop_duplicates().T
    return df_combined


from scipy.stats import spearmanr
import seaborn as sb


def correlation_heatmap(df, target_col):
    # Copiar el DataFrame para no modificar el original
    df_copy = df.copy()

    # Calcular la correlación de Spearman entre la columna 'y' y las demás columnas
    spearman_corr = {}
    for col in df_copy.columns:
        if col != target_col:
            spearman_corr[col] = spearmanr(df_copy[col], df_copy[target_col])[0]
    spearman_corr_df = pd.DataFrame(spearman_corr, index=['Spearman Corr']).T

    # Calcular la correlación de Pearson para las demás columnas
    pearson_corr_df = df_copy.drop(columns=[target_col]).corr()

    # Añadir la columna 'y' al DataFrame de correlaciones de Pearson
    for col in spearman_corr_df.index:
        pearson_corr_df.loc[target_col, col] = spearman_corr_df.loc[col, 'Spearman Corr']
        pearson_corr_df.loc[col, target_col] = spearman_corr_df.loc[col, 'Spearman Corr']

    # Crear el mapa de calor
    plt.figure(figsize=(12, 8))
    sb.heatmap(pearson_corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()


# Función para obtener el valor más alto y su índice para cada columna
def max_values(df_test, df_train, n=10, top_n_global=3):
    global_top_values_mu = []
    global_top_values_y = []

    # Obtener y mostrar los top n valores por columna
    for column in df_test.columns:
        if 'Accuracy_mu' in column:
            if column in df_train.columns:
                top_n = df_test[column].nlargest(n)
                values_indices = list(top_n.index)
                print(f"Top {len(values_indices)} accuracies for column '{column}':")
                for idx in values_indices:
                    if idx in df_train.index:
                        value = df_test.at[idx, column]
                        train_value = df_train.at[idx, column]
                        difference = abs(value - train_value)
                        global_top_values_mu.append((value, idx, column, difference))
                        print(
                            f"Index: {idx}, Test Accuracy: {value}, Train Accuracy: {train_value}, Difference: {difference}")
                    else:
                        print(f"Warning: Index {idx} not found in df_train for column '{column}'.")
                print("-" * 40)
            else:
                print(f"Warning: Column '{column}' not found in df_train.")
        elif 'Accuracy_y' in column:
            if column in df_train.columns:
                top_n = df_test[column].nlargest(n)
                values_indices = list(top_n.index)
                print(f"Top {len(values_indices)} accuracies for column '{column}':")
                for idx in values_indices:
                    if idx in df_train.index:
                        value = df_test.at[idx, column]
                        train_value = df_train.at[idx, column]
                        difference = abs(value - train_value)
                        global_top_values_y.append((value, idx, column, difference))
                        print(
                            f"Index: {idx}, Test Accuracy: {value}, Train Accuracy: {train_value}, Difference: {difference}")
                    else:
                        print(f"Warning: Index {idx} not found in df_train for column '{column}'.")
                print("-" * 40)
            else:
                print(f"Warning: Column '{column}' not found in df_train.")

    # Ordenar por test accuracy y luego por la menor diferencia para 'Accuracy_mu'
    global_top_values_mu.sort(key=lambda x: (x[0], -x[3]), reverse=True)
    global_top_values_mu = global_top_values_mu[:top_n_global]

    print(f"Top {top_n_global} values for 'Accuracy_mu' considering minimal difference:")
    for value, idx, column, difference in global_top_values_mu:
        print(
            f"Test Accuracy: {value}, Train Accuracy: {df_train.at[idx, column]}, Difference: {difference}, Index: {idx}, Column: {column}")
    print("=" * 40)

    # Ordenar por test accuracy y luego por la menor diferencia para 'Accuracy_y'
    global_top_values_y.sort(key=lambda x: (x[0], -x[3]), reverse=True)
    global_top_values_y = global_top_values_y[:top_n_global]

    print(f"Top {top_n_global} accuracies for 'Accuracy_y' considering minimal difference:")
    for value, idx, column, difference in global_top_values_y:
        print(
            f"Test Accuracy: {value}, Train Accuracy: {df_train.at[idx, column]}, Difference: {difference}, Index: {idx}, Column: {column}")
    print("=" * 40)
