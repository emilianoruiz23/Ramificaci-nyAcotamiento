import streamlit as st
import numpy as np
import math
from scipy.optimize import linprog
import graphviz

st.set_page_config(layout="wide", page_title="Método de Ramificación y Acotamiento")

class Node:
    def __init__(self, id, bounds, parent_id=None, branch_desc="Raíz"):
        self.id = id
        self.bounds = bounds # Lista de tuplas (min, max) para cada variable
        self.parent_id = parent_id
        self.branch_desc = branch_desc
        self.z = None
        self.x = None
        self.status = "No resuelto" # Puede ser: Factible, Infactible, Agotado, Solución Entera

def is_integer(val, tol=1e-5):
    return abs(val - round(val)) < tol

def solve_relaxation(c, A_ub, b_ub, bounds, is_max):
    # linprog minimiza por defecto. Si es max, invertimos c
    c_opt = [-x for x in c] if is_max else c
    res = linprog(c_opt, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        z = -res.fun if is_max else res.fun
        return True, z, res.x
    return False, None, None

def branch_and_bound(c, A_ub, b_ub, var_types, is_max):
    num_vars = len(c)
    
    # Inicializar límites: continuas/enteras [0, inf], binarias [0, 1]
    initial_bounds = []
    for t in var_types:
        if t == 'Binaria (B)':
            initial_bounds.append((0, 1))
        else:
            initial_bounds.append((0, None))
            
    root = Node(id=0, bounds=initial_bounds)
    nodes = [root]
    active_nodes = [root]
    
    best_z = -float('inf') if is_max else float('inf')
    best_x = None
    node_counter = 0
    
    while active_nodes:
        # Seleccionar el nodo (Exploración en profundidad / LIFO para simplicidad)
        current = active_nodes.pop()
        
        # 1. Resolver el modelo relajado
        success, z, x = solve_relaxation(c, A_ub, b_ub, current.bounds, is_max)
        
        if not success:
            current.status = "Infactible"
            continue
            
        current.z = z
        current.x = np.round(x, 5) # Redondear para evitar errores de coma flotante
        
        # 4. Analizar la cota (Problema agotado)
        if (is_max and z <= best_z) or (not is_max and z >= best_z):
            current.status = "Agotado (Por Cota)"
            continue
            
        # Verificar si la solución cumple con las condiciones de integridad
        is_feasible_integer = True
        branch_var_idx = -1
        branch_val = None
        
        for i in range(num_vars):
            if var_types[i] in ['Entera (I)', 'Binaria (B)']:
                if not is_integer(current.x[i]):
                    is_feasible_integer = False
                    branch_var_idx = i
                    branch_val = current.x[i]
                    break # Seleccionamos la primera que incumpla (Regla arbitraria como dice la teoría)
                    
        if is_feasible_integer:
            current.status = "Solución Entera"
            best_z = z
            best_x = current.x
        else:
            current.status = "Ramificado"
            # 3. Crear subproblemas
            node_counter += 1
            left_bounds = current.bounds.copy()
            right_bounds = current.bounds.copy()
            
            if var_types[branch_var_idx] == 'Binaria (B)':
                # Modelo Binario: x=0 y x=1
                left_bounds[branch_var_idx] = (0, 0)
                right_bounds[branch_var_idx] = (1, 1)
                desc_l = f"x{branch_var_idx+1} = 0"
                desc_r = f"x{branch_var_idx+1} = 1"
            else:
                # Modelo Entero: x <= piso, x >= piso + 1
                floor_val = math.floor(branch_val)
                left_bounds[branch_var_idx] = (left_bounds[branch_var_idx][0], floor_val)
                right_bounds[branch_var_idx] = (floor_val + 1, right_bounds[branch_var_idx][1])
                desc_l = f"x{branch_var_idx+1} <= {floor_val}"
                desc_r = f"x{branch_var_idx+1} >= {floor_val + 1}"
            
            node_left = Node(id=node_counter, bounds=left_bounds, parent_id=current.id, branch_desc=desc_l)
            node_counter += 1
            node_right = Node(id=node_counter, bounds=right_bounds, parent_id=current.id, branch_desc=desc_r)
            
            nodes.extend([node_left, node_right])
            # Se añaden a los nodos activos
            active_nodes.extend([node_left, node_right])

    return nodes, best_z, best_x

# --- INTERFAZ STREAMLIT ---
st.title("🌳 Método de Ramificación y Acotamiento")
st.markdown("Basado en modelos Enteros Puros, Mixtos y Binarios.")

col_config, col_main = st.columns([1, 2])

with col_config:
    st.header("Configuración del Modelo")
    opt_type = st.selectbox("Objetivo:", ["Maximizar", "Minimizar"])
    is_max = opt_type == "Maximizar"
    
    num_vars = st.number_input("Número de variables (Máx 10)", min_value=1, max_value=10, value=2)
    num_cons = st.number_input("Número de restricciones", min_value=1, max_value=10, value=2)
    
    st.subheader("Función Objetivo (Z)")
    c = []
    cols_c = st.columns(num_vars)
    for i in range(num_vars):
        val = cols_c[i].number_input(f"x{i+1}", value=1.0, key=f"c_{i}")
        c.append(val)
        
    st.subheader("Tipos de Variables")
    var_types = []
    cols_t = st.columns(num_vars)
    for i in range(num_vars):
        t = cols_t[i].selectbox(f"Tipo x{i+1}", ["Entera (I)", "Continua (C)", "Binaria (B)"], key=f"t_{i}")
        var_types.append(t)
        
    st.subheader("Restricciones (<=)")
    A_ub = []
    b_ub = []
    for i in range(num_cons):
        st.write(f"Restricción {i+1}")
        cols_r = st.columns(num_vars + 1)
        row = []
        for j in range(num_vars):
            val = cols_r[j].number_input(f"x{j+1}", value=1.0, key=f"a_{i}_{j}")
            row.append(val)
        A_ub.append(row)
        b = cols_r[-1].number_input("RHS", value=10.0, key=f"b_{i}")
        b_ub.append(b)

with col_main:
    if st.button("🚀 Resolver Modelo", type="primary"):
        with st.spinner("Construyendo el árbol..."):
            nodes, best_z, best_x = branch_and_bound(c, A_ub, b_ub, var_types, is_max)
            
            # Crear gráfico
            dot = graphviz.Digraph(format='png')
            dot.attr(rankdir='TB', size='8,8')
            
            for n in nodes:
                # Determinar color según estado
                color = "lightblue"
                if n.status == "Infactible": color = "lightcoral"
                elif n.status == "Agotado (Por Cota)": color = "lightgrey"
                elif n.status == "Solución Entera": color = "lightgreen"
                
                label = f"Nodo {n.id}\n"
                if n.z is not None:
                    label += f"Z = {n.z:.2f}\n"
                    # Mostrar variables
                    x_str = ", ".join([f"x{i+1}={n.x[i]:.2f}" for i in range(len(n.x))])
                    label += f"({x_str})\n"
                label += f"Estado: {n.status}"
                
                dot.node(str(n.id), label, style="filled", fillcolor=color, shape="box")
                
                if n.parent_id is not None:
                    dot.edge(str(n.parent_id), str(n.id), label=n.branch_desc)
            
            st.success("¡Modelo resuelto!")
            st.write("### Solución Óptima Encontrada")
            if best_x is not None:
                st.write(f"**Z:** {best_z:.4f}")
                st.write("**Variables:**", {f"x{i+1}": best_x[i] for i in range(num_vars)})
            else:
                st.error("No se encontró ninguna solución factible entera/binaria.")
                
            st.write("### Árbol de Ramificación y Acotamiento")
            st.graphviz_chart(dot)