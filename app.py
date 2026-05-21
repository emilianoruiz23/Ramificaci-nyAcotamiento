import streamlit as st
import numpy as np
import math
from scipy.optimize import linprog
import graphviz

# Configuración de la página
st.set_page_config(layout="wide", page_title="Método de Ramificación y Acotamiento")

# --- CLASES Y FUNCIONES DEL ALGORITMO ---

class Node:
    def __init__(self, id, bounds, parent_id=None, branch_desc="Raíz"):
        self.id = id
        self.bounds = bounds
        self.parent_id = parent_id
        self.branch_desc = branch_desc
        self.z = None
        self.x = None
        self.status = "No resuelto"

def is_integer(val, tol=1e-5):
    return abs(val - round(val)) < tol

def solve_relaxation(c, A, b, senses, bounds, is_max):
    c_opt = [-x for x in c] if is_max else c
    
    A_ub, b_ub = [], []
    A_eq, b_eq = [], []
    
    # Procesar tipos de restricciones
    for i in range(len(senses)):
        if senses[i] == "<=":
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif senses[i] == ">=":
            A_ub.append([-val for val in A[i]]) # Multiplicar por -1 para linprog
            b_ub.append(-b[i])
        elif senses[i] == "==":
            A_eq.append(A[i])
            b_eq.append(b[i])
            
    # Manejar listas vacías para scipy
    A_ub = A_ub if len(A_ub) > 0 else None
    b_ub = b_ub if len(b_ub) > 0 else None
    A_eq = A_eq if len(A_eq) > 0 else None
    b_eq = b_eq if len(b_eq) > 0 else None

    res = linprog(c_opt, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        z = -res.fun if is_max else res.fun
        return True, z, res.x
    return False, None, None

def branch_and_bound(c, A, b, senses, var_types, is_max):
    num_vars = len(c)
    
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
        current = active_nodes.pop()
        
        success, z, x = solve_relaxation(c, A, b, senses, current.bounds, is_max)
        
        if not success:
            current.status = "Infactible"
            continue
            
        current.z = z
        current.x = np.round(x, 5)
        
        if (is_max and z <= best_z) or (not is_max and z >= best_z):
            current.status = "Agotado (Por Cota)"
            continue
            
        is_feasible_integer = True
        branch_var_idx = -1
        branch_val = None
        
        for i in range(num_vars):
            if var_types[i] in ['Entera (I)', 'Binaria (B)']:
                if not is_integer(current.x[i]):
                    is_feasible_integer = False
                    branch_var_idx = i
                    branch_val = current.x[i]
                    break
                    
        if is_feasible_integer:
            current.status = "Solución Entera"
            best_z = z
            best_x = current.x
        else:
            current.status = "Ramificado"
            node_counter += 1
            left_bounds = current.bounds.copy()
            right_bounds = current.bounds.copy()
            
            if var_types[branch_var_idx] == 'Binaria (B)':
                left_bounds[branch_var_idx] = (0, 0)
                right_bounds[branch_var_idx] = (1, 1)
                desc_l = f"x{branch_var_idx+1}=0"
                desc_r = f"x{branch_var_idx+1}=1"
            else:
                floor_val = math.floor(branch_val)
                left_bounds[branch_var_idx] = (left_bounds[branch_var_idx][0], floor_val)
                right_bounds[branch_var_idx] = (floor_val + 1, right_bounds[branch_var_idx][1])
                desc_l = f"x{branch_var_idx+1}<={floor_val}"
                desc_r = f"x{branch_var_idx+1}>={floor_val + 1}"
            
            node_left = Node(id=node_counter, bounds=left_bounds, parent_id=current.id, branch_desc=desc_l)
            node_counter += 1
            node_right = Node(id=node_counter, bounds=right_bounds, parent_id=current.id, branch_desc=desc_r)
            
            nodes.extend([node_left, node_right])
            active_nodes.extend([node_left, node_right])

    return nodes, best_z, best_x

# --- INTERFAZ STREAMLIT ---

st.title("🌳 Método de Ramificación y Acotamiento")
st.markdown("Resolución paso a paso para modelos Enteros Puros, Mixtos y Binarios.")

col_config, col_main = st.columns([1, 2])

with col_config:
    st.header("Configuración")
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
        t = cols_t[i].selectbox(f"x{i+1}", ["Entera (I)", "Continua (C)", "Binaria (B)"], key=f"t_{i}")
        var_types.append(t)
        
    st.subheader("Restricciones")
    A = []
    b = []
    senses = []
    for i in range(num_cons):
        st.write(f"Restricción {i+1}")
        # num_vars + 2 columnas (para variables, tipo de signo y resultado)
        cols_r = st.columns(num_vars + 2)
        row = []
        for j in range(num_vars):
            val = cols_r[j].number_input(f"x{j+1}", value=1.0, key=f"a_{i}_{j}", label_visibility="collapsed")
            row.append(val)
        A.append(row)
        
        # Selector de tipo de restricción (<=, >=, ==)
        sense = cols_r[-2].selectbox("Tipo", ["<=", ">=", "=="], key=f"s_{i}", label_visibility="collapsed")
        senses.append(sense)
        
        # Lado derecho de la restricción (RHS)
        b_val = cols_r[-1].number_input("RHS", value=10.0, key=f"b_{i}", label_visibility="collapsed")
        b.append(b_val)

with col_main:
    if st.button("🚀 Resolver Modelo", type="primary"):
        with st.spinner("Construyendo el árbol..."):
            nodes, best_z, best_x = branch_and_bound(c, A, b, senses, var_types, is_max)
            
            # --- CREACIÓN DEL GRÁFICO (MÁS COMPACTO) ---
            dot = graphviz.Digraph(format='png')
            
            # Ajustes compactos: nodos más pequeños, menos separación
            dot.attr(rankdir='TB', nodesep='0.4', ranksep='0.6', splines='polyline')
            dot.attr('node', fontname='Helvetica', shape='box', style='filled', margin='0.1', fontsize='10')
            dot.attr('edge', fontname='Helvetica', fontsize='9', fontcolor='darkblue')
            
            for n in nodes:
                color = "#e1f5fe" 
                if n.status == "Infactible": color = "#ffcdd2" 
                elif n.status == "Agotado (Por Cota)": color = "#f5f5f5" 
                elif n.status == "Solución Entera": color = "#c8e6c9" 
                
                label = f"NODO {n.id}\n"
                if n.id != 0:
                    label += f"({n.branch_desc})\n"
                
                if n.z is not None:
                    label += "-" * 15 + "\n"
                    label += f"Z = {n.z:.3f}\n"
                    for i in range(len(n.x)):
                        label += f"x{i+1} = {n.x[i]:.3f}\n"
                
                label += "-" * 15 + "\n"
                label += f"[{n.status}]"
                
                dot.node(str(n.id), label, fillcolor=color)
                
                if n.parent_id is not None:
                    dot.edge(str(n.parent_id), str(n.id), label=f" {n.branch_desc} ")
            
            # --- RENDERIZADO Y DESCARGA ---
            st.success("¡Modelo resuelto!")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.write("### Solución Óptima Encontrada")
                if best_x is not None:
                    st.write(f"**Z:** {best_z:.4f}")
                    st.write("**Variables:**", {f"x{i+1}": best_x[i] for i in range(num_vars)})
                else:
                    st.error("No se encontró ninguna solución factible entera/binaria.")
                    
            with col_res2:
                # Generar el PDF y mostrar el botón de descarga
                pdf_bytes = dot.pipe(format='pdf')
                st.write("### Exportar")
                st.download_button(
                    label="📄 Descargar Árbol Completo (PDF)",
                    data=pdf_bytes,
                    file_name="Arbol_Ramificacion.pdf",
                    mime="application/pdf"
                )
                
            st.write("### Visualización del Árbol")
            # Gráfico mostrado en pantalla
            st.graphviz_chart(dot, use_container_width=True)
