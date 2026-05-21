import streamlit as st
import numpy as np
import math
from scipy.optimize import linprog
import graphviz
import matplotlib.pyplot as plt

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
    
    for i in range(len(senses)):
        if senses[i] == "<=":
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif senses[i] == ">=":
            A_ub.append([-val for val in A[i]])
            b_ub.append(-b[i])
        elif senses[i] == "==":
            A_eq.append(A[i])
            b_eq.append(b[i])
            
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
                desc_l = f"x{branch_var_idx+1} = 0"
                desc_r = f"x{branch_var_idx+1} = 1"
            else:
                floor_val = math.floor(branch_val)
                left_bounds[branch_var_idx] = (left_bounds[branch_var_idx][0], floor_val)
                right_bounds[branch_var_idx] = (floor_val + 1, right_bounds[branch_var_idx][1])
                desc_l = f"x{branch_var_idx+1} <= {floor_val}"
                desc_r = f"x{branch_var_idx+1} >= {floor_val + 1}"
            
            node_left = Node(id=node_counter, bounds=left_bounds, parent_id=current.id, branch_desc=desc_l)
            node_counter += 1
            node_right = Node(id=node_counter, bounds=right_bounds, parent_id=current.id, branch_desc=desc_r)
            
            nodes.extend([node_left, node_right])
            active_nodes.extend([node_left, node_right])

    # Ordenar nodos por ID para mostrarlos correctamente luego
    nodes.sort(key=lambda n: n.id)
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
        cols_r = st.columns(num_vars + 2)
        row = []
        for j in range(num_vars):
            val = cols_r[j].number_input(f"x{j+1}", value=1.0, key=f"a_{i}_{j}", label_visibility="collapsed")
            row.append(val)
        A.append(row)
        
        sense = cols_r[-2].selectbox("Tipo", ["<=", ">=", "=="], key=f"s_{i}", label_visibility="collapsed")
        senses.append(sense)
        
        b_val = cols_r[-1].number_input("RHS", value=10.0, key=f"b_{i}", label_visibility="collapsed")
        b.append(b_val)

with col_main:
    if st.button("🚀 Resolver Modelo", type="primary"):
        with st.spinner("Resolviendo y construyendo el árbol..."):
            nodes, best_z, best_x = branch_and_bound(c, A, b, senses, var_types, is_max)
            
            st.success("¡Modelo resuelto!")
            st.write("### Solución Óptima Encontrada")
            if best_x is not None:
                st.write(f"**Z:** {best_z:.4f}")
                st.write("**Variables:**", {f"x{i+1}": best_x[i] for i in range(num_vars)})
            else:
                st.error("No se encontró ninguna solución factible entera/binaria.")

            # --- 1. MÉTODO GRÁFICO (SOLO PARA 2 VARIABLES) ---
            if num_vars == 2:
                st.write("### 📈 Método Gráfico y Cortes de Ramificación")
                st.markdown("Las líneas negras sólidas son las restricciones originales. Las líneas discontinuas de colores representan los subproblemas (cortes de ramificación).")
                
                # Calcular límite máximo del gráfico basado en intercepciones
                max_val = 5
                for i in range(len(A)):
                    for j in range(2):
                        if A[i][j] > 0:
                            max_val = max(max_val, b[i] / A[i][j])
                if best_x is not None:
                    max_val = max(max_val, best_x[0], best_x[1])
                max_val = math.ceil(max_val * 1.2) # Margen del 20%
                
                fig, ax = plt.subplots(figsize=(7, 5))
                x_vals = np.linspace(0, max_val, 400)
                
                # Graficar restricciones originales
                for i in range(len(A)):
                    a1, a2 = A[i][0], A[i][1]
                    if a2 != 0:
                        y_vals = (b[i] - a1 * x_vals) / a2
                        ax.plot(x_vals, y_vals, label=f'R{i+1}: {a1}x1 + {a2}x2 {senses[i]} {b[i]}', color='black', linewidth=1.5)
                    elif a1 != 0:
                        ax.axvline(x=b[i]/a1, label=f'R{i+1}: {a1}x1 {senses[i]} {b[i]}', color='black', linewidth=1.5)
                
                # Graficar líneas de subproblemas (Ramificaciones)
                colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
                c_idx = 0
                drawn_desc = set()
                
                for n in nodes:
                    if n.id != 0 and n.branch_desc and n.branch_desc not in drawn_desc:
                        parts = n.branch_desc.split()
                        if len(parts) == 3:
                            var_str, op, val_str = parts
                            val = float(val_str)
                            color_to_use = colors[c_idx % len(colors)]
                            
                            if var_str == 'x1':
                                ax.axvline(x=val, color=color_to_use, linestyle='--', alpha=0.7, label=f'Corte: {n.branch_desc}')
                            elif var_str == 'x2':
                                ax.axhline(y=val, color=color_to_use, linestyle='--', alpha=0.7, label=f'Corte: {n.branch_desc}')
                            
                            drawn_desc.add(n.branch_desc)
                            c_idx += 1
                
                # Marcar la solución óptima si existe
                if best_x is not None:
                    ax.plot(best_x[0], best_x[1], 'r*', markersize=12, label='Óptimo Entero/Mixto')

                ax.set_xlim(0, max_val)
                ax.set_ylim(0, max_val)
                ax.set_xlabel('$x_1$')
                ax.set_ylabel('$x_2$')
                # Colocar leyenda fuera del gráfico
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
            else:
                st.info("El método gráfico solo está disponible para modelos de 2 variables. Revisa el árbol y la evidencia matemática abajo.")

            # --- 2. CREACIÓN DEL ÁRBOL ---
            st.write("### 🌳 Árbol de Ramificación y Acotamiento")
            
            dot = graphviz.Digraph(format='png')
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
            
            # Gráfico y botón PDF
            col_pdf, col_empty = st.columns([1, 2])
            with col_pdf:
                pdf_bytes = dot.pipe(format='pdf')
                st.download_button(
                    label="📄 Descargar Árbol en PDF",
                    data=pdf_bytes,
                    file_name="Arbol_Ramificacion.pdf",
                    mime="application/pdf"
                )
            
            st.graphviz_chart(dot, use_container_width=True)

            # --- 3. EVIDENCIA DE SUBPROBLEMAS ---
            st.write("### 📝 Evidencia de Subproblemas (Modelos Evaluados)")
            st.markdown("Desglose del modelo matemático resuelto en cada nodo del árbol.")
            
            with st.expander("Ver desglose completo de nodos", expanded=True):
                for n in nodes:
                    padre = f"Hijo del Nodo {n.parent_id}" if n.parent_id is not None else "Nodo Raíz"
                    st.markdown(f"#### Nodo {n.id} ({padre})")
                    
                    if n.id != 0:
                        st.markdown(f"**Restricción añadida en la ramificación:** $ {n.branch_desc} $")
                    
                    # Formatear los límites activos para este subproblema
                    bounds_str = []
                    for i, bnd in enumerate(n.bounds):
                        if bnd[0] == 0 and bnd[1] is None:
                            bounds_str.append(f"$x_{i+1} \\ge 0$")
                        else:
                            if bnd[0] == bnd[1]:
                                bounds_str.append(f"$x_{i+1} = {bnd[0]}$")
                            else:
                                if bnd[0] is not None and bnd[0] > 0:
                                    bounds_str.append(f"$x_{i+1} \\ge {bnd[0]}$")
                                if bnd[1] is not None:
                                    bounds_str.append(f"$x_{i+1} \\le {bnd[1]}$")
                    
                    st.markdown("**Límites activos evaluados:** " + ", ".join(bounds_str))
                    
                    # Resultado del subproblema
                    if n.status == "Infactible":
                        st.error(f"**Estado:** {n.status}. No hay región factible para este conjunto de límites.")
                    else:
                        if n.status == "Solución Entera":
                            st.success(f"**Estado:** {n.status}. Z = {n.z:.4f}")
                        elif n.status == "Agotado (Por Cota)":
                            st.warning(f"**Estado:** {n.status}. Z = {n.z:.4f} (No mejora la solución conocida)")
                        else:
                            st.info(f"**Estado:** {n.status}. Z = {n.z:.4f}")
                        
                        # Mostrar variables del resultado
                        x_res = ", ".join([f"$x_{i+1} = {n.x[i]:.4f}$" for i in range(len(n.x))])
                        st.markdown(f"**Variables:** {x_res}")
                    
                    st.markdown("---")
