import streamlit as st
import numpy as np
import math
from scipy.optimize import linprog
import graphviz
import matplotlib.pyplot as plt
from PIL import Image
import io

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
    
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    
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
    A, b, senses = [], [], []
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

            # --- PREPARAR IMÁGENES PARA EL REPORTE PDF ---
            pdf_images = []
            
            # --- 1. MÉTODO GRÁFICO (SOLO PARA 2 VARIABLES) ---
            if num_vars == 2:
                st.write("### 📈 Método Gráfico y Cortes de Ramificación")
                st.markdown("La zona sombreada en verde es la **región factible inicial**. El punto azul es el óptimo lineal relajado.")
                
                max_val = 5
                for i in range(len(A)):
                    for j in range(2):
                        if A[i][j] > 0: max_val = max(max_val, b[i] / A[i][j])
                if best_x is not None: max_val = max(max_val, best_x[0], best_x[1])
                max_val = math.ceil(max_val * 1.2)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Sombreado de la Región Factible
                d = np.linspace(0, max_val, 400)
                x_grid, y_grid = np.meshgrid(d, d)
                feasible_region = np.ones_like(x_grid, dtype=bool)
                
                for i in range(len(A)):
                    a1, a2 = A[i][0], A[i][1]
                    if senses[i] == "<=":
                        feasible_region &= (a1*x_grid + a2*y_grid <= b[i])
                    elif senses[i] == ">=":
                        feasible_region &= (a1*x_grid + a2*y_grid >= b[i])
                
                # Dibujar la sombra verde
                ax.imshow(feasible_region.astype(int), extent=(0, max_val, 0, max_val), origin='lower', cmap='Greens', alpha=0.3)
                
                # Graficar restricciones con diferentes colores
                colors_const = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                x_vals = np.linspace(0, max_val, 400)
                
                for i in range(len(A)):
                    a1, a2 = A[i][0], A[i][1]
                    c_color = colors_const[i % len(colors_const)]
                    if a2 != 0:
                        y_vals = (b[i] - a1 * x_vals) / a2
                        ax.plot(x_vals, y_vals, label=f'R{i+1}: {a1}x1 + {a2}x2 {senses[i]} {b[i]}', color=c_color, linewidth=2)
                    elif a1 != 0:
                        ax.axvline(x=b[i]/a1, label=f'R{i+1}: {a1}x1 {senses[i]} {b[i]}', color=c_color, linewidth=2)
                
                # Marcar la solución Lineal Relajada (Nodo 0)
                if nodes[0].x is not None:
                    ax.plot(nodes[0].x[0], nodes[0].x[1], 'bo', markersize=9, label='Óptimo Relajado (Lineal)')
                
                # Graficar líneas de subproblemas (Ramificaciones en gris discontinuo)
                drawn_desc = set()
                for n in nodes:
                    if n.id != 0 and n.branch_desc and n.branch_desc not in drawn_desc:
                        parts = n.branch_desc.split()
                        if len(parts) == 3:
                            var_str, op, val_str = parts
                            val = float(val_str)
                            if var_str == 'x1':
                                ax.axvline(x=val, color='gray', linestyle='--', alpha=0.8)
                            elif var_str == 'x2':
                                ax.axhline(y=val, color='gray', linestyle='--', alpha=0.8)
                            drawn_desc.add(n.branch_desc)
                
                # Marcar la solución óptima entera final si existe
                if best_x is not None:
                    ax.plot(best_x[0], best_x[1], 'r*', markersize=15, label='Óptimo Entero/Mixto (Final)')

                ax.set_xlim(0, max_val)
                ax.set_ylim(0, max_val)
                ax.set_xlabel('$x_1$')
                ax.set_ylabel('$x_2$')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Guardar gráfico en memoria para el PDF
                buf_fig = io.BytesIO()
                fig.savefig(buf_fig, format='png', bbox_inches='tight', dpi=150)
                buf_fig.seek(0)
                pdf_images.append(Image.open(buf_fig).convert('RGB'))

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
                if n.id != 0: label += f"({n.branch_desc})\n"
                
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
            
            st.graphviz_chart(dot, use_container_width=True)
            
            # Guardar árbol en memoria para el PDF
            tree_png = dot.pipe(format='png')
            img_tree = Image.open(io.BytesIO(tree_png)).convert('RGB')
            pdf_images.append(img_tree)
            
            # Generar el PDF final Unificado
            if len(pdf_images) > 0:
                pdf_buffer = io.BytesIO()
                # Guarda la primera imagen y adjunta las demás como páginas extra
                pdf_images[0].save(
                    pdf_buffer, 
                    format='PDF', 
                    save_all=True, 
                    append_images=pdf_images[1:] if len(pdf_images) > 1 else []
                )
                
                st.download_button(
                    label="📄 Descargar Reporte Completo (Gráfico + Árbol en PDF)",
                    data=pdf_buffer.getvalue(),
                    file_name="Reporte_Ramificacion.pdf",
                    mime="application/pdf",
                    type="primary"
                )

            # --- 3. EVIDENCIA DE SUBPROBLEMAS ---
            st.write("### 📝 Evidencia de Subproblemas")
            with st.expander("Ver desglose completo de nodos", expanded=False):
                for n in nodes:
                    padre = f"Hijo del Nodo {n.parent_id}" if n.parent_id is not None else "Nodo Raíz"
                    st.markdown(f"#### Nodo {n.id} ({padre})")
                    if n.id != 0: st.markdown(f"**Restricción añadida:** $ {n.branch_desc} $")
                    
                    bounds_str = []
                    for i, bnd in enumerate(n.bounds):
                        if bnd[0] == 0 and bnd[1] is None: bounds_str.append(f"$x_{i+1} \\ge 0$")
                        else:
                            if bnd[0] == bnd[1]: bounds_str.append(f"$x_{i+1} = {bnd[0]}$")
                            else:
                                if bnd[0] is not None and bnd[0] > 0: bounds_str.append(f"$x_{i+1} \\ge {bnd[0]}$")
                                if bnd[1] is not None: bounds_str.append(f"$x_{i+1} \\le {bnd[1]}$")
                    
                    st.markdown("**Límites activos evaluados:** " + ", ".join(bounds_str))
                    
                    if n.status == "Infactible":
                        st.error(f"**Estado:** {n.status}. No hay región factible.")
                    else:
                        if n.status == "Solución Entera": st.success(f"**Estado:** {n.status}. Z = {n.z:.4f}")
                        elif n.status == "Agotado (Por Cota)": st.warning(f"**Estado:** {n.status}. Z = {n.z:.4f}")
                        else: st.info(f"**Estado:** {n.status}. Z = {n.z:.4f}")
                        
                        x_res = ", ".join([f"$x_{i+1} = {n.x[i]:.4f}$" for i in range(len(n.x))])
                        st.markdown(f"**Variables:** {x_res}")
                    st.markdown("---")
