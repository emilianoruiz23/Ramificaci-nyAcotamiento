import streamlit as st
import numpy as np
import math
from scipy.optimize import linprog
import graphviz
import matplotlib.pyplot as plt
from PIL import Image
import io
import itertools

st.set_page_config(layout="wide", page_title="Método de Ramificación y Acotamiento")

# --- CLASES Y FUNCIONES DEL ALGORITMO ---

class Node:
    def __init__(self, id, bounds, parent_id=None, branch_desc="Ninguna"):
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

def get_all_added_constraints(n_id, nodes_dict):
    curr = nodes_dict[n_id]
    cons = []
    while curr.parent_id is not None:
        cons.append(curr.branch_desc)
        curr = nodes_dict[curr.parent_id]
    return " y ".join(reversed(cons)) if cons else "Ninguna (Modelo Original)"

def branch_and_bound(c, A, b, senses, var_types, is_max):
    num_vars = len(c)
    initial_bounds = [(0, 1) if t == 'Binaria (B)' else (0, None) for t in var_types]
            
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
            current.status = "Agotado (Cota superada)"
            continue
            
        is_feasible_integer = True
        branch_var_idx, branch_val = -1, None
        
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
            left_bounds, right_bounds = current.bounds.copy(), current.bounds.copy()
            
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

    nodes.sort(key=lambda n: n.id)
    return nodes, best_z, best_x

def find_feasible_corners(c, A, b, senses, is_max):
    lines_A = [row[:] for row in A]
    lines_b = b[:]
    lines_A.extend([[1, 0], [0, 1]])
    lines_b.extend([0, 0])
    
    corners = []
    for combo in itertools.combinations(range(len(lines_A)), 2):
        try:
            A_mat = np.array([lines_A[combo[0]], lines_A[combo[1]]])
            b_vec = np.array([lines_b[combo[0]], lines_b[combo[1]]])
            x = np.linalg.solve(A_mat, b_vec)
            x = np.round(x, 5)
            
            if x[0] < 0 or x[1] < 0: continue
            
            feasible = True
            for i in range(len(A)):
                val = A[i][0]*x[0] + A[i][1]*x[1]
                if senses[i] == '<=' and val > b[i] + 1e-5: feasible = False
                elif senses[i] == '>=' and val < b[i] - 1e-5: feasible = False
                elif senses[i] == '==' and abs(val - b[i]) > 1e-5: feasible = False
            
            if feasible and not any(np.allclose(x, c_pt) for c_pt in corners):
                corners.append(x)
        except np.linalg.LinAlgError:
            pass
            
    evaluations = [(pt, c[0]*pt[0] + c[1]*pt[1]) for pt in corners]
    evaluations.sort(key=lambda item: item[1], reverse=is_max)
    return evaluations

# --- INTERFAZ STREAMLIT ---

st.title("🌳 Método de Ramificación y Acotamiento")
st.markdown("Resolución paso a paso con tabla de subproblemas y pdf unificado.")

col_config, col_main = st.columns([1, 2.5])

with col_config:
    st.header("Configuración")
    opt_type = st.selectbox("Objetivo:", ["Maximizar", "Minimizar"])
    is_max = opt_type == "Maximizar"
    
    num_vars = st.number_input("Número de variables (Máx 10)", min_value=1, max_value=10, value=2)
    num_cons = st.number_input("Número de restricciones", min_value=1, max_value=10, value=2)
    
    st.subheader("Función Objetivo (Z)")
    c = []
    cols_c = st.columns(num_vars)
    for i in range(num_vars): c.append(cols_c[i].number_input(f"x{i+1}", value=1.0, key=f"c_{i}"))
        
    st.subheader("Tipos de Variables")
    var_types = []
    cols_t = st.columns(num_vars)
    for i in range(num_vars): var_types.append(cols_t[i].selectbox(f"x{i+1}", ["Entera (I)", "Continua (C)", "Binaria (B)"], key=f"t_{i}"))
        
    st.subheader("Restricciones")
    A, b, senses = [], [], []
    for i in range(num_cons):
        cols_r = st.columns(num_vars + 2)
        row = []
        for j in range(num_vars): row.append(cols_r[j].number_input(f"x{j+1}", value=1.0, key=f"a_{i}_{j}", label_visibility="collapsed"))
        A.append(row)
        senses.append(cols_r[-2].selectbox("Tipo", ["<=", ">=", "=="], key=f"s_{i}", label_visibility="collapsed"))
        b.append(cols_r[-1].number_input("RHS", value=10.0, key=f"b_{i}", label_visibility="collapsed"))

with col_main:
    if st.button("🚀 Resolver Modelo", type="primary"):
        with st.spinner("Resolviendo y construyendo reporte..."):
            nodes, best_z, best_x = branch_and_bound(c, A, b, senses, var_types, is_max)
            nodes_dict = {n.id: n for n in nodes}
            
            st.success("¡Modelo resuelto!")
            if best_x is not None:
                st.write(f"### **Óptimo Global -> Z:** {best_z:.4f} | **Variables:** " + ", ".join([f"x{i+1}={best_x[i]}" for i in range(num_vars)]))
            else:
                st.error("No se encontró ninguna solución factible entera/binaria.")

            pdf_images = []
            
            # --- 1. MÉTODO GRÁFICO Y COORDENADAS ---
            if num_vars == 2:
                evaluations = find_feasible_corners(c, A, b, senses, is_max)
                
                fig = plt.figure(figsize=(10, 4.5))
                gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
                ax_plot = fig.add_subplot(gs[0])
                ax_text = fig.add_subplot(gs[1])
                
                # Gráfico
                max_val = 5
                for i in range(len(A)):
                    for j in range(2):
                        if A[i][j] > 0: max_val = max(max_val, b[i] / A[i][j])
                if best_x is not None: max_val = max(max_val, best_x[0], best_x[1])
                max_val = math.ceil(max_val * 1.2)
                
                d = np.linspace(0, max_val, 400)
                x_grid, y_grid = np.meshgrid(d, d)
                feasible_region = np.ones_like(x_grid, dtype=bool)
                for i in range(len(A)):
                    a1, a2 = A[i][0], A[i][1]
                    if senses[i] == "<=": feasible_region &= (a1*x_grid + a2*y_grid <= b[i])
                    elif senses[i] == ">=": feasible_region &= (a1*x_grid + a2*y_grid >= b[i])
                
                ax_plot.imshow(feasible_region.astype(int), extent=(0, max_val, 0, max_val), origin='lower', cmap='Greens', alpha=0.3)
                colors_const = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                x_vals = np.linspace(0, max_val, 400)
                
                for i in range(len(A)):
                    a1, a2, c_color = A[i][0], A[i][1], colors_const[i % len(colors_const)]
                    if a2 != 0: ax_plot.plot(x_vals, (b[i] - a1 * x_vals) / a2, label=f'R{i+1}: {a1}x1+{a2}x2{senses[i]}{b[i]}', color=c_color, linewidth=2)
                    elif a1 != 0: ax_plot.axvline(x=b[i]/a1, label=f'R{i+1}: {a1}x1{senses[i]}{b[i]}', color=c_color, linewidth=2)
                
                if nodes[0].x is not None: ax_plot.plot(nodes[0].x[0], nodes[0].x[1], 'bo', markersize=8, label='Óptimo Relajado')
                if best_x is not None: ax_plot.plot(best_x[0], best_x[1], 'r*', markersize=12, label='Óptimo Entero')
                
                ax_plot.set_xlim(0, max_val); ax_plot.set_ylim(0, max_val)
                ax_plot.legend(loc='upper right', fontsize='x-small')
                ax_plot.grid(True, alpha=0.3)
                
                # Texto de Coordenadas
                ax_text.axis('off')
                txt_content = "Evaluación de Vértices (Sol. Relajada):\n\n"
                txt_content += "Intersecciones factibles:\n"
                for idx, (pt, val) in enumerate(evaluations):
                    mark = " (Óptimo Inicial)" if idx == 0 else ""
                    txt_content += f"P{idx+1} ({pt[0]:.2f}, {pt[1]:.2f}) -> Z = {val:.2f}{mark}\n"
                txt_content += "\nEstos valores muestran cómo se obtiene\nla coordenada inicial antes de ramificar."
                ax_text.text(0, 0.9, txt_content, fontsize=10, verticalalignment='top', family='monospace')
                
                st.write("### 📈 Solución Lineal Inicial y Coordenadas")
                st.pyplot(fig)
                
                buf_fig = io.BytesIO()
                fig.savefig(buf_fig, format='png', bbox_inches='tight', dpi=150)
                buf_fig.seek(0)
                pdf_images.append(Image.open(buf_fig).convert('RGB'))

            # --- 2. ÁRBOL DE RAMIFICACIÓN ---
            st.write("### 🌳 Árbol de Ramificación")
            dot = graphviz.Digraph(format='png')
            dot.attr(rankdir='TB', nodesep='0.3', ranksep='0.5')
            dot.attr('node', fontname='Helvetica', shape='box', style='filled', fontsize='9', margin='0.1')
            dot.attr('edge', fontname='Helvetica', fontsize='9', fontcolor='darkblue')
            
            for n in nodes:
                color = "#e1f5fe" 
                if n.status == "Infactible": color = "#ffcdd2" 
                elif "Agotado" in n.status: color = "#f5f5f5" 
                elif n.status == "Solución Entera": color = "#c8e6c9" 
                
                label = f"NODO {n.id}\n"
                if n.id != 0: label += f"[{n.branch_desc}]\n"
                if n.z is not None:
                    label += f"Z = {n.z:.3f}\n" + ", ".join([f"x{i+1}={n.x[i]:.2f}" for i in range(len(n.x))]) + f"\n({n.status})"
                else:
                    label += f"({n.status})"
                
                dot.node(str(n.id), label, fillcolor=color)
                if n.parent_id is not None: dot.edge(str(n.parent_id), str(n.id), label=f" {n.branch_desc} ")
            
            st.graphviz_chart(dot, use_container_width=True)
            img_tree = Image.open(io.BytesIO(dot.pipe(format='png'))).convert('RGB')
            pdf_images.append(img_tree)

            # --- 3. TABLA DE SUBPROBLEMAS ---
            st.write("### 📝 Tabla de Subproblemas Evaluados")
            fo_str = "Max " if is_max else "Min "
            fo_str += "Z = " + " + ".join([f"{c[i]}x{i+1}" for i in range(len(c))])
            
            table_data = []
            for n in nodes:
                nodo_str = f"Nodo {n.id}"
                added_res = get_all_added_constraints(n.id, nodes_dict)
                if n.status == "Infactible": z_str, x_str = "-", "-"
                else:
                    z_str = f"{n.z:.4f}"
                    x_str = ", ".join([f"x{i+1}={n.x[i]:.4f}" for i in range(len(n.x))])
                table_data.append([nodo_str, fo_str, added_res, x_str, z_str, n.status])
                
            cols = ["Subproblema", "F.O.", "Restricciones Usadas (Acumuladas)", "Valores (x)", "Z", "Cota / Estado"]
            st.table([dict(zip(cols, row)) for row in table_data])
            
            # Generar imagen de la tabla para el PDF
            fig_tab, ax_tab = plt.subplots(figsize=(12, max(3, len(table_data) * 0.4)))
            ax_tab.axis('tight'); ax_tab.axis('off')
            tab = ax_tab.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
            tab.auto_set_font_size(False)
            tab.set_fontsize(8)
            tab.scale(1, 1.5)
            
            buf_tab = io.BytesIO()
            fig_tab.savefig(buf_tab, format='png', bbox_inches='tight', dpi=150)
            buf_tab.seek(0)
            pdf_images.append(Image.open(buf_tab).convert('RGB'))
            
            # --- DESCARGA PDF UNIFICADO ---
            if len(pdf_images) > 0:
                pdf_buffer = io.BytesIO()
                pdf_images[0].save(pdf_buffer, format='PDF', save_all=True, append_images=pdf_images[1:])
                st.download_button(
                    label="📄 Descargar Reporte Completo (Gráfico + Árbol + Tabla en PDF)",
                    data=pdf_buffer.getvalue(),
                    file_name="Reporte_Ramificacion_Unificado.pdf",
                    mime="application/pdf",
                    type="primary"
                )
