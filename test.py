import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide", page_title="Calculateur Bassin & Fouille")

# --- FONCTIONS GEOMETRIQUES ---

def get_offset(depth, angle_deg):
    """Calcule le recul horizontal (offset) selon la pente."""
    if angle_deg >= 89.9:
        return 0.0
    angle_safe = max(0.1, angle_deg)
    return depth / math.tan(math.radians(angle_safe))

def calc_triangle(W, L, Depth, offset, angle_deg):
    """Logique pour le bassin triangulaire."""
    # Sommets HAUT (3 points)
    top = [
        np.array([0, 0, 0]),
        np.array([W, 0, 0]),
        np.array([0, L, 0])
    ]
    
    # Sommets BAS (3 points)
    p_bot_origin = np.array([offset, offset, -Depth])
    
    hyp_len = math.sqrt(W**2 + L**2)
    c_orig = W * L
    c_new = c_orig - offset * hyp_len
    
    is_valid = True
    if offset >= W or offset >= L:
        is_valid = False
    elif c_new <= 0:
        is_valid = False
    else:
        x_bot = (c_new - W * offset) / L
        y_bot = (c_new - L * offset) / W
        if x_bot <= offset or y_bot <= offset:
            is_valid = False

    if not is_valid:
        bot = [p_bot_origin] * 3
    else:
        if angle_deg >= 89.9:
            bot = [
                np.array([offset, offset, -Depth]),
                np.array([W, 0, -Depth]),
                np.array([0, L, -Depth])
            ]
        else:
            bot = [
                p_bot_origin,
                np.array([x_bot, offset, -Depth]),
                np.array([offset, y_bot, -Depth])
            ]
            
    return {"top": top, "bot": bot, "is_valid": is_valid, "type": "triangle"}

def calc_rectangle(W, L, Depth, offset, angle_deg):
    """Logique pour le bassin rectangulaire."""
    # Sommets HAUT (4 points)
    top = [
        np.array([0, 0, 0]),
        np.array([W, 0, 0]),
        np.array([W, L, 0]),
        np.array([0, L, 0])
    ]
    
    is_valid = True
    if (2 * offset) >= W or (2 * offset) >= L:
        is_valid = False
    
    if not is_valid:
        bot = [np.array([W/2, L/2, -Depth])] * 4
    else:
        x_min, x_max = offset, W - offset
        y_min, y_max = offset, L - offset
        bot = [
            np.array([x_min, y_min, -Depth]),
            np.array([x_max, y_min, -Depth]),
            np.array([x_max, y_max, -Depth]),
            np.array([x_min, y_max, -Depth])
        ]
        
    return {"top": top, "bot": bot, "is_valid": is_valid, "type": "rectangle"}

def calculate_volume(geom, depth):
    if geom["type"] == "triangle":
        w_top = geom["top"][1][0]
        l_top = geom["top"][2][1]
        area_top = 0.5 * w_top * l_top
        
        b = geom["bot"]
        w_bot = max(0, b[1][0] - b[0][0])
        l_bot = max(0, b[2][1] - b[0][1])
        area_bot = 0.5 * w_bot * l_bot
        
    else: # Rectangle
        w_top = geom["top"][1][0]
        l_top = geom["top"][3][1]
        area_top = w_top * l_top
        
        b = geom["bot"]
        w_bot = max(0, b[1][0] - b[0][0])
        l_bot = max(0, b[3][1] - b[0][1])
        area_bot = w_bot * l_bot

    w_mid = (w_top + w_bot) / 2
    l_mid = (l_top + l_bot) / 2
    
    if geom["type"] == "triangle":
        area_mid = 0.5 * w_mid * l_mid
    else:
        area_mid = w_mid * l_mid
    
    vol = (depth / 6) * (area_top + area_bot + 4 * area_mid)
    return vol, area_top, area_bot

def create_manual_mesh(geom):
    """Crée les indices i, j, k manuellement pour éviter les artefacts."""
    # Liste de tous les sommets : D'abord TOP, ensuite BOT
    verts = geom["top"] + geom["bot"]
    x = [v[0] for v in verts]
    y = [v[1] for v in verts]
    z = [v[2] for v in verts]
    
    n = len(geom["top"]) # 3 ou 4
    
    # Listes des indices pour les triangles
    I = []
    J = []
    K = []
    
    # 1. FACE DU HAUT (TOP)
    if n == 3: # Triangle
        I += [0]; J += [1]; K += [2]
    else: # Rectangle (2 triangles)
        I += [0, 0]; J += [1, 2]; K += [2, 3]
        
    # 2. FACE DU BAS (BOT)
    # Les indices du bas commencent à 'n'
    if n == 3:
        I += [n]; J += [n+1]; K += [n+2]
    else:
        I += [n, n]; J += [n+1, n+2]; K += [n+2, n+3]
        
    # 3. FACES LATERALES (CÔTÉS)
    # Chaque côté est un quad formé par (Top1, Top2, Bot2, Bot1)
    # On le coupe en 2 triangles
    for k in range(n):
        # Indices des 4 coins de la face
        t1 = k
        t2 = (k + 1) % n
        b1 = k + n
        b2 = ((k + 1) % n) + n
        
        # Triangle 1
        I.append(t1); J.append(t2); K.append(b2)
        # Triangle 2
        I.append(t1); J.append(b2); K.append(b1)
        
    return x, y, z, I, J, K

# --- INTERFACE ---

st.title("🚜 Calculateur de Fouille")
st.markdown("Dimensionnement de bassin ou trou avec pentes.")

with st.sidebar:
    st.header("1. Forme")
    shape_type = st.radio("Type de base :", ["Triangle Rectangle", "Rectangle"], index=0)
    
    st.header("2. Dimensions")
    val_W = st.number_input("Largeur X (m)", value=10.0, step=0.5, min_value=1.0)
    val_L = st.number_input("Longueur Y (m)", value=8.0, step=0.5, min_value=1.0)
    
    st.header("3. Creusement")
    val_D = st.number_input("Profondeur Z (m)", value=2.0, step=0.1, min_value=0.1)
    val_Angle = st.slider("Angle Pente (°)", min_value=1, max_value=90, value=45)
    
    offset = get_offset(val_D, val_Angle)
    st.caption(f"Recul horizontal requis : {offset:.2f} m")

# CALCUL
if shape_type == "Triangle Rectangle":
    geom = calc_triangle(val_W, val_L, val_D, offset, val_Angle)
else:
    geom = calc_rectangle(val_W, val_L, val_D, offset, val_Angle)

# ERROR CHECK
if not geom["is_valid"]:
    st.error("🛑 **CONFIGURATION IMPOSSIBLE**")
    st.markdown(f"""
    Les pentes sont trop douces pour la profondeur demandée. 
    Les parois se croisent et le fond n'existe plus.
    **Recul nécessaire :** {offset:.2f}m.
    """)
    st.stop()

vol, a_top, a_bot = calculate_volume(geom, val_D)

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Volume Déblai", f"{vol:.2f} m³")
k2.metric("Surface Fond", f"{a_bot:.2f} m²")
k3.metric("Surface Haut", f"{a_top:.2f} m²")
k4.metric("Recul Pente", f"{offset:.2f} m")

# VISUALISATION
tab1, tab2 = st.tabs(["Vue 3D", "Plan 2D"])

with tab1:
    # Récupération des données MAILLÉES manuellement (plus de alphahull)
    mx, my, mz, mi, mj, mk = create_manual_mesh(geom)
    
    fig_3d = go.Figure(data=[
        go.Mesh3d(
            x=mx, y=my, z=mz,
            # On fournit explicitement les triangles (i, j, k)
            i=mi, j=mj, k=mk,
            opacity=0.6,
            color='#00a8ff',
            flatshading=True,
            name="Volume",
            hoverinfo='none',
            # Ces paramètres aident à l'éclairage
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2)
        )
    ])
    
    # Arêtes (Wireframe) pour faire joli
    def add_line(fig, p1, p2, color="black"):
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines', line=dict(color=color, width=4), showlegend=False
        ))

    n_points = len(geom["top"])
    for i in range(n_points):
        next_i = (i + 1) % n_points
        add_line(fig_3d, geom["top"][i], geom["top"][next_i], "blue") # Haut
        add_line(fig_3d, geom["bot"][i], geom["bot"][next_i], "red")  # Bas
        add_line(fig_3d, geom["top"][i], geom["bot"][i], "black")     # Piliers

    fig_3d.update_layout(scene=dict(aspectmode='data'), height=600, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    fig_2d = go.Figure()
    
    t_x = [v[0] for v in geom["top"]] + [geom["top"][0][0]]
    t_y = [v[1] for v in geom["top"]] + [geom["top"][0][1]]
    b_x = [v[0] for v in geom["bot"]] + [geom["bot"][0][0]]
    b_y = [v[1] for v in geom["bot"]] + [geom["bot"][0][1]]

    fig_2d.add_trace(go.Scatter(
        x=t_x, y=t_y, fill='toself', fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='blue'), name='Ouverture'
    ))
    fig_2d.add_trace(go.Scatter(
        x=b_x, y=b_y, fill='toself', fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='red', dash='dash'), name='Fond'
    ))

    # Cotes
    w_bot_val = geom["bot"][1][0] - geom["bot"][0][0]
    if w_bot_val > 0.05:
        fig_2d.add_annotation(
            x=geom["bot"][0][0] + w_bot_val/2, y=geom["bot"][0][1], 
            text=f"{w_bot_val:.2f}m", showarrow=True, ay=20
        )
        
    idx_l = 2 if shape_type == "Triangle Rectangle" else 3
    l_bot_val = geom["bot"][idx_l][1] - geom["bot"][0][1]
    if l_bot_val > 0.05:
        fig_2d.add_annotation(
            x=geom["bot"][0][0], y=geom["bot"][0][1] + l_bot_val/2, 
            text=f"{l_bot_val:.2f}m", showarrow=True, ax=-40
        )

    fig_2d.update_layout(
        title=f"Vue de dessus",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=600
    )
    st.plotly_chart(fig_2d, use_container_width=True)