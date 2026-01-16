import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide", page_title="Calculateur Bassin Retention")

# --- FONCTIONS DE CALCUL ---
def calculate_geometry(W, L, Depth, angle_deg):
    """Calcul les coordonnées des sommets du bassin."""
    
    # 1. Calcul du recul (Offset) basé sur la pente
    # Si 90°, mur vertical, offset = 0
    if angle_deg >= 89.9:
        offset = 0.0
    else:
        # Conversion radians
        angle_safe = max(0.1, angle_deg) 
        angle_rad = math.radians(angle_safe)
        offset = Depth / math.tan(angle_rad)
    
    # --- Sommets du HAUT (Grande Base) ---
    p_top_origin = np.array([0, 0, 0])
    p_top_x = np.array([W, 0, 0])
    p_top_y = np.array([0, L, 0])
    
    # --- Sommets du BAS (Petite Base) ---
    # Origine du fond (décalée de 'offset' en X et Y)
    p_bot_origin = np.array([offset, offset, -Depth])
    
    # Calcul de la ligne oblique (Hypoténuse du fond)
    # Equation de base : L*x + W*y = W*L
    # On décale cette ligne vers l'intérieur de la distance 'offset'
    hyp_len = math.sqrt(W**2 + L**2)
    c_orig = W * L
    c_new = c_orig - offset * hyp_len
    
    # --- VALIDATION STRICTE ---
    is_valid = True
    x_bot = 0
    y_bot = 0
    
    # Condition 1 : L'offset ne doit pas être plus grand que les côtés du mur
    if offset >= W or offset >= L:
        is_valid = False
    
    # Condition 2 : Calcul des intersections
    else:
        # Intersection sur l'axe X (à hauteur y = offset)
        # L*x + W*offset = c_new
        x_bot = (c_new - W * offset) / L
        
        # Intersection sur l'axe Y (à largeur x = offset)
        # L*offset + W*y = c_new
        y_bot = (c_new - L * offset) / W
        
        # Condition 3 (CRITIQUE) : Inversion du fond
        # Si x_bot est plus petit que l'offset, c'est que le triangle s'est retourné
        if x_bot <= offset or y_bot <= offset:
            is_valid = False

    # Si invalide, on renvoie des valeurs bidons pour ne pas faire planter Python, 
    # mais on signale l'erreur via "is_valid"
    if not is_valid:
        # On crée un tout petit triangle au centre juste pour la forme des variables
        center_x, center_y = W/2, L/2
        p_bot_origin = np.array([center_x, center_y, -Depth])
        p_bot_x = np.array([center_x+0.01, center_y, -Depth])
        p_bot_y = np.array([center_x, center_y+0.01, -Depth])
    else:
        # Cas Valide
        if angle_deg >= 89.9:
            # Vertical
            p_bot_x = np.array([W, 0, -Depth])
            p_bot_y = np.array([0, L, -Depth])
        else:
            # Normal
            p_bot_x = np.array([x_bot, offset, -Depth])
            p_bot_y = np.array([offset, y_bot, -Depth])
        
    return {
        "top": [p_top_origin, p_top_x, p_top_y],
        "bot": [p_bot_origin, p_bot_x, p_bot_y],
        "is_valid": is_valid,
        "offset": offset
    }

def calculate_volume(geom, depth):
    # Formule du Prismatoïde
    t = geom["top"]
    w_top = t[1][0]
    l_top = t[2][1]
    area_top = 0.5 * w_top * l_top
    
    b = geom["bot"]
    # On s'assure que les distances sont positives
    w_bot = max(0, b[1][0] - b[0][0]) 
    l_bot = max(0, b[2][1] - b[0][1])
    area_bot = 0.5 * w_bot * l_bot
    
    w_mid = (w_top + w_bot) / 2
    l_mid = (l_top + l_bot) / 2
    area_mid = 0.5 * w_mid * l_mid
    
    vol = (depth / 6) * (area_top + area_bot + 4 * area_mid)
    return vol, area_top, area_bot

# --- INTERFACE ---

st.title("🚜 Calculateur de Bassin")
st.markdown("Outil de dimensionnement - Triangle rectangle tronqué")

# 1. PARAMETRES
with st.sidebar:
    st.header("Entrées")
    
    # Dimensions
    val_W = st.number_input("Largeur X (Mur 1)", value=10.0, step=0.5, min_value=1.0)
    val_L = st.number_input("Longueur Y (Mur 2)", value=8.0, step=0.5, min_value=1.0)
    
    st.markdown("---")
    
    # Paramètres de creusement
    val_D = st.number_input("Profondeur (m)", value=2.0, step=0.1, min_value=0.1)
    val_Angle = st.slider("Angle Pente (°)", min_value=1, max_value=90, value=45)

    

# 2. CALCUL GEOMETRIE
geom = calculate_geometry(val_W, val_L, val_D, val_Angle)

# 3. VERIFICATION ERREUR
if not geom["is_valid"]:
    st.error("🛑 **STOP : CONFIGURATION IMPOSSIBLE**")
    st.markdown(f"""
    Le trou est **trop profond** pour la surface disponible avec cette pente.
    
    Les parois se rejoignent avant d'atteindre **{val_D}m** de profondeur.
    
    **Solutions :**
    1. Augmentez l'angle (raidir la pente).
    2. Réduisez la profondeur.
    3. Agrandissez la surface de départ.
    """)
    st.stop() # Arrête l'exécution ici, n'affiche pas le reste

# 4. CALCUL VOLUME (Si valide)
vol, a_top, a_bot = calculate_volume(geom, val_D)

# 5. RESULTATS KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Volume Déblai", f"{vol:.2f} m³", help="Volume total de terre à évacuer")
kpi2.metric("Surface Fond", f"{a_bot:.2f} m²", help="Surface plate au fond du trou")
kpi3.metric("Surface Haut", f"{a_top:.2f} m²", help="Emprise au sol totale")
kpi4.metric("Recul Pente", f"{geom['offset']:.2f} m", help="Distance horizontale prise par la pente")

# 6. VISUALISATION
tab1, tab2 = st.tabs(["Vue 3D", "Plan 2D"])

with tab1:
    verts = geom["top"] + geom["bot"]
    x = [v[0] for v in verts]
    y = [v[1] for v in verts]
    z = [v[2] for v in verts]
    
    fig_3d = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            alphahull=0, 
            opacity=0.6,
            color='#00a8ff',
            flatshading=True,
            hoverinfo='none'
        )
    ])
    
    # Arêtes
    def add_edge(fig, p1, p2, col="black"):
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines', line=dict(color=col, width=4), showlegend=False
        ))

    # Top (Bleu)
    add_edge(fig_3d, geom["top"][0], geom["top"][1], "blue")
    add_edge(fig_3d, geom["top"][1], geom["top"][2], "blue")
    add_edge(fig_3d, geom["top"][2], geom["top"][0], "blue")
    
    # Bot (Rouge)
    add_edge(fig_3d, geom["bot"][0], geom["bot"][1], "red")
    add_edge(fig_3d, geom["bot"][1], geom["bot"][2], "red")
    add_edge(fig_3d, geom["bot"][2], geom["bot"][0], "red")
    
    # Verticales (Noir)
    add_edge(fig_3d, geom["top"][0], geom["bot"][0])
    add_edge(fig_3d, geom["top"][1], geom["bot"][1])
    add_edge(fig_3d, geom["top"][2], geom["bot"][2])

    fig_3d.update_layout(scene=dict(aspectmode='data'), height=600, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    fig_2d = go.Figure()

    # Haut
    t = geom["top"]
    fig_2d.add_trace(go.Scatter(
        x=[t[0][0], t[1][0], t[2][0], t[0][0]],
        y=[t[0][1], t[1][1], t[2][1], t[0][1]],
        fill='toself', fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='blue'), name='Ouverture'
    ))
    
    # Bas
    b = geom["bot"]
    fig_2d.add_trace(go.Scatter(
        x=[b[0][0], b[1][0], b[2][0], b[0][0]],
        y=[b[0][1], b[1][1], b[2][1], b[0][1]],
        fill='toself', fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='red', dash='dash'), name='Fond'
    ))

    # Cotes
    if geom['offset'] > 0.05:
        fig_2d.add_annotation(x=geom['offset']/2, y=geom['offset']/2, text=f"Recul: {geom['offset']:.2f}", font=dict(color="green", size=10), showarrow=False)

    dist_x_bot = b[1][0] - b[0][0]
    dist_y_bot = b[2][1] - b[0][1]
    
    # Fleches cotes fond
    fig_2d.add_annotation(x=b[0][0] + dist_x_bot/2, y=b[0][1], text=f"L_fond: {dist_x_bot:.2f}m", showarrow=True, ay=25)
    fig_2d.add_annotation(x=b[0][0], y=b[0][1] + dist_y_bot/2, text=f"l_fond: {dist_y_bot:.2f}m", showarrow=True, ax=-40)

    fig_2d.update_layout(
        title="Vue de dessus",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=600
    )
    st.plotly_chart(fig_2d, use_container_width=True)
