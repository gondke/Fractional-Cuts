import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fractions import Fraction
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gomory's Fractional Cut Lab (ILP)",
    page_icon="📐",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem;}
    .step-badge {
        background-color: #DBEAFE; color: #1E40AF; padding: 4px 10px;
        border-radius: 6px; font-weight: 600; font-size: 0.9rem;
    }
    .cut-card {
        background-color: #F8FAFC; border-left: 4px solid #2563EB;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📐 Geometry & Simplex Lab: Gomory\'s Fractional Cuts</div>', unsafe_allow_html=True)
st.write(
    "Solve the LP Relaxation, inspect the unit lattice square around the real optimum, "
    "choose a Gomory cutting plane from the optimal simplex tableau, and observe how the feasible region tightens toward the integer optimum."
)

# --- EXACT RATIONAL ARITHMETIC HELPERS ---
def to_frac(val):
    if isinstance(val, Fraction):
        return val
    return Fraction(val).limit_denominator(100000)

def fmt_frac(val):
    f = to_frac(val)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"

def frac_part(val):
    """Strict positive fractional part f = val - floor(val), 0 <= f < 1."""
    f = to_frac(val)
    fl = math.floor(f)
    return f - fl

# --- EXACT SIMPLEX & DUAL SIMPLEX ENGINE ---
def solve_initial_lp(c, A, b):
    """
    Solves Maximize c^T x subject to A x <= b, x >= 0 (with b >= 0)
    using exact Fraction Simplex Method. Returns optimal tableau dictionary.
    """
    m = len(b)
    n = len(c)
    total_vars = n + m  # x1, x2, s1, s2, ...
    var_names = [f"x{j+1}" for j in range(n)] + [f"s{i+1}" for i in range(m)]
    
    c_all = [to_frac(cj) for cj in c] + [Fraction(0)] * m
    
    # Build tableau rows: [coeffs..., rhs]
    rows = []
    for i in range(m):
        row = [to_frac(A[i][j]) for j in range(n)] + [Fraction(1) if i == k else Fraction(0) for k in range(m)] + [to_frac(b[i])]
        rows.append(row)
        
    basis = [n + i for i in range(m)]  # Initial slack variables
    
    max_iters = 50
    for _ in range(max_iters):
        # Compute Zj and Cj - Zj
        zj = [sum(c_all[basis[i]] * rows[i][j] for i in range(m)) for j in range(total_vars)]
        cj_minus_zj = [c_all[j] - zj[j] for j in range(total_vars)]
        
        # Check optimality (all Cj - Zj <= 0)
        if all(val <= 0 for val in cj_minus_zj):
            break
            
        # Entering variable: most positive Cj - Zj
        enter_col = max(range(total_vars), key=lambda j: cj_minus_zj[j])
        
        # Leaving variable: minimum positive ratio
        ratios = []
        for i in range(m):
            if rows[i][enter_col] > 0:
                ratios.append((rows[i][-1] / rows[i][enter_col], i))
        if not ratios:
            return None  # Unbounded
            
        _, leave_row = min(ratios, key=lambda item: item[0])
        
        # Pivot
        pivot_val = rows[leave_row][enter_col]
        rows[leave_row] = [val / pivot_val for val in rows[leave_row]]
        for i in range(m):
            if i != leave_row:
                factor = rows[i][enter_col]
                rows[i] = [rows[i][j] - factor * rows[leave_row][j] for j in range(total_vars + 1)]
        basis[leave_row] = enter_col
        
    return {
        "rows": rows,
        "basis": basis,
        "c_all": c_all,
        "var_names": var_names,
        "n_orig": n
    }

def add_cut_and_reoptimize(tableau, cut_info):
    """
    Appends a selected Gomory cut row (in <= form with new slack variable)
    and restores feasibility/optimality using the Dual Simplex Method.
    """
    rows = [r[:] for r in tableau["rows"]]
    basis = tableau["basis"][:]
    c_all = tableau["c_all"][:]
    var_names = tableau["var_names"][:]
    
    m = len(rows)
    total_vars = len(var_names)
    new_slack_idx = total_vars
    new_slack_name = f"s{m + 1}"
    
    # Expand existing rows with 0 column for the new slack variable
    for i in range(m):
        rhs = rows[i][-1]
        rows[i] = rows[i][:-1] + [Fraction(0), rhs]
        
    # Construct the new cut row:
    # Gomory cut: sum_{j in NB} f_ij * x_j >= f_i0
    # Multiply by -1 and add new slack s_{m+1}:
    # -sum_{j in NB} f_ij * x_j + s_{m+1} = -f_i0
    new_row = [Fraction(0)] * (total_vars + 2)
    for j, f_val in cut_info["nb_coeffs"].items():
        new_row[j] = -f_val
    new_row[new_slack_idx] = Fraction(1)
    new_row[-1] = -cut_info["f_rhs"]
    
    rows.append(new_row)
    basis.append(new_slack_idx)
    c_all.append(Fraction(0))
    var_names.append(new_slack_name)
    
    m += 1
    total_vars += 1
    
    # Dual Simplex Iterations
    for _ in range(50):
        # Check primal feasibility (all RHS >= 0)
        if all(rows[i][-1] >= 0 for i in range(m)):
            break
            
        # Leaving variable: most negative RHS
        leave_row = min(range(m), key=lambda i: rows[i][-1])
        
        # Compute Zj - Cj (for maximization optimal tableau, Zj - Cj >= 0)
        zj_minus_cj = [
            sum(c_all[basis[i]] * rows[i][j] for i in range(m)) - c_all[j]
            for j in range(total_vars)
        ]
        
        # Dual ratio test: min { (Zj - Cj) / |a_rj| : a_rj < 0 }
        dual_ratios = []
        for j in range(total_vars):
            if rows[leave_row][j] < 0:
                dual_ratios.append((zj_minus_cj[j] / abs(rows[leave_row][j]), j))
                
        if not dual_ratios:
            return None  # Infeasible after cut
            
        _, enter_col = min(dual_ratios, key=lambda item: item[0])
        
        # Pivot
        pivot_val = rows[leave_row][enter_col]
        rows[leave_row] = [val / pivot_val for val in rows[leave_row]]
        for i in range(m):
            if i != leave_row:
                factor = rows[i][enter_col]
                rows[i] = [rows[i][j] - factor * rows[leave_row][j] for j in range(total_vars + 1)]
        basis[leave_row] = enter_col
        
    return {
        "rows": rows,
        "basis": basis,
        "c_all": c_all,
        "var_names": var_names,
        "n_orig": tableau["n_orig"]
    }

def extract_solution(tableau):
    """Returns (x1, x2), Z, and boolean is_integer."""
    m = len(tableau["rows"])
    sol = {j: Fraction(0) for j in range(len(tableau["var_names"]))}
    for i in range(m):
        sol[tableau["basis"][i]] = tableau["rows"][i][-1]
    x1, x2 = sol[0], sol[1]
    z = sum(tableau["c_all"][tableau["basis"][i]] * tableau["rows"][i][-1] for i in range(m))
    is_int = (x1.denominator == 1) and (x2.denominator == 1)
    return (x1, x2), z, is_int

def get_candidate_cuts(tableau, A_orig, b_orig):
    """
    Extracts all valid Gomory Fractional Cuts from rows of the optimal tableau
    where the basic variable has a non-zero fractional RHS.
    Projects each cut onto (x1, x2) space: alpha_1 * x1 + alpha_2 * x2 <= beta.
    """
    rows = tableau["rows"]
    basis = tableau["basis"]
    var_names = tableau["var_names"]
    m = len(rows)
    total_vars = len(var_names)
    non_basic = [j for j in range(total_vars) if j not in basis]
    
    # Express every variable (x1, x2, s1, s2, ...) in terms of (x1, x2):
    # x1 = 1*x1 + 0*x2 + 0
    # x2 = 0*x1 + 1*x2 + 0
    # s_k = b_k - A_k0*x1 - A_k1*x2   (for initial slacks k = 0, 1)
    # Any higher slack s_r (r >= 2) from previous cuts is expressed recursively via its tableau/cut history,
    # OR directly from the tableau row where it is non-basic!
    # Even simpler and exact: every variable j has an exact affine representation in (x1, x2)
    # obtained by solving for basic variables in terms of non-basic variables or tracking cut inequalities.
    return non_basic

def tableau_to_dataframe(tableau):
    """Formats the exact Fraction tableau into a clean pandas DataFrame."""
    rows = tableau["rows"]
    basis = tableau["basis"]
    c_all = tableau["c_all"]
    var_names = tableau["var_names"]
    m = len(rows)
    total_vars = len(var_names)
    
    data = []
    for i in range(m):
        b_idx = basis[i]
        row_dict = {
            "CB": fmt_frac(c_all[b_idx]),
            "Basic Var": var_names[b_idx],
        }
        for j, name in enumerate(var_names):
            row_dict[name] = fmt_frac(rows[i][j])
        row_dict["RHS (b)"] = fmt_frac(rows[i][-1])
        row_dict["Frac(b)"] = fmt_frac(frac_part(rows[i][-1]))
        data.append(row_dict)
        
    # Zj row
    zj_row = {"CB": "", "Basic Var": "Zj"}
    cj_zj_row = {"CB": "", "Basic Var": "Cj - Zj"}
    for j, name in enumerate(var_names):
        zj_val = sum(c_all[basis[i]] * rows[i][j] for i in range(m))
        zj_row[name] = fmt_frac(zj_val)
        cj_zj_row[name] = fmt_frac(c_all[j] - zj_val)
        
    z_total = sum(c_all[basis[i]] * rows[i][-1] for i in range(m))
    zj_row["RHS (b)"] = f"Z = {fmt_frac(z_total)}"
    zj_row["Frac(b)"] = ""
    cj_zj_row["RHS (b)"] = ""
    cj_zj_row["Frac(b)"] = ""
    
    data.append(zj_row)
    data.append(cj_zj_row)
    return pd.DataFrame(data)

# --- SIDEBAR: PROBLEM INPUTS ---
st.sidebar.header("1. Define 2-Variable ILP")

preset = st.sidebar.selectbox(
    "Select Problem Preset",
    [
        "Textbook Classic (Max 5x₁ + 4x₂)",
        "Example 2 (Max 7x₁ + 9x₂)",
        "Example 3 (Max 3x₁ + 4x₂)",
        "Custom Input"
    ]
)

if preset == "Textbook Classic (Max 5x₁ + 4x₂)":
    c_def = (5, 4)
    c1_def = (1, 1, 5)
    c2_def = (10, 6, 45)
elif preset == "Example 2 (Max 7x₁ + 9x₂)":
    c_def = (7, 9)
    c1_def = (-1, 3, 6)
    c2_def = (7, 1, 35)
elif preset == "Example 3 (Max 3x₁ + 4x₂)":
    c_def = (3, 4)
    c1_def = (2, 1, 6)
    c2_def = (2, 3, 9)
else:
    c_def = (4, 3)
    c1_def = (2, 3, 12)
    c2_def = (4, 1, 11)

st.sidebar.markdown("**Objective: Maximise $Z = c_1 x_1 + c_2 x_2$**")
col_c1, col_c2 = st.sidebar.columns(2)
c1 = col_c1.number_input("c₁", value=int(c_def[0]), step=1)
c2 = col_c2.number_input("c₂", value=int(c_def[1]), step=1)

st.sidebar.markdown("**Constraint 1: $a_{11}x_1 + a_{12}x_2 \\le b_1$**")
r1_1, r1_2, r1_b = st.sidebar.columns(3)
a11 = r1_1.number_input("a₁₁", value=int(c1_def[0]), step=1)
a12 = r1_2.number_input("a₁₂", value=int(c1_def[1]), step=1)
b1 = r1_b.number_input("b₁", value=int(c1_def[2]), min_value=1, step=1)

st.sidebar.markdown("**Constraint 2: $a_{21}x_1 + a_{22}x_2 \\le b_2$**")
r2_1, r2_2, r2_b = st.sidebar.columns(3)
a21 = r2_1.number_input("a₂₁", value=int(c2_def[0]), step=1)
a22 = r2_2.number_input("a₂₂", value=int(c2_def[1]), step=1)
b2 = r2_b.number_input("b₂", value=int(c2_def[2]), min_value=1, step=1)

# --- SESSION STATE MANAGEMENT ---
problem_signature = (c1, c2, a11, a12, b1, a21, a22, b2)

if "problem_sig" not in st.session_state or st.session_state.problem_sig != problem_signature:
    st.session_state.problem_sig = problem_signature
    init_tab = solve_initial_lp([c1, c2], [[a11, a12], [a21, a22]], [b1, b2])
    st.session_state.steps = [
        {
            "tableau": init_tab,
            "cuts_2d": [],  # List of tuples (alpha1, alpha2, beta, label)
            "slack_2d_map": {
                2: (-to_frac(a11), -to_frac(a12), to_frac(b1)),  # s1 = b1 - a11*x1 - a12*x2
                3: (-to_frac(a21), -to_frac(a22), to_frac(b2)),  # s2 = b2 - a21*x1 - a22*x2
            },
            "chosen_cut_desc": "Initial LP Relaxation (No Gomory Cuts Applied)"
        }
    ]

if st.sidebar.button("🔄 Reset to Initial LP Relaxation", use_container_width=True):
    init_tab = solve_initial_lp([c1, c2], [[a11, a12], [a21, a22]], [b1, b2])
    st.session_state.steps = [
        {
            "tableau": init_tab,
            "cuts_2d": [],
            "slack_2d_map": {
                2: (-to_frac(a11), -to_frac(a12), to_frac(b1)),
                3: (-to_frac(a21), -to_frac(a22), to_frac(b2)),
            },
            "chosen_cut_desc": "Initial LP Relaxation (No Gomory Cuts Applied)"
        }
    ]
    st.rerun()

# --- SIDEBAR: ZOOM CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.header("2. Lattice Square Zoom Controls")
zoom_mode = st.sidebar.radio(
    "Graph View Mode",
    ["Full Feasible Region", "Zoom into Unit Lattice Square"],
    index=0
)
zoom_padding = st.sidebar.slider(
    "Zoom Margin Around Lattice Square (units)",
    min_value=0.2, max_value=3.0, value=0.8, step=0.2,
    disabled=(zoom_mode == "Full Feasible Region"),
    help="Decrease margin to zoom in tightly on the 1×1 integer square containing the current fractional optimal point."
)

# --- CURRENT STATE INSPECTION ---
current_state = st.session_state.steps[-1]
curr_tab = current_state["tableau"]

if curr_tab is None:
    st.error("The problem is unbounded or infeasible with the current parameters. Please adjust the constraints in the sidebar.")
    st.stop()

(x1_opt, x2_opt), z_opt, is_integer = extract_solution(curr_tab)
step_idx = len(st.session_state.steps) - 1

# --- TOP STATUS BANNER ---
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("Current Stage", f"Stage {step_idx}" + (" (LP Relaxation)" if step_idx == 0 else f" (After {step_idx} Cut{'s' if step_idx>1 else ''})"))
col_s2.metric("Optimal x₁", fmt_frac(x1_opt))
col_s3.metric("Optimal x₂", fmt_frac(x2_opt))
col_s4.metric("Objective Value (Z)", fmt_frac(z_opt))

if is_integer:
    st.success(f"🎉 **Integer Optimal Solution Reached!** $(x_1^*, x_2^*) = ({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)})$ with $Z^* = {fmt_frac(z_opt)}$.")
else:
    lx, ly = math.floor(x1_opt), math.floor(x2_opt)
    st.info(
        f"📍 **Current Real Solution:** $({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)})$ is **not integer**. "
        f"It lies inside/on the unit lattice square $[{lx}, {lx+1}] \\times [{ly}, {ly+1}]$. "
        f"Switch to **Zoom into Unit Lattice Square** in the sidebar to inspect which corner lattice point it will approach!"
    )

# --- MAIN LAYOUT: GRAPH (LEFT) + SIMPLEX & CUT SELECTOR (RIGHT) ---
col_graph, col_right = st.columns([1.15, 1.15])

with col_graph:
    st.subheader("1. Geometric View & Unit Lattice Square")
    
    # Determine bounding box for full view
    x_candidates = [5.0]
    y_candidates = [5.0]
    if a11 > 0: x_candidates.append(float(b1) / float(a11))
    if a21 > 0: x_candidates.append(float(b2) / float(a21))
    if a12 > 0: y_candidates.append(float(b1) / float(a12))
    if a22 > 0: y_candidates.append(float(b2) / float(a22))
    x_candidates.append(float(x1_opt) + 1.5)
    y_candidates.append(float(x2_opt) + 1.5)
    
    full_xmax = max(x_candidates) * 1.12
    full_ymax = max(y_candidates) * 1.12
    
    # Unit lattice square around current optimum
    sq_x0 = math.floor(x1_opt)
    sq_y0 = math.floor(x2_opt)
    
    if zoom_mode == "Zoom into Unit Lattice Square":
        xmin = max(-0.2, sq_x0 - zoom_padding)
        xmax = sq_x0 + 1.0 + zoom_padding
        ymin = max(-0.2, sq_y0 - zoom_padding)
        ymax = sq_y0 + 1.0 + zoom_padding
    else:
        xmin, xmax = -0.3, full_xmax
        ymin, ymax = -0.3, full_ymax
        
    x_grid = np.linspace(max(0, xmin), xmax, 400)
    y_grid = np.linspace(max(0, ymin), ymax, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    
    # Shade Original LP Feasible Region
    orig_feas = (a11 * X + a12 * Y <= b1 + 1e-6) & (a21 * X + a22 * Y <= b2 + 1e-6) & (X >= -1e-6) & (Y >= -1e-6)
    ax.contourf(X, Y, orig_feas, levels=[0.5, 1.5], colors=['#CBD5E1'], alpha=0.45)
    
    # Shade Active Cut Feasible Region
    curr_feas = orig_feas.copy()
    for (al1, al2, bt, _) in current_state["cuts_2d"]:
        curr_feas &= (float(al1) * X + float(al2) * Y <= float(bt) + 1e-6)
    ax.contourf(X, Y, curr_feas, levels=[0.5, 1.5], colors=['#3B82F6'], alpha=0.32)
    
    # Highlight the Unit Lattice Square containing the current solution
    lattice_rect = patches.Rectangle(
        (sq_x0, sq_y0), 1.0, 1.0,
        linewidth=2.2, edgecolor='#F59E0B', facecolor='#FEF3C7', alpha=0.35,
        linestyle='--', zorder=2, label=f"Unit Lattice Square [{sq_x0},{sq_x0+1}]×[{sq_y0},{sq_y0+1}]"
    )
    ax.add_patch(lattice_rect)
    
    # Plot Original Constraints
    x_line = np.linspace(max(0, xmin), xmax, 300)
    if a12 != 0:
        ax.plot(x_line, (b1 - a11 * x_line) / a12, 'k-', lw=1.8, label=f"C1: {a11}x₁ + {a12}x₂ ≤ {b1}")
    elif a11 != 0:
        ax.axvline(b1 / a11, color='k', lw=1.8, label=f"C1: {a11}x₁ ≤ {b1}")
        
    if a22 != 0:
        ax.plot(x_line, (b2 - a21 * x_line) / a22, 'k--', lw=1.8, label=f"C2: {a21}x₁ + {a22}x₂ ≤ {b2}")
    elif a21 != 0:
        ax.axvline(b2 / a21, color='k', linestyle='--', lw=1.8, label=f"C2: {a21}x₁ ≤ {b2}")
        
    # Plot Applied Gomory Cutting Planes
    cut_palette = ['#DC2626', '#7C3AED', '#059669', '#DB2777', '#EA580C']
    for idx, (al1, al2, bt, lbl) in enumerate(current_state["cuts_2d"]):
        col = cut_palette[idx % len(cut_palette)]
        f_al1, f_al2, f_bt = float(al1), float(al2), float(bt)
        if abs(f_al2) > 1e-9:
            ax.plot(x_line, (f_bt - f_al1 * x_line) / f_al2, color=col, lw=2.4, linestyle='-.',
                    label=f"Cut {idx+1}: {fmt_frac(al1)}x₁ + {fmt_frac(al2)}x₂ ≤ {fmt_frac(bt)}")
        elif abs(f_al1) > 1e-9:
            ax.axvline(f_bt / f_al1, color=col, lw=2.4, linestyle='-.',
                       label=f"Cut {idx+1}: {fmt_frac(al1)}x₁ ≤ {fmt_frac(bt)}")
            
    # Plot Integer Lattice Points
    ix_vals = np.arange(max(0, int(math.floor(xmin))), int(math.ceil(xmax)) + 1)
    iy_vals = np.arange(max(0, int(math.floor(ymin))), int(math.ceil(ymax)) + 1)
    IX, IY = np.meshgrid(ix_vals, iy_vals)
    IX_f, IY_f = IX.flatten(), IY.flatten()
    
    int_feas_mask = (a11 * IX_f + a12 * IY_f <= b1 + 1e-6) & (a21 * IX_f + a22 * IY_f <= b2 + 1e-6)
    for (al1, al2, bt, _) in current_state["cuts_2d"]:
        int_feas_mask &= (float(al1) * IX_f + float(al2) * IY_f <= float(bt) + 1e-6)
        
    ax.scatter(IX_f[~int_feas_mask], IY_f[~int_feas_mask], color='#94A3B8', s=30, alpha=0.55, zorder=3, label="Infeasible Integer Points")
    ax.scatter(IX_f[int_feas_mask], IY_f[int_feas_mask], color='#0F172A', s=75, zorder=5, label="Feasible Integer Lattice Points")
    
    # Annotate integer coordinates when zoomed in
    if zoom_mode == "Zoom into Unit Lattice Square":
        for px, py in zip(IX_f, IY_f):
            if xmin <= px <= xmax and ymin <= py <= ymax:
                ax.annotate(f"({px},{py})", (px, py), textcoords="offset points", xytext=(6, 6),
                            fontsize=9, fontweight='bold', color='#1E293B')
                
    # Plot Initial LP Optimum (if cuts have been added)
    init_x1, init_x2 = extract_solution(st.session_state.steps[0]["tableau"])[0]
    if step_idx > 0:
        ax.scatter(float(init_x1), float(init_x2), color='#991B1B', s=110, marker='x', zorder=6,
                   label=f"Initial LP Opt ({fmt_frac(init_x1)}, {fmt_frac(init_x2)})")
        
    # Plot Current Optimum
    curr_marker = '*' if is_integer else 'D'
    curr_color = '#16A34A' if is_integer else '#DC2626'
    curr_lbl = "Integer Optimum" if is_integer else "Current Real Optimum"
    ax.scatter(float(x1_opt), float(x2_opt), color=curr_color, s=190, marker=curr_marker, zorder=7,
               label=f"{curr_lbl} ({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)})")
    
    # Iso-profit objective line through current optimum
    if c2 != 0:
        ax.plot(x_line, (float(z_opt) - c1 * x_line) / c2, color='#059669', linestyle=':', lw=1.6,
                label=f"Iso-Profit Z = {fmt_frac(z_opt)}")
        
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x₁", fontsize=11, fontweight='bold')
    ax.set_ylabel("x₂", fontsize=11, fontweight='bold')
    ax.set_title(
        "Zoomed Unit Lattice Square View" if zoom_mode == "Zoom into Unit Lattice Square" else "Full Feasible Region & Cutting Planes",
        fontsize=12, fontweight='bold'
    )
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.92)
    st.pyplot(fig)

with col_right:
    st.subheader("2. Optimal Simplex Table (Exact p/q Form)")
    df_tab = tableau_to_dataframe(curr_tab)
    st.dataframe(df_tab, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("3. Gomory Cutting Planes from Current Table")
    
    if is_integer:
        st.success("All basic decision variables are integers. No further Gomory cuts are required!")
    else:
        # Build all candidate cuts from rows with fractional RHS
        rows = curr_tab["rows"]
        basis = curr_tab["basis"]
        var_names = curr_tab["var_names"]
        m = len(rows)
        total_vars = len(var_names)
        non_basic = [j for j in range(total_vars) if j not in basis]
        slack_map = current_state["slack_2d_map"]
        
        candidate_cuts = []
        for i in range(m):
            rhs_val = rows[i][-1]
            f_rhs = frac_part(rhs_val)
            if f_rhs > 0:
                b_var_name = var_names[basis[i]]
                nb_coeffs = {}
                nb_terms_str = []
                
                # Project cut onto (x1, x2) space:
                # sum_{j in NB} f_ij * var_j >= f_rhs
                # Substitute var_0 = x1, var_1 = x2, and slack var_j = c_x1*x1 + c_x2*x2 + const
                lhs_x1 = Fraction(0)
                lhs_x2 = Fraction(0)
                lhs_const = Fraction(0)
                
                for j in non_basic:
                    f_ij = frac_part(rows[i][j])
                    nb_coeffs[j] = f_ij
                    if f_ij > 0:
                        nb_terms_str.append(f"({fmt_frac(f_ij)}){var_names[j]}")
                        if j == 0:
                            lhs_x1 += f_ij
                        elif j == 1:
                            lhs_x2 += f_ij
                        elif j in slack_map:
                            sx1, sx2, sconst = slack_map[j]
                            lhs_x1 += f_ij * sx1
                            lhs_x2 += f_ij * sx2
                            lhs_const += f_ij * sconst
                            
                # Inequality is: lhs_x1 * x1 + lhs_x2 * x2 + lhs_const >= f_rhs
                # Convert to <= form: (-lhs_x1)*x1 + (-lhs_x2)*x2 <= lhs_const - f_rhs
                alpha_1 = -lhs_x1
                alpha_2 = -lhs_x2
                beta = lhs_const - f_rhs
                
                nb_ineq_str = (" + ".join(nb_terms_str) if nb_terms_str else "0") + f" ≥ {fmt_frac(f_rhs)}"
                xy_ineq_str = f"({fmt_frac(alpha_1)})x₁ + ({fmt_frac(alpha_2)})x₂ ≤ {fmt_frac(beta)}"
                
                candidate_cuts.append({
                    "row_idx": i,
                    "source_var": b_var_name,
                    "rhs_val": rhs_val,
                    "f_rhs": f_rhs,
                    "nb_coeffs": nb_coeffs,
                    "nb_ineq_str": nb_ineq_str,
                    "xy_ineq_str": xy_ineq_str,
                    "alpha_1": alpha_1,
                    "alpha_2": alpha_2,
                    "beta": beta
                })
                
        if not candidate_cuts:
            st.warning("No fractional rows available to generate a Gomory cut.")
        else:
            st.write("Select which row's **Gomory Fractional Cut** you want to add to the problem:")
            for c_idx, cut in enumerate(candidate_cuts):
                st.markdown(f"""
                <div class="cut-card">
                    <b>Option {c_idx + 1} — Source Row: <code>{cut['source_var']}</code></b> (RHS = {fmt_frac(cut['rhs_val'])}, Fractional part $f_0$ = {fmt_frac(cut['f_rhs'])})<br>
                    • <b>Non-Basic Cut Form:</b> <code>{cut['nb_ineq_str']}</code><br>
                    • <b>2D Geometric Line:</b> <code>{cut['xy_ineq_str']}</code>
                </div>
                """, unsafe_allow_html=True)
                
            cut_labels = [
                f"Option {idx+1}: Row {c['source_var']}  →  {c['xy_ineq_str']}"
                for idx, c in enumerate(candidate_cuts)
            ]
            selected_idx = st.radio("Choose Cutting Plane to Apply:", range(len(candidate_cuts)), format_func=lambda idx: cut_labels[idx])
            
            col_btn1, col_btn2 = st.columns([1.2, 1])
            if col_btn1.button("✂️ Apply Chosen Cutting Plane", type="primary", use_container_width=True):
                chosen_cut = candidate_cuts[selected_idx]
                new_tab = add_cut_and_reoptimize(curr_tab, chosen_cut)
                
                # Register the newly created slack variable in slack_2d_map:
                # alpha_1*x1 + alpha_2*x2 + s_new = beta  =>  s_new = (-alpha_1)*x1 + (-alpha_2)*x2 + beta
                new_slack_idx = len(curr_tab["var_names"])
                new_slack_map = dict(slack_map)
                new_slack_map[new_slack_idx] = (-chosen_cut["alpha_1"], -chosen_cut["alpha_2"], chosen_cut["beta"])
                
                new_cuts_2d = list(current_state["cuts_2d"]) + [
                    (chosen_cut["alpha_1"], chosen_cut["alpha_2"], chosen_cut["beta"], chosen_cut["xy_ineq_str"])
                ]
                
                st.session_state.steps.append({
                    "tableau": new_tab,
                    "cuts_2d": new_cuts_2d,
                    "slack_2d_map": new_slack_map,
                    "chosen_cut_desc": chosen_cut["xy_ineq_str"]
                })
                st.rerun()
                
            if step_idx > 0:
                if col_btn2.button("⬅️ Undo Last Cut", use_container_width=True):
                    st.session_state.steps.pop()
                    st.rerun()
