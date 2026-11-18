import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    .main-header {font-size: 2.0rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem;}
    .cut-card {
        background-color: #F8FAFC; border-left: 4px solid #2563EB;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
    }
    .lattice-box {
        background-color: #FFFBEB; border: 1px solid #F59E0B;
        padding: 10px 14px; border-radius: 6px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
    total_vars = n + m
    var_names = [f"x{j+1}" for j in range(n)] + [f"s{i+1}" for i in range(m)]
    c_all = [to_frac(cj) for cj in c] + [Fraction(0)] * m

    rows = []
    for i in range(m):
        row = (
            [to_frac(A[i][j]) for j in range(n)]
            + [Fraction(1) if i == k else Fraction(0) for k in range(m)]
            + [to_frac(b[i])]
        )
        rows.append(row)

    basis = [n + i for i in range(m)]

    for _ in range(60):
        zj = [sum(c_all[basis[i]] * rows[i][j] for i in range(m)) for j in range(total_vars)]
        cj_minus_zj = [c_all[j] - zj[j] for j in range(total_vars)]

        if all(val <= 0 for val in cj_minus_zj):
            break

        enter_col = max(range(total_vars), key=lambda j: cj_minus_zj[j])

        ratios = []
        for i in range(m):
            if rows[i][enter_col] > 0:
                ratios.append((rows[i][-1] / rows[i][enter_col], i))
        if not ratios:
            return None

        _, leave_row = min(ratios, key=lambda item: item[0])

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

    for i in range(m):
        rhs = rows[i][-1]
        rows[i] = rows[i][:-1] + [Fraction(0), rhs]

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

    for _ in range(60):
        if all(rows[i][-1] >= 0 for i in range(m)):
            break

        leave_row = min(range(m), key=lambda i: rows[i][-1])

        zj_minus_cj = [
            sum(c_all[basis[i]] * rows[i][j] for i in range(m)) - c_all[j]
            for j in range(total_vars)
        ]

        dual_ratios = []
        for j in range(total_vars):
            if rows[leave_row][j] < 0:
                dual_ratios.append((zj_minus_cj[j] / abs(rows[leave_row][j]), j))

        if not dual_ratios:
            return None

        _, enter_col = min(dual_ratios, key=lambda item: item[0])

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
    m = len(tableau["rows"])
    sol = {j: Fraction(0) for j in range(len(tableau["var_names"]))}
    for i in range(m):
        sol[tableau["basis"][i]] = tableau["rows"][i][-1]
    x1, x2 = sol[0], sol[1]
    z = sum(tableau["c_all"][tableau["basis"][i]] * tableau["rows"][i][-1] for i in range(m))
    is_int = (x1.denominator == 1) and (x2.denominator == 1)
    return (x1, x2), z, is_int

def build_candidate_cuts(tableau, slack_map):
    rows = tableau["rows"]
    basis = tableau["basis"]
    var_names = tableau["var_names"]
    m = len(rows)
    total_vars = len(var_names)
    non_basic = [j for j in range(total_vars) if j not in basis]

    candidate_cuts = []
    for i in range(m):
        rhs_val = rows[i][-1]
        f_rhs = frac_part(rhs_val)
        if f_rhs > 0:
            b_var_name = var_names[basis[i]]
            nb_coeffs = {}
            nb_terms_str = []

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
    return candidate_cuts

def apply_cut_action(chosen_cut):
    current_state = st.session_state.steps[-1]
    curr_tab = current_state["tableau"]
    slack_map = current_state["slack_2d_map"]

    new_tab = add_cut_and_reoptimize(curr_tab, chosen_cut)
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

def tableau_to_dataframe(tableau):
    rows = tableau["rows"]
    basis = tableau["basis"]
    c_all = tableau["c_all"]
    var_names = tableau["var_names"]
    m = len(rows)

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

# --- PERSISTENT SIDEBAR: PROBLEM SETUP ---
st.sidebar.header("⚙️ 2-Variable ILP Problem")

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
    c_def, c1_def, c2_def = (5, 4), (1, 1, 5), (10, 6, 45)
elif preset == "Example 2 (Max 7x₁ + 9x₂)":
    c_def, c1_def, c2_def = (7, 9), (-1, 3, 6), (7, 1, 35)
elif preset == "Example 3 (Max 3x₁ + 4x₂)":
    c_def, c1_def, c2_def = (3, 4), (2, 1, 6), (2, 3, 9)
else:
    c_def, c1_def, c2_def = (4, 3), (2, 3, 12), (4, 1, 11)

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

# --- INITIALIZE SESSION STATE ---
problem_signature = (c1, c2, a11, a12, b1, a21, a22, b2)

if "problem_sig" not in st.session_state or st.session_state.problem_sig != problem_signature:
    st.session_state.problem_sig = problem_signature
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
    st.session_state.zoom_margin = None  # None means Full Feasible Region

col_rst1, col_rst2 = st.sidebar.columns(2)
if col_rst1.button("🔄 Reset Problem", use_container_width=True):
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
    st.session_state.zoom_margin = None
    st.rerun()

if len(st.session_state.steps) > 1:
    if col_rst2.button("⬅️ Undo Cut", use_container_width=True):
        st.session_state.steps.pop()
        st.rerun()

# --- HELPER: TOP STATUS BANNER ---
def render_status_banner():
    current_state = st.session_state.steps[-1]
    curr_tab = current_state["tableau"]
    if curr_tab is None:
        st.error("The problem is unbounded or infeasible with the current constraints.")
        st.stop()

    (x1_opt, x2_opt), z_opt, is_integer = extract_solution(curr_tab)
    step_idx = len(st.session_state.steps) - 1

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Current Iteration", f"Stage {step_idx}" + (" (LP Relaxation)" if step_idx == 0 else f" (After {step_idx} Cut{'s' if step_idx>1 else ''})"))
    c_m2.metric("Optimal x₁", fmt_frac(x1_opt))
    c_m3.metric("Optimal x₂", fmt_frac(x2_opt))
    c_m4.metric("Objective Value (Z)", fmt_frac(z_opt))

    if is_integer:
        st.success(f"🎉 **Final Integer Optimal Solution Reached!** $(x_1^*, x_2^*) = ({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)})$ with $Z^* = {fmt_frac(z_opt)}$.")
    else:
        lx, ly = math.floor(x1_opt), math.floor(x2_opt)
        st.info(
            f"📍 **Non-Integer Real Solution:** $({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)})$ lies inside/on the unit lattice square "
            f"$[{lx}, {lx+1}] \\times [{ly}, {ly+1}]$. Use the zoom controls or choose a Gomory cutting plane to approach the integer optimum."
        )
    return current_state, curr_tab, (x1_opt, x2_opt), z_opt, is_integer, step_idx

# ==============================================================================
# PAGE 1: GEOMETRIC & LATTICE ZOOM VIEW
# ==============================================================================
def page_geometry():
    st.markdown('<div class="main-header">📐 Page 1: Geometric Feasible Region & Lattice Square Zoom</div>', unsafe_allow_html=True)
    current_state, curr_tab, (x1_opt, x2_opt), z_opt, is_integer, step_idx = render_status_banner()

    # Unit lattice square enclosing current solution
    sq_x0 = math.floor(x1_opt)
    sq_y0 = math.floor(x2_opt)

    # Interactive Zoom Control Bar
    st.markdown("### 🔍 Lattice Square Zoom & Pan Controls")
    z_col1, z_col2, z_col3, z_col4, z_col5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.5])

    if z_col1.button("🌐 Full Feasible Region", use_container_width=True):
        st.session_state.zoom_margin = None
        st.rerun()

    if z_col2.button("🎯 Focus Lattice Square", use_container_width=True):
        st.session_state.zoom_margin = 0.6
        st.rerun()

    if z_col3.button("➕ Zoom In (Tighter)", use_container_width=True):
        curr_m = 1.4 if st.session_state.zoom_margin is None else st.session_state.zoom_margin
        st.session_state.zoom_margin = max(0.15, round(curr_m - 0.35, 2))
        st.rerun()

    if z_col4.button("➖ Zoom Out (Wider)", use_container_width=True):
        curr_m = 0.6 if st.session_state.zoom_margin is None else st.session_state.zoom_margin
        new_m = round(curr_m + 0.5, 2)
        st.session_state.zoom_margin = None if new_m > 4.0 else new_m
        st.rerun()

    margin_val = z_col5.slider(
        "Zoom Margin around Unit Square",
        min_value=0.15, max_value=4.0,
        value=0.6 if st.session_state.zoom_margin is None else float(st.session_state.zoom_margin),
        step=0.05,
        label_visibility="collapsed"
    )
    if st.session_state.zoom_margin is not None and abs(margin_val - st.session_state.zoom_margin) > 1e-4:
        st.session_state.zoom_margin = margin_val
        st.rerun()

    # Compute Bounding Box
    x_candidates = [5.0, float(x1_opt) + 1.5]
    y_candidates = [5.0, float(x2_opt) + 1.5]
    if a11 > 0: x_candidates.append(float(b1) / float(a11))
    if a21 > 0: x_candidates.append(float(b2) / float(a21))
    if a12 > 0: y_candidates.append(float(b1) / float(a12))
    if a22 > 0: y_candidates.append(float(b2) / float(a22))

    full_xmax = max(x_candidates) * 1.15
    full_ymax = max(y_candidates) * 1.15

    if st.session_state.zoom_margin is None:
        xmin, xmax = -0.3, full_xmax
        ymin, ymax = -0.3, full_ymax
        view_title = "Full Feasible Region (Drag box on graph to zoom, or click 'Focus Lattice Square')"
    else:
        m_pad = st.session_state.zoom_margin
        xmin = max(-0.25, sq_x0 - m_pad)
        xmax = sq_x0 + 1.0 + m_pad
        ymin = max(-0.25, sq_y0 - m_pad)
        ymax = sq_y0 + 1.0 + m_pad
        view_title = f"Zoomed View: Unit Lattice Square [{sq_x0}, {sq_x0+1}] × [{sq_y0}, {sq_y0+1}] (Margin = ±{m_pad})"

    # Build Interactive Plotly Figure
    fig = go.Figure()

    # Grid for shaded feasible regions
    gx = np.linspace(max(0, xmin), xmax, 220)
    gy = np.linspace(max(0, ymin), ymax, 220)
    GX, GY = np.meshgrid(gx, gy)

    orig_mask = ((a11 * GX + a12 * GY <= b1 + 1e-6) & (a21 * GX + a22 * GY <= b2 + 1e-6)).astype(float)
    fig.add_trace(go.Contour(
        x=gx, y=gy, z=orig_mask,
        contours=dict(start=0.5, end=1.5, size=1, coloring="fill"),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(148, 163, 184, 0.28)"]],
        showscale=False, hoverinfo="skip", name="Original LP Region"
    ))

    if current_state["cuts_2d"]:
        curr_mask = orig_mask.copy()
        for (al1, al2, bt, _) in current_state["cuts_2d"]:
            curr_mask *= (float(al1) * GX + float(al2) * GY <= float(bt) + 1e-6).astype(float)
        fig.add_trace(go.Contour(
            x=gx, y=gy, z=curr_mask,
            contours=dict(start=0.5, end=1.5, size=1, coloring="fill"),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(59, 130, 246, 0.30)"]],
            showscale=False, hoverinfo="skip", name="Tightened Feasible Region"
        ))

    # Highlight Unit Lattice Square
    fig.add_shape(
        type="rect",
        x0=sq_x0, y0=sq_y0, x1=sq_x0 + 1, y1=sq_y0 + 1,
        line=dict(color="#F59E0B", width=2.5, dash="dash"),
        fillcolor="rgba(254, 243, 199, 0.32)",
        layer="below"
    )
    fig.add_annotation(
        x=sq_x0 + 0.5, y=sq_y0 + 1.03,
        text=f"Enclosing Lattice Square [{sq_x0},{sq_x0+1}]×[{sq_y0},{sq_y0+1}]",
        showarrow=False, font=dict(color="#B45309", size=11)
    )

    # Original Constraint Lines
    x_line = np.linspace(max(0, xmin), xmax, 250)
    if a12 != 0:
        fig.add_trace(go.Scatter(
            x=x_line, y=(b1 - a11 * x_line) / a12, mode="lines",
            line=dict(color="#1E293B", width=2),
            name=f"Constraint 1: {a11}x₁ + {a12}x₂ ≤ {b1}"
        ))
    elif a11 != 0:
        fig.add_vline(x=b1 / a11, line_color="#1E293B", line_width=2)

    if a22 != 0:
        fig.add_trace(go.Scatter(
            x=x_line, y=(b2 - a21 * x_line) / a22, mode="lines",
            line=dict(color="#475569", width=2, dash="dash"),
            name=f"Constraint 2: {a21}x₁ + {a22}x₂ ≤ {b2}"
        ))
    elif a21 != 0:
        fig.add_vline(x=b2 / a21, line_color="#475569", line_width=2, line_dash="dash")

    # Gomory Cutting Planes
    cut_colors = ["#DC2626", "#7C3AED", "#059669", "#DB2777", "#EA580C"]
    for idx, (al1, al2, bt, _) in enumerate(current_state["cuts_2d"]):
        col = cut_colors[idx % len(cut_colors)]
        f_al1, f_al2, f_bt = float(al1), float(al2), float(bt)
        if abs(f_al2) > 1e-9:
            fig.add_trace(go.Scatter(
                x=x_line, y=(f_bt - f_al1 * x_line) / f_al2, mode="lines",
                line=dict(color=col, width=2.8, dash="dashdot"),
                name=f"Gomory Cut {idx+1}: {fmt_frac(al1)}x₁ + {fmt_frac(al2)}x₂ ≤ {fmt_frac(bt)}"
            ))
        elif abs(f_al1) > 1e-9:
            fig.add_vline(x=f_bt / f_al1, line_color=col, line_width=2.8, line_dash="dashdot")

    # Integer Lattice Points
    ix_vals = np.arange(max(0, int(math.floor(xmin))), int(math.ceil(xmax)) + 1)
    iy_vals = np.arange(max(0, int(math.floor(ymin))), int(math.ceil(ymax)) + 1)
    IX, IY = np.meshgrid(ix_vals, iy_vals)
    IX_f, IY_f = IX.flatten(), IY.flatten()

    int_feas_mask = (a11 * IX_f + a12 * IY_f <= b1 + 1e-6) & (a21 * IX_f + a22 * IY_f <= b2 + 1e-6)
    for (al1, al2, bt, _) in current_state["cuts_2d"]:
        int_feas_mask &= (float(al1) * IX_f + float(al2) * IY_f <= float(bt) + 1e-6)

    # Infeasible Integer Lattice Points
    fig.add_trace(go.Scatter(
        x=IX_f[~int_feas_mask], y=IY_f[~int_feas_mask],
        mode="markers",
        marker=dict(color="#94A3B8", size=7, symbol="circle-open"),
        name="Infeasible Integer Points",
        hovertemplate="Integer Point (%{x}, %{y}) — Infeasible<extra></extra>"
    ))

    # Feasible Integer Lattice Points (with Z value hover)
    feas_ix = IX_f[int_feas_mask]
    feas_iy = IY_f[int_feas_mask]
    feas_z = [c1 * int(px) + c2 * int(py) for px, py in zip(feas_ix, feas_iy)]
    feas_labels = [f"({int(px)}, {int(py)})" for px, py in zip(feas_ix, feas_iy)]

    fig.add_trace(go.Scatter(
        x=feas_ix, y=feas_iy,
        mode="markers+text" if st.session_state.zoom_margin is not None else "markers",
        text=feas_labels,
        textposition="top right",
        textfont=dict(size=11, color="#0F172A"),
        marker=dict(color="#0F172A", size=10, symbol="circle"),
        customdata=feas_z,
        name="Feasible Integer Lattice Points",
        hovertemplate="Feasible Lattice Point (%{x}, %{y})<br>Objective Z = %{customdata}<extra></extra>"
    ))

    # Initial LP Optimum marker (when cuts exist)
    init_x1, init_x2 = extract_solution(st.session_state.steps[0]["tableau"])[0]
    if step_idx > 0:
        fig.add_trace(go.Scatter(
            x=[float(init_x1)], y=[float(init_x2)],
            mode="markers",
            marker=dict(color="#991B1B", size=12, symbol="x"),
            name=f"Initial LP Opt ({fmt_frac(init_x1)}, {fmt_frac(init_x2)})"
        ))

    # Current Optimum marker
    fig.add_trace(go.Scatter(
        x=[float(x1_opt)], y=[float(x2_opt)],
        mode="markers+text",
        text=[f"  ({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)})"],
        textposition="bottom right",
        textfont=dict(size=12, color="#16A34A" if is_integer else "#DC2626"),
        marker=dict(
            color="#16A34A" if is_integer else "#DC2626",
            size=15,
            symbol="star" if is_integer else "diamond"
        ),
        name=("Final Integer Optimum" if is_integer else "Current Real Optimum")
        + f" ({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)})"
    ))

    # Iso-profit line
    if c2 != 0:
        fig.add_trace(go.Scatter(
            x=x_line, y=(float(z_opt) - c1 * x_line) / c2,
            mode="lines",
            line=dict(color="#10B981", width=1.6, dash="dot"),
            name=f"Iso-Profit Z = {fmt_frac(z_opt)}"
        ))

    fig.update_layout(
        title=view_title,
        xaxis=dict(title="x₁", range=[xmin, xmax], dtick=1 if st.session_state.zoom_margin is not None else None, gridcolor="#E2E8F0"),
        yaxis=dict(title="x₂", range=[ymin, ymax], dtick=1 if st.session_state.zoom_margin is not None else None, gridcolor="#E2E8F0"),
        height=620,
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.88)"),
        dragmode="zoom"
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

    # Quick Action Panel below Graph to Choose & Apply Cut Directly
    if not is_integer:
        st.markdown("---")
        st.subheader("✂️ Apply a Gomory Cutting Plane & Watch the Solution Shift")
        candidate_cuts = build_candidate_cuts(curr_tab, current_state["slack_2d_map"])
        if candidate_cuts:
            c_sel1, c_sel2 = st.columns([2.5, 1])
            cut_options = [
                f"Cut from Row {c['source_var']} (RHS = {fmt_frac(c['rhs_val'])})  →  {c['xy_ineq_str']}"
                for c in candidate_cuts
            ]
            chosen_idx = c_sel1.selectbox("Select Cutting Plane (view full Simplex Tableaux on Page 2):", range(len(candidate_cuts)), format_func=lambda i: cut_options[i])
            if c_sel2.button("✂️ Apply Selected Cut", type="primary", use_container_width=True):
                apply_cut_action(candidate_cuts[chosen_idx])
                st.rerun()

# ==============================================================================
# PAGE 2: SIMPLEX TABLES & CUTTING PLANE DERIVATIONS
# ==============================================================================
def page_simplex():
    st.markdown('<div class="main-header">📊 Page 2: Optimal Simplex Tables (p/q Form) & Cut Selection</div>', unsafe_allow_html=True)
    current_state, curr_tab, (x1_opt, x2_opt), z_opt, is_integer, step_idx = render_status_banner()

    st.subheader(f"Current Optimal Simplex Tableau — Stage {step_idx} (Exact Fractional Form)")
    df_tab = tableau_to_dataframe(curr_tab)
    st.dataframe(df_tab, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Constructed Gomory Fractional Cutting Planes from Current Tableau")

    if is_integer:
        st.success("All basic decision variables have integer values. No further cutting planes can be generated.")
    else:
        candidate_cuts = build_candidate_cuts(curr_tab, current_state["slack_2d_map"])
        if not candidate_cuts:
            st.warning("No rows with fractional right-hand sides are available.")
        else:
            st.write("Each row in the optimal simplex tableau with a fractional RHS ($f_0 > 0$) yields a valid Gomory cutting plane:")
            for idx, cut in enumerate(candidate_cuts):
                st.markdown(f"""
                <div class="cut-card">
                    <b>Option {idx + 1} — Derived from Basic Variable Row <code>{cut['source_var']}</code></b><br>
                    • <b>Basic Value:</b> <code>{fmt_frac(cut['rhs_val'])}</code> &nbsp;|&nbsp; <b>Fractional Part ($f_0$):</b> <code>{fmt_frac(cut['f_rhs'])}</code><br>
                    • <b>Non-Basic Inequality ($\\sum f_{{ij}} x_j \\ge f_0$):</b> <code>{cut['nb_ineq_str']}</code><br>
                    • <b>2D Geometric Cutting Line:</b> <code>{cut['xy_ineq_str']}</code>
                </div>
                """, unsafe_allow_html=True)

            col_pick, col_apply = st.columns([2.5, 1])
            selected_cut_idx = col_pick.radio(
                "Choose a Cutting Plane to Add to the Simplex Tableau:",
                range(len(candidate_cuts)),
                format_func=lambda idx: f"Option {idx+1} (Row {candidate_cuts[idx]['source_var']}):  {candidate_cuts[idx]['xy_ineq_str']}"
            )
            if col_apply.button("✂️ Add Cut & Re-Optimize (Dual Simplex)", type="primary", use_container_width=True):
                apply_cut_action(candidate_cuts[selected_cut_idx])
                st.rerun()

    # Historical Tableaux Accordion
    if step_idx > 0:
        st.markdown("---")
        st.subheader("📜 Previous Iteration Tableaux History")
        for past_idx in range(step_idx - 1, -1, -1):
            past_state = st.session_state.steps[past_idx]
            (px1, px2), pz, _ = extract_solution(past_state["tableau"])
            with st.expander(f"Stage {past_idx}: {past_state['chosen_cut_desc']}  |  Solution: ({fmt_frac(px1)}, {fmt_frac(px2)}), Z = {fmt_frac(pz)}"):
                st.dataframe(tableau_to_dataframe(past_state["tableau"]), use_container_width=True, hide_index=True)

# --- MULTI-PAGE NAVIGATION ---
nav = st.navigation([
    st.Page(page_geometry, title="1. Geometric & Lattice Zoom View", icon="📐"),
    st.Page(page_simplex, title="2. Simplex Tables & Cut Derivations", icon="📊")
])
nav.run()
