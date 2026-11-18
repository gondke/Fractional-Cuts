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
    .lattice-summary-card {
        background-color: #FFFBEB; border-left: 4px solid #F59E0B;
        padding: 10px 14px; border-radius: 6px; margin-top: 6px; font-size: 0.92rem;
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
    st.session_state.zoom_margin = None

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
            f"$[{lx}, {lx+1}] \\times [{ly}, {ly+1}]$. Inspect the isolated lattice squares on **Page 3** or apply a Gomory cutting plane!"
        )
    return current_state, curr_tab, (x1_opt, x2_opt), z_opt, is_integer, step_idx

# ==============================================================================
# PAGE 1: GEOMETRIC & LATTICE ZOOM VIEW
# ==============================================================================
def page_geometry():
    st.markdown('<div class="main-header">📐 Page 1: Geometric Feasible Region & Lattice Square Zoom</div>', unsafe_allow_html=True)
    current_state, curr_tab, (x1_opt, x2_opt), z_opt, is_integer, step_idx = render_status_banner()

    sq_x0 = math.floor(x1_opt)
    sq_y0 = math.floor(x2_opt)

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

    fig = go.Figure()

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

    ix_vals = np.arange(max(0, int(math.floor(xmin))), int(math.ceil(xmax)) + 1)
    iy_vals = np.arange(max(0, int(math.floor(ymin))), int(math.ceil(ymax)) + 1)
    IX, IY = np.meshgrid(ix_vals, iy_vals)
    IX_f, IY_f = IX.flatten(), IY.flatten()

    int_feas_mask = (a11 * IX_f + a12 * IY_f <= b1 + 1e-6) & (a21 * IX_f + a22 * IY_f <= b2 + 1e-6)
    for (al1, al2, bt, _) in current_state["cuts_2d"]:
        int_feas_mask &= (float(al1) * IX_f + float(al2) * IY_f <= float(bt) + 1e-6)

    fig.add_trace(go.Scatter(
        x=IX_f[~int_feas_mask], y=IY_f[~int_feas_mask],
        mode="markers",
        marker=dict(color="#94A3B8", size=7, symbol="circle-open"),
        name="Infeasible Integer Points",
        hovertemplate="Integer Point (%{x}, %{y}) — Infeasible<extra></extra>"
    ))

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

    # Draw Step-by-Step Trajectory Arrows on the Main Plot
    if step_idx > 0:
        pts_x = [float(extract_solution(s["tableau"])[0][0]) for s in st.session_state.steps]
        pts_y = [float(extract_solution(s["tableau"])[0][1]) for s in st.session_state.steps]
        fig.add_trace(go.Scatter(
            x=pts_x, y=pts_y,
            mode="lines+markers",
            line=dict(color="#F97316", width=2, dash="dot"),
            marker=dict(color="#991B1B", size=9, symbol="x"),
            name="Optimum Shift Path"
        ))
        for k in range(len(pts_x) - 1):
            fig.add_annotation(
                x=pts_x[k + 1], y=pts_y[k + 1],
                ax=pts_x[k], ay=pts_y[k],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2,
                arrowcolor="#EA580C"
            )

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
        height=600,
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.88)"),
        dragmode="zoom"
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

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

    if step_idx > 0:
        st.markdown("---")
        st.subheader("📜 Previous Iteration Tableaux History")
        for past_idx in range(step_idx - 1, -1, -1):
            past_state = st.session_state.steps[past_idx]
            (px1, px2), pz, _ = extract_solution(past_state["tableau"])
            with st.expander(f"Stage {past_idx}: {past_state['chosen_cut_desc']}  |  Solution: ({fmt_frac(px1)}, {fmt_frac(px2)}), Z = {fmt_frac(pz)}"):
                st.dataframe(tableau_to_dataframe(past_state["tableau"]), use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 3: ISOLATED UNIT LATTICE SQUARES COMPARISON & TRAJECTORY
# ==============================================================================
def build_isolated_lattice_square_fig(stage_idx, steps_list, c1, c2, a11, a12, b1, a21, a22, b2):
    """
    Constructs a dedicated, zoomed-in plot of ONLY the unit lattice square
    enclosing Stage `stage_idx`'s solution, annotated with:
      - Feasible region inside the lattice square (blue)
      - Region sliced off by Gomory cuts (hatched/pink)
      - 4 Corner lattice vertices labeled with coordinates, Z-value, and feasibility symbol (● vs ○)
      - Non-integer optimal point (◆) or Final integer point (★)
      - Incoming arrow (from Stage k-1) and Outgoing arrow (to Stage k+1)
      - Normal/push arrow on the newly applied Gomory cut showing the direction it shaves space
    """
    state = steps_list[stage_idx]
    (x1_opt, x2_opt), z_opt, is_int = extract_solution(state["tableau"])
    fx1, fx2 = float(x1_opt), float(x2_opt)

    # Determine the 1x1 unit lattice square enclosing (x1_opt, x2_opt)
    # If x1_opt or x2_opt is already an integer, frame the unit square [x0, x0+1] x [y0, y0+1]
    # anchored at its integer coordinates (or shifted back 1 if on upper edge of previous square)
    sq_x0 = math.floor(x1_opt)
    sq_y0 = math.floor(x2_opt)
    if x1_opt.denominator == 1 and sq_x0 > 0 and stage_idx > 0:
        prev_x1 = extract_solution(steps_list[stage_idx - 1]["tableau"])[0][0]
        if prev_x1 < x1_opt:
            sq_x0 -= 1
    if x2_opt.denominator == 1 and sq_y0 > 0 and stage_idx > 0:
        prev_x2 = extract_solution(steps_list[stage_idx - 1]["tableau"])[0][1]
        if prev_x2 < x2_opt:
            sq_y0 -= 1

    pad = 0.28
    xmin, xmax = sq_x0 - pad, sq_x0 + 1.0 + pad
    ymin, ymax = sq_y0 - pad, sq_y0 + 1.0 + pad

    fig = go.Figure()

    # Grid inside lattice square window
    gx = np.linspace(xmin, xmax, 160)
    gy = np.linspace(ymin, ymax, 160)
    GX, GY = np.meshgrid(gx, gy)

    orig_mask = (
        (a11 * GX + a12 * GY <= b1 + 1e-6)
        & (a21 * GX + a22 * GY <= b2 + 1e-6)
        & (GX >= -1e-6)
        & (GY >= -1e-6)
    ).astype(float)

    # Show shaved-off region in soft red if cuts exist
    if state["cuts_2d"]:
        curr_mask = orig_mask.copy()
        for (al1, al2, bt, _) in state["cuts_2d"]:
            curr_mask *= (float(al1) * GX + float(al2) * GY <= float(bt) + 1e-6).astype(float)

        shaved_mask = np.clip(orig_mask - curr_mask, 0, 1)
        fig.add_trace(go.Contour(
            x=gx, y=gy, z=shaved_mask,
            contours=dict(start=0.5, end=1.5, size=1, coloring="fill"),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(239, 68, 68, 0.22)"]],
            showscale=False, hoverinfo="skip", name="Shaved-Off Region"
        ))
        fig.add_trace(go.Contour(
            x=gx, y=gy, z=curr_mask,
            contours=dict(start=0.5, end=1.5, size=1, coloring="fill"),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(59, 130, 246, 0.28)"]],
            showscale=False, hoverinfo="skip", name="Remaining Feasible"
        ))
    else:
        fig.add_trace(go.Contour(
            x=gx, y=gy, z=orig_mask,
            contours=dict(start=0.5, end=1.5, size=1, coloring="fill"),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(59, 130, 246, 0.24)"]],
            showscale=False, hoverinfo="skip", name="LP Feasible Region"
        ))

    # Draw the isolated 1x1 Unit Lattice Square boundary
    fig.add_shape(
        type="rect",
        x0=sq_x0, y0=sq_y0, x1=sq_x0 + 1, y1=sq_y0 + 1,
        line=dict(color="#D97706", width=3, dash="solid"),
        fillcolor="rgba(254, 243, 199, 0.12)",
        layer="below"
    )

    # Original constraint lines passing through the lattice square
    x_line = np.linspace(xmin, xmax, 150)
    if a12 != 0:
        fig.add_trace(go.Scatter(
            x=x_line, y=(b1 - a11 * x_line) / a12, mode="lines",
            line=dict(color="#334155", width=1.6),
            name=f"C1: {a11}x₁+{a12}x₂≤{b1}"
        ))
    if a22 != 0:
        fig.add_trace(go.Scatter(
            x=x_line, y=(b2 - a21 * x_line) / a22, mode="lines",
            line=dict(color="#64748B", width=1.6, dash="dash"),
            name=f"C2: {a21}x₁+{a22}x₂≤{b2}"
        ))

    # Plot Gomory cuts active at this stage + push arrows
    cut_colors = ["#DC2626", "#7C3AED", "#059669", "#DB2777", "#EA580C"]
    for idx, (al1, al2, bt, _) in enumerate(state["cuts_2d"]):
        col = cut_colors[idx % len(cut_colors)]
        f_al1, f_al2, f_bt = float(al1), float(al2), float(bt)
        is_latest = (idx == len(state["cuts_2d"]) - 1)
        if abs(f_al2) > 1e-9:
            y_cut = (f_bt - f_al1 * x_line) / f_al2
            fig.add_trace(go.Scatter(
                x=x_line, y=y_cut, mode="lines",
                line=dict(color=col, width=3.2 if is_latest else 1.8, dash="dashdot"),
                name=f"Cut {idx+1}: {fmt_frac(al1)}x₁+{fmt_frac(al2)}x₂≤{fmt_frac(bt)}"
            ))
            # Add inward normal arrow on the latest cut showing feasible half-space direction
            if is_latest:
                mid_x = sq_x0 + 0.5
                mid_y = (f_bt - f_al1 * mid_x) / f_al2
                if ymin <= mid_y <= ymax:
                    norm = math.hypot(f_al1, f_al2)
                    if norm > 0:
                        nx, ny = -f_al1 / norm * 0.18, -f_al2 / norm * 0.18
                        fig.add_annotation(
                            x=mid_x + nx, y=mid_y + ny,
                            ax=mid_x, ay=mid_y,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
                            arrowcolor=col
                        )
        elif abs(f_al1) > 1e-9:
            vx = f_bt / f_al1
            fig.add_vline(x=vx, line_color=col, line_width=3.2 if is_latest else 1.8, line_dash="dashdot")

    # Evaluate the 4 Lattice Corners of the Unit Square
    corners = [
        (sq_x0, sq_y0),
        (sq_x0 + 1, sq_y0),
        (sq_x0, sq_y0 + 1),
        (sq_x0 + 1, sq_y0 + 1),
    ]
    for (cx, cy) in corners:
        feas = (cx >= 0) and (cy >= 0) and (a11 * cx + a12 * cy <= b1 + 1e-6) and (a21 * cx + a22 * cy <= b2 + 1e-6)
        for (al1, al2, bt, _) in state["cuts_2d"]:
            feas &= (float(al1) * cx + float(al2) * cy <= float(bt) + 1e-6)

        z_corner = c1 * cx + c2 * cy
        is_curr_winner = is_int and (cx == int(x1_opt)) and (cy == int(x2_opt))

        if is_curr_winner:
            sym, m_col, m_sz, status_txt = "star", "#16A34A", 18, "★ OPTIMAL"
        elif feas:
            sym, m_col, m_sz, status_txt = "circle", "#0F172A", 12, "● Feasible"
        else:
            sym, m_col, m_sz, status_txt = "circle-open", "#94A3B8", 11, "○ Cut Off"

        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(symbol=sym, color=m_col, size=m_sz, line=dict(width=2, color=m_col)),
            showlegend=False,
            hovertemplate=f"Lattice Corner ({cx}, {cy})<br>Status: {status_txt}<br>Z = {z_corner}<extra></extra>"
        ))

        # Corner label positioning
        y_off = -18 if cy == sq_y0 else 16
        x_off = -8 if cx == sq_x0 else 8
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>({cx},{cy})</b> Z={z_corner}<br>{status_txt}",
            showarrow=False,
            xshift=x_off, yshift=y_off,
            font=dict(size=10, color="#16A34A" if is_curr_winner else ("#0F172A" if feas else "#64748B")),
            bgcolor="rgba(255,255,255,0.82)"
        )

    # Incoming Arrow from Previous Stage (if stage_idx > 0)
    if stage_idx > 0:
        (px1, px2), _, _ = extract_solution(steps_list[stage_idx - 1]["tableau"])
        fpx1, fpx2 = float(px1), float(px2)
        if abs(fpx1 - fx1) > 1e-6 or abs(fpx2 - fx2) > 1e-6:
            fig.add_trace(go.Scatter(
                x=[fpx1], y=[fpx2], mode="markers",
                marker=dict(symbol="x", color="#991B1B", size=11),
                showlegend=False,
                hovertemplate=f"Previous Stage {stage_idx-1}: ({fmt_frac(px1)}, {fmt_frac(px2)})<extra></extra>"
            ))
            fig.add_annotation(
                x=fx1, y=fx2, ax=fpx1, ay=fpx2,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.5,
                arrowcolor="#EA580C"
            )
            mid_ax, mid_ay = (fpx1 + fx1) / 2, (fpx2 + fx2) / 2
            fig.add_annotation(
                x=mid_ax, y=mid_ay,
                text=f"<b>Step {stage_idx}</b>",
                showarrow=False, yshift=11,
                font=dict(size=10, color="#C2410C"),
                bgcolor="rgba(255,247,237,0.9)"
            )

    # Outgoing Arrow Preview to Next Stage (if stage_idx < len(steps_list) - 1)
    if stage_idx < len(steps_list) - 1:
        (nx1, nx2), _, _ = extract_solution(steps_list[stage_idx + 1]["tableau"])
        fnx1, fnx2 = float(nx1), float(nx2)
        if abs(fnx1 - fx1) > 1e-6 or abs(fnx2 - fx2) > 1e-6:
            fig.add_annotation(
                x=fnx1, y=fnx2, ax=fx1, ay=fx2,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.3, arrowwidth=2,
                arrowcolor="#2563EB"
            )

    # Current Solution Marker (if non-integer; integer star is already plotted at corner)
    if not is_int:
        fig.add_trace(go.Scatter(
            x=[fx1], y=[fx2],
            mode="markers+text",
            text=[f"◆ ({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)})"],
            textposition="top left",
            textfont=dict(size=11, color="#DC2626"),
            marker=dict(symbol="diamond", color="#DC2626", size=14, line=dict(width=1.5, color="white")),
            showlegend=False,
            hovertemplate=f"Non-Integer Optimum ({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)})<br>Z = {fmt_frac(z_opt)}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>Stage {stage_idx}: Square [{sq_x0},{sq_x0+1}]×[{sq_y0},{sq_y0+1}]</b><br><sup>Sol: ({fmt_frac(x1_opt)}, {fmt_frac(x2_opt)}) | Z = {fmt_frac(z_opt)}</sup>",
            font=dict(size=13)
        ),
        xaxis=dict(range=[xmin, xmax], dtick=1, title="x₁", gridcolor="#E2E8F0", zeroline=True),
        yaxis=dict(range=[ymin, ymax], dtick=1, title="x₂", gridcolor="#E2E8F0", zeroline=True),
        height=410,
        margin=dict(l=35, r=25, t=65, b=40),
        template="plotly_white",
        showlegend=False,
        dragmode="zoom"
    )
    return fig, (sq_x0, sq_y0), (x1_opt, x2_opt), z_opt, is_int

def page_lattice_squares():
    st.markdown('<div class="main-header">🔲 Page 3: Isolated Unit Lattice Squares & Convergence Trajectory</div>', unsafe_allow_html=True)
    current_state, curr_tab, (x1_opt, x2_opt), z_opt, is_integer, step_idx = render_status_banner()

    # Legend / Symbol Key Banner
    st.markdown("""
    **Symbol & Arrow Legend:** &nbsp;
    `◆` Non-Integer LP Optimum &nbsp;|&nbsp;
    `★` Final Optimal Integer Point &nbsp;|&nbsp;
    `●` Feasible Lattice Corner &nbsp;|&nbsp;
    `○` Infeasible / Sliced-Off Corner &nbsp;|&nbsp;
    `➔ (Orange)` Shift from Previous Stage &nbsp;|&nbsp;
    `➔ (Blue)` Next Cut Shift &nbsp;|&nbsp;
    `➔ (Cut Color)` Half-Space Kept by Cut
    """)

    steps_list = st.session_state.steps
    num_stages = len(steps_list)

    # Quick Cut Application Bar right on Page 3 so user can iteratively generate & compare squares
    if not is_integer:
        with st.expander("✂️ Apply Next Gomory Cutting Plane Directly from Here", expanded=True):
            candidate_cuts = build_candidate_cuts(curr_tab, current_state["slack_2d_map"])
            if candidate_cuts:
                c1_col, c2_col = st.columns([2.6, 1])
                cut_opts = [
                    f"Cut from Row {c['source_var']} (RHS = {fmt_frac(c['rhs_val'])})  →  {c['xy_ineq_str']}"
                    for c in candidate_cuts
                ]
                sel_i = c1_col.selectbox("Choose Gomory Cut to Add Next Lattice Square:", range(len(candidate_cuts)), format_func=lambda i: cut_opts[i])
                if c2_col.button("✂️ Add Cut & Compare Square", type="primary", use_container_width=True):
                    apply_cut_action(candidate_cuts[sel_i])
                    st.rerun()

    st.markdown("### 1. Side-by-Side Comparison of All Enclosing Unit Lattice Squares")
    st.write("Each panel isolates **only the $1 \\times 1$ unit integer lattice square** enclosing the solution at that stage (use mouse scroll or box-drag inside any square to zoom further).")

    # Display Lattice Squares in a 2-column (or 3-column if >= 3 stages) responsive grid
    cols_per_row = 2 if num_stages <= 2 else 3
    for row_start in range(0, num_stages, cols_per_row):
        row_cols = st.columns(cols_per_row)
        for offset in range(cols_per_row):
            s_idx = row_start + offset
            if s_idx < num_stages:
                with row_cols[offset]:
                    sq_fig, (sq_x0, sq_y0), (sx1, sx2), sz, s_int = build_isolated_lattice_square_fig(
                        s_idx, steps_list, c1, c2, a11, a12, b1, a21, a22, b2
                    )
                    st.plotly_chart(sq_fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

                    # Summary card under each lattice square
                    if s_idx == 0:
                        delta_txt = "Starting Real LP Relaxation Optimum"
                    else:
                        prev_x, prev_z, _ = extract_solution(steps_list[s_idx - 1]["tableau"])
                        dx1 = sx1 - prev_x[0]
                        dx2 = sx2 - prev_x[1]
                        dz = sz - prev_z
                        delta_txt = f"Δx₁ = {fmt_frac(dx1)}, Δx₂ = {fmt_frac(dx2)}, ΔZ = {fmt_frac(dz)}"

                    status_badge = "✅ Integer Optimal Reached" if s_int else f"⏳ Fractional Parts: ({fmt_frac(frac_part(sx1))}, {fmt_frac(frac_part(sx2))})"
                    st.markdown(f"""
                    <div class="lattice-summary-card">
                        <b>Square:</b> [{sq_x0}, {sq_x0+1}] × [{sq_y0}, {sq_y0+1}] &nbsp;|&nbsp; {status_badge}<br>
                        <b>Constraint Added:</b> <code>{steps_list[s_idx]['chosen_cut_desc']}</code><br>
                        <b>Movement:</b> {delta_txt}
                    </div>
                    """, unsafe_allow_html=True)

    # Combined Lattice Neighbourhood Trajectory Overview
    st.markdown("---")
    st.markdown("### 2. Combined Lattice Neighbourhood Trajectory (All Stages & Arrows)")
    st.write("A unified view spanning **only the integer lattice squares visited** from the initial real solution to the current/final integer solution.")

    all_pts = [extract_solution(s["tableau"])[0] for s in steps_list]
    all_x = [float(p[0]) for p in all_pts]
    all_y = [float(p[1]) for p in all_pts]

    min_sq_x = min(math.floor(p[0]) for p in all_pts)
    max_sq_x = max(math.floor(p[0]) + 1 for p in all_pts)
    min_sq_y = min(math.floor(p[1]) for p in all_pts)
    max_sq_y = max(math.floor(p[1]) + 1 for p in all_pts)

    ov_xmin, ov_xmax = max(-0.3, min_sq_x - 0.5), max_sq_x + 0.5
    ov_ymin, ov_ymax = max(-0.3, min_sq_y - 0.5), max_sq_y + 0.5

    ov_fig = go.Figure()

    # Draw dashed unit lattice square outlines for every visited square
    visited_squares = set((math.floor(p[0]), math.floor(p[1])) for p in all_pts)
    for (vx0, vy0) in visited_squares:
        ov_fig.add_shape(
            type="rect",
            x0=vx0, y0=vy0, x1=vx0 + 1, y1=vy0 + 1,
            line=dict(color="#F59E0B", width=2, dash="dash"),
            fillcolor="rgba(254, 243, 199, 0.25)",
            layer="below"
        )

    # Active cuts
    x_ov = np.linspace(ov_xmin, ov_xmax, 200)
    cut_colors = ["#DC2626", "#7C3AED", "#059669", "#DB2777", "#EA580C"]
    for idx, (al1, al2, bt, _) in enumerate(current_state["cuts_2d"]):
        col = cut_colors[idx % len(cut_colors)]
        f_al1, f_al2, f_bt = float(al1), float(al2), float(bt)
        if abs(f_al2) > 1e-9:
            ov_fig.add_trace(go.Scatter(
                x=x_ov, y=(f_bt - f_al1 * x_ov) / f_al2, mode="lines",
                line=dict(color=col, width=2.4, dash="dashdot"),
                name=f"Cut {idx+1}: {fmt_frac(al1)}x₁ + {fmt_frac(al2)}x₂ ≤ {fmt_frac(bt)}"
            ))

    # Lattice grid points in visited neighbourhood
    ix_ov = np.arange(max(0, int(math.floor(ov_xmin))), int(math.ceil(ov_xmax)) + 1)
    iy_ov = np.arange(max(0, int(math.floor(ov_ymin))), int(math.ceil(ov_ymax)) + 1)
    IX_o, IY_o = np.meshgrid(ix_ov, iy_ov)
    IX_of, IY_of = IX_o.flatten(), IY_o.flatten()

    feas_ov = (a11 * IX_of + a12 * IY_of <= b1 + 1e-6) & (a21 * IX_of + a22 * IY_of <= b2 + 1e-6)
    for (al1, al2, bt, _) in current_state["cuts_2d"]:
        feas_ov &= (float(al1) * IX_of + float(al2) * IY_of <= float(bt) + 1e-6)

    ov_fig.add_trace(go.Scatter(
        x=IX_of[~feas_ov], y=IY_of[~feas_ov], mode="markers",
        marker=dict(color="#94A3B8", size=9, symbol="circle-open"),
        name="Infeasible / Cut-Off Lattice Points"
    ))
    ov_fig.add_trace(go.Scatter(
        x=IX_of[feas_ov], y=IY_of[feas_ov], mode="markers+text",
        text=[f"({int(px)},{int(py)})" for px, py in zip(IX_of[feas_ov], IY_of[feas_ov])],
        textposition="bottom center",
        marker=dict(color="#0F172A", size=11, symbol="circle"),
        name="Feasible Lattice Points"
    ))

    # Trajectory path & stage markers
    stage_labels = [
        f"S{k}: ({fmt_frac(p[0])}, {fmt_frac(p[1])})"
        for k, p in enumerate(all_pts)
    ]
    ov_fig.add_trace(go.Scatter(
        x=all_x, y=all_y,
        mode="lines+markers+text",
        text=stage_labels,
        textposition="top right",
        line=dict(color="#EA580C", width=2.5, dash="dot"),
        marker=dict(
            size=[15 if (k == num_stages - 1 and is_integer) else 12 for k in range(num_stages)],
            color=["#16A34A" if (k == num_stages - 1 and is_integer) else "#DC2626" for k in range(num_stages)],
            symbol=["star" if (k == num_stages - 1 and is_integer) else "diamond" for k in range(num_stages)]
        ),
        name="Solution Trajectory"
    ))

    # Arrows connecting consecutive stages
    for k in range(num_stages - 1):
        ov_fig.add_annotation(
            x=all_x[k + 1], y=all_y[k + 1],
            ax=all_x[k], ay=all_y[k],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2.6,
            arrowcolor="#EA580C"
        )

    ov_fig.update_layout(
        title="Step-by-Step Convergence Path Across Unit Lattice Squares",
        xaxis=dict(range=[ov_xmin, ov_xmax], dtick=1, title="x₁", gridcolor="#E2E8F0"),
        yaxis=dict(range=[ov_ymin, ov_ymax], dtick=1, title="x₂", gridcolor="#E2E8F0"),
        height=460,
        template="plotly_white",
        dragmode="zoom"
    )
    st.plotly_chart(ov_fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

# --- MULTI-PAGE NAVIGATION ---
nav = st.navigation([
    st.Page(page_geometry, title="1. Geometric & Lattice Zoom View", icon="📐"),
    st.Page(page_simplex, title="2. Simplex Tables & Cut Derivations", icon="📊"),
    st.Page(page_lattice_squares, title="3. Isolated Lattice Squares & Arrows", icon="🔲")
])
nav.run()
