import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from fractions import Fraction

# Page Configuration
st.set_page_config(
    page_title="Gomory's Fractional Cut Visualizer",
    page_icon="📐",
    layout="wide"
)

# Custom CSS for clean mathematical styling
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1E3A8A;}
    .sub-header {font-size: 1.3rem; font-weight: 600; color: #2563EB; margin-top: 1rem;}
    .math-box {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📐 Geometry of Gomory\'s Fractional Cut (ILP)</div>', unsafe_allow_html=True)
st.write(
    "Explore how **Gomory Cutting Planes** iteratively shave off non-integer regions of an LP relaxation "
    "without removing any feasible integer lattice points."
)

# --- SIDEBAR: PROBLEM SETUP ---
st.sidebar.header("⚙️ Define 2-Variable ILP")

preset = st.sidebar.selectbox(
    "Choose a Preset Example",
    [
        "Classic Textbook Example (Max 5x₁ + 4x₂)",
        "Steep Slope Example (Max 3x₁ + 4x₂)",
        "Custom Problem"
    ]
)

if preset == "Classic Textbook Example (Max 5x₁ + 4x₂)":
    c1_def, c2_def = 5.0, 4.0
    a11_def, a12_def, b1_def = 1.0, 1.0, 5.0
    a21_def, a22_def, b2_def = 10.0, 6.0, 45.0
elif preset == "Steep Slope Example (Max 3x₁ + 4x₂)":
    c1_def, c2_def = 3.0, 4.0
    a11_def, a12_def, b1_def = 2.0, 1.0, 6.0
    a21_def, a22_def, b2_def = 2.0, 3.0, 9.0
else:
    c1_def, c2_def = 4.0, 3.0
    a11_def, a12_def, b1_def = 2.0, 3.0, 12.0
    a21_def, a22_def, b2_def = 4.0, 1.0, 11.0

st.sidebar.subheader("Objective Function (Maximise Z)")
col_c1, col_c2 = st.sidebar.columns(2)
c1 = col_c1.number_input("Coeff of x₁ (c₁)", value=c1_def, step=1.0)
c2 = col_c2.number_input("Coeff of x₂ (c₂)", value=c2_def, step=1.0)

st.sidebar.subheader("Constraint 1: a₁₁x₁ + a₁₂x₂ ≤ b₁")
c1_1, c1_2, c1_b = st.sidebar.columns(3)
a11 = c1_1.number_input("a₁₁", value=a11_def, step=1.0)
a12 = c1_2.number_input("a₁₂", value=a12_def, step=1.0)
b1 = c1_b.number_input("b₁", value=b1_def, step=1.0)

st.sidebar.subheader("Constraint 2: a₂₁x₁ + a₂₂x₂ ≤ b₂")
c2_1, c2_2, c2_b = st.sidebar.columns(3)
a21 = c2_1.number_input("a₂₁", value=a21_def, step=1.0)
a22 = c2_2.number_input("a₂₂", value=a22_def, step=1.0)
b2 = c2_b.number_input("b₂", value=b2_def, step=1.0)

max_cuts = st.sidebar.slider("Number of Gomory Cuts to Apply", min_value=0, max_value=4, value=1)

# --- HELPER FUNCTIONS ---
def to_frac_str(val, tol=1e-6):
    """Convert float to clean p/q fraction string."""
    if abs(val - round(val)) < tol:
        return str(int(round(val)))
    f = Fraction(val).limit_denominator(1000)
    return f"{f.numerator}/{f.denominator}"

def frac_part(val, tol=1e-7):
    """Compute positive fractional part f = val - floor(val)."""
    f = val - np.floor(val + tol)
    if abs(f) < tol or abs(f - 1.0) < tol:
        return 0.0
    return f

def solve_gomory_cuts(c, A_orig, b_orig, num_cuts):
    """
    Solves LP relaxation and iteratively generates Gomory fractional cuts
    expressed in terms of the original decision variables (x1, x2).
    """
    A_curr = np.array(A_orig, dtype=float)
    b_curr = np.array(b_orig, dtype=float)
    
    history = []
    
    for step in range(num_cuts + 1):
        # Solve current LP relaxation: Maximize c^T x => Minimize -c^T x
        res = linprog(-np.array(c), A_ub=A_curr, b_ub=b_curr, bounds=[(0, None), (0, None)], method='highs')
        
        if not res.success:
            break
            
        x_opt = res.x
        z_opt = -res.fun
        
        step_info = {
            "step": step,
            "A": A_curr.copy(),
            "b": b_curr.copy(),
            "x_opt": x_opt,
            "z_opt": z_opt,
            "is_integer": all(frac_part(val) == 0 for val in x_opt),
            "cut_generated": None
        }
        
        if step_info["is_integer"] or step == num_cuts:
            history.append(step_info)
            break
            
        # Construct optimal simplex tableau row via Basis Matrix B^-1
        # Standard form: A_curr * x + I * s = b_curr
        m = len(b_curr)
        A_std = np.hstack([A_curr, np.eye(m)])
        x_full = np.concatenate([x_opt, b_curr - A_curr @ x_opt])
        
        # Identify basic variables (m variables with largest values / non-zero)
        # Use tolerance to pick m linearly independent basic columns
        basic_indices = []
        for idx in np.argsort(-x_full):
            if len(basic_indices) < m:
                candidate_cols = basic_indices + [idx]
                if np.linalg.matrix_rank(A_std[:, candidate_cols]) == len(candidate_cols):
                    basic_indices.append(idx)
        basic_indices = sorted(basic_indices)
        non_basic_indices = [j for j in range(2 + m) if j not in basic_indices]
        
        B = A_std[:, basic_indices]
        B_inv = np.linalg.inv(B)
        tableau = B_inv @ A_std
        rhs = B_inv @ b_curr
        
        # Select source row with largest fractional part among basic decision variables (or any basic var)
        best_row = -1
        max_f = -1.0
        for r_idx, b_var in enumerate(basic_indices):
            f_val = frac_part(rhs[r_idx])
            if f_val > max_f + 1e-6:
                max_f = f_val
                best_row = r_idx
                
        if best_row == -1 or max_f < 1e-6:
            step_info["is_integer"] = True
            history.append(step_info)
            break
            
        # Gomory Cut in non-basic variables: sum_{j in NB} f_ij * x_j >= f_i0
        f_rhs = frac_part(rhs[best_row])
        cut_std = np.zeros(2 + m)
        for j in non_basic_indices:
            cut_std[j] = frac_part(tableau[best_row, j])
            
        # Substitute slack variables s_k = b_k - sum(A_kj * x_j) to express cut in (x1, x2) space:
        # sum_{j=0,1} cut_std[j]*x_j + sum_{k=0..m-1} cut_std[2+k]*(b_k - A_k0*x1 - A_k1*x2) >= f_rhs
        # Rearranging to <= form for plotting & next LP: alpha_1 * x1 + alpha_2 * x2 <= beta
        s_coeffs = cut_std[2:]
        alpha_1 = -(cut_std[0] - np.dot(s_coeffs, A_curr[:, 0]))
        alpha_2 = -(cut_std[1] - np.dot(s_coeffs, A_curr[:, 1]))
        beta = -(f_rhs - np.dot(s_coeffs, b_curr))
        
        step_info["cut_generated"] = {
            "source_var": f"x_{basic_indices[best_row]+1}" if basic_indices[best_row] < 2 else f"s_{basic_indices[best_row]-1}",
            "rhs_val": rhs[best_row],
            "f_rhs": f_rhs,
            "non_basic_terms": [(j, cut_std[j]) for j in non_basic_indices if cut_std[j] > 1e-6],
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "beta": beta
        }
        history.append(step_info)
        
        # Add new cut to constraints
        A_curr = np.vstack([A_curr, [alpha_1, alpha_2]])
        b_curr = np.append(b_curr, beta)
        
    return history

# Run solver
A_init = [[a11, a12], [a21, a22]]
b_init = [b1, b2]
c_vec = [c1, c2]
history = solve_gomory_cuts(c_vec, A_init, b_init, max_cuts)

# --- MAIN LAYOUT ---
tab1, tab2 = st.tabs(["📊 Interactive Geometric Plot", "📖 Mathematical & Geometric Intuition"])

with tab1:
    if not history:
        st.error("The LP problem is infeasible with the given constraints. Please adjust the sidebar values.")
    else:
        col_plot, col_details = st.columns([1.35, 1])
        
        with col_plot:
            st.subheader("Feasible Region, Integer Lattice & Cutting Planes")
            
            # Determine plot bounds
            x_max_bound = max(b1 / max(a11, 0.1), b2 / max(a21, 0.1), 5) * 1.15
            y_max_bound = max(b1 / max(a12, 0.1), b2 / max(a22, 0.1), 5) * 1.15
            x_max_bound = min(x_max_bound, 25)
            y_max_bound = min(y_max_bound, 25)
            
            x_vals = np.linspace(0, x_max_bound, 400)
            y_vals = np.linspace(0, y_max_bound, 400)
            X, Y = np.meshgrid(x_vals, y_vals)
            
            fig, ax = plt.subplots(figsize=(8, 6.5))
            
            # 1. Original LP Feasible Region (Light Gray)
            orig_feasible = (a11 * X + a12 * Y <= b1 + 1e-5) & (a21 * X + a22 * Y <= b2 + 1e-5)
            ax.contourf(X, Y, orig_feasible, levels=[0.5, 1.5], colors=['#CBD5E1'], alpha=0.45)
            
            # 2. Current Cut Feasible Region (Soft Blue)
            final_state = history[-1]
            curr_feasible = np.ones_like(X, dtype=bool)
            for row_a, val_b in zip(final_state["A"], final_state["b"]):
                curr_feasible &= (row_a[0] * X + row_a[1] * Y <= val_b + 1e-5)
            ax.contourf(X, Y, curr_feasible, levels=[0.5, 1.5], colors=['#3B82F6'], alpha=0.35)
            
            # 3. Plot Original Constraint Lines
            if a12 != 0:
                ax.plot(x_vals, (b1 - a11 * x_vals) / a12, 'k-', lw=1.8, label=f"C1: {to_frac_str(a11)}x₁ + {to_frac_str(a12)}x₂ ≤ {to_frac_str(b1)}")
            if a22 != 0:
                ax.plot(x_vals, (b2 - a21 * x_vals) / a22, 'k--', lw=1.8, label=f"C2: {to_frac_str(a21)}x₁ + {to_frac_str(a22)}x₂ ≤ {to_frac_str(b2)}")
                
            # 4. Plot Gomory Cuts
            cut_colors = ['#DC2626', '#D97706', '#7C3AED', '#059669']
            for idx in range(2, len(final_state["b"])):
                a_cut = final_state["A"][idx]
                b_cut = final_state["b"][idx]
                color = cut_colors[(idx - 2) % len(cut_colors)]
                if abs(a_cut[1]) > 1e-6:
                    ax.plot(x_vals, (b_cut - a_cut[0] * x_vals) / a_cut[1], color=color, lw=2.3, linestyle='-.',
                            label=f"Gomory Cut {idx-1}: {to_frac_str(a_cut[0])}x₁ + {to_frac_str(a_cut[1])}x₂ ≤ {to_frac_str(b_cut)}")
                elif abs(a_cut[0]) > 1e-6:
                    ax.axvline(b_cut / a_cut[0], color=color, lw=2.3, linestyle='-.',
                               label=f"Gomory Cut {idx-1}: x₁ ≤ {to_frac_str(b_cut / a_cut[0])}")
            
            # 5. Plot Integer Lattice Points
            int_x = np.arange(0, int(np.floor(x_max_bound)) + 1)
            int_y = np.arange(0, int(np.floor(y_max_bound)) + 1)
            IX, IY = np.meshgrid(int_x, int_y)
            IX_flat, IY_flat = IX.flatten(), IY.flatten()
            
            is_int_feas = (a11 * IX_flat + a12 * IY_flat <= b1 + 1e-5) & (a21 * IX_flat + a22 * IY_flat <= b2 + 1e-5)
            
            # Infeasible integer points
            ax.scatter(IX_flat[~is_int_feas], IY_flat[~is_int_feas], color='#94A3B8', s=25, alpha=0.5, label="Infeasible Integer Points")
            # Feasible integer points
            ax.scatter(IX_flat[is_int_feas], IY_flat[is_int_feas], color='#1E293B', s=65, zorder=5, label="Feasible Integer Lattice Points")
            
            # 6. Plot LP Relaxation Optimal & Current Optimal
            lp_opt = history[0]["x_opt"]
            ax.scatter(lp_opt[0], lp_opt[1], color='#DC2626', s=140, marker='X', zorder=6,
                       label=f"Initial LP Optimum ({to_frac_str(lp_opt[0])}, {to_frac_str(lp_opt[1])})")
            
            curr_opt = final_state["x_opt"]
            marker_style = '*' if final_state["is_integer"] else 'D'
            marker_color = '#16A34A' if final_state["is_integer"] else '#2563EB'
            status_lbl = "Integer Optimum" if final_state["is_integer"] else "Current Cut LP Optimum"
            ax.scatter(curr_opt[0], curr_opt[1], color=marker_color, s=180, marker=marker_style, zorder=7,
                       label=f"{status_lbl} ({to_frac_str(curr_opt[0])}, {to_frac_str(curr_opt[1])})")
            
            # 7. Objective Function Iso-profit Contour through current optimum
            if c2 != 0:
                z_curr = final_state["z_opt"]
                ax.plot(x_vals, (z_curr - c1 * x_vals) / c2, color='#10B981', linestyle=':', lw=1.5,
                        label=f"Iso-Profit Line (Z = {to_frac_str(z_curr)})")
            
            ax.set_xlim(0, x_max_bound)
            ax.set_ylim(0, y_max_bound)
            ax.set_xlabel("x₁", fontsize=12, fontweight='bold')
            ax.set_ylabel("x₂", fontsize=12, fontweight='bold')
            ax.set_title("Geometric Shaving of Non-Integer Feasible Space", fontsize=13, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right', fontsize=8.5, framealpha=0.92)
            
            st.pyplot(fig)
            
        with col_details:
            st.subheader("Step-by-Step Algebraic & Geometric Breakdown")
            
            for state in history:
                step_num = state["step"]
                x1_str = to_frac_str(state["x_opt"][0])
                x2_str = to_frac_str(state["x_opt"][1])
                z_str = to_frac_str(state["z_opt"])
                
                with st.expander(f"Iteration {step_num}: {'LP Relaxation' if step_num == 0 else f'After Cut {step_num}'}", expanded=True):
                    st.markdown(f"**Current LP Optimum:** $(x_1, x_2) = \\left({x1_str},\\; {x2_str}\\right)$")
                    st.markdown(f"**Objective Value:** $Z = {z_str}$")
                    
                    if state["is_integer"]:
                        st.success(f"🎉 **Integer Solution Found!** Both $x_1 = {x1_str}$ and $x_2 = {x2_str}$ are integers. Optimal ILP solution reached!")
                    elif state["cut_generated"] is not None:
                        cut = state["cut_generated"]
                        st.warning("⚠️ Solution is fractional. Generating Gomory Cut...")
                        
                        # Build non-basic string
                        nb_strs = []
                        for var_idx, coeff in cut["non_basic_terms"]:
                            v_name = f"x_{var_idx+1}" if var_idx < 2 else f"s_{var_idx-1}"
                            nb_strs.append(f"\\frac{{{Fraction(coeff).limit_denominator(1000).numerator}}}{{{Fraction(coeff).limit_denominator(1000).denominator}}}{v_name}")
                        
                        lhs_nb = " + ".join(nb_strs) if nb_strs else "0"
                        f_rhs_frac = Fraction(cut["f_rhs"]).limit_denominator(1000)
                        
                        st.markdown(f"""
                        <div class="math-box">
                        <b>1. Source Row ({cut['source_var']}):</b><br>
                        Basic value $= {to_frac_str(cut['rhs_val'])} \\implies$ fractional part $f_0 = \\frac{{{f_rhs_frac.numerator}}}{{{f_rhs_frac.denominator}}}$<br><br>
                        <b>2. Gomory Inequality (Non-Basic Space):</b><br>
                        $${lhs_nb} \\ge \\frac{{{f_rhs_frac.numerator}}}{{{f_rhs_frac.denominator}}}$$
                        <b>3. Projected onto $(x_1, x_2)$ Plane:</b><br>
                        $${to_frac_str(cut['alpha_1'])}x_1 + {to_frac_str(cut['alpha_2'])}x_2 \\le {to_frac_str(cut['beta'])}$$
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Increase the **Number of Gomory Cuts** slider in the sidebar to apply the next cutting plane.")

with tab2:
    st.subheader("How Does a Gomory Fractional Cut Work Geometrically?")
    
    st.markdown("""
    ### 1. The Core Geometric Idea: The Convex Hull
    When we solve an **Integer Linear Programming (ILP)** problem by ignoring the integer requirement, we solve its **LP Relaxation**. 
    Geometrically, the LP relaxation forms a continuous polytope (shaded gray in the plot). Because of linear programming theory, the optimal solution always occurs at a **vertex (corner point)** of this polytope—which frequently has fractional coordinates.
    
    The true feasible region of the ILP is just the set of **integer grid points (lattice points)** inside that polytope. If we could shrink-wrap those integer points tightly, the resulting shape is called the **Integer Convex Hull**. Every vertex of the Integer Convex Hull is guaranteed to be an integer point!
    
    ### 2. What a Gomory Cut Does
    A **Gomory Fractional Cut** is a newly constructed linear constraint (hyperplane) that satisfies two strict geometric rules:
    1. **Separating Hyperplane:** It strictly cuts off the current fractional LP optimal vertex (the red $\\times$ on the graph), making that fractional point infeasible.
    2. **Validity (No Integer Points Lost):** Every single feasible integer lattice point inside the original region satisfies the new inequality. Not a single valid integer solution is ever sliced off.
    
    ### 3. Why the Algebra Guarantees This Geometry
    In the optimal simplex tableau, suppose a basic variable $x_i$ has a fractional value:
    $$x_i + \\sum_{j \\in NB} \\bar{a}_{ij} x_j = \\bar{b}_i$$
    
    We split every coefficient $\\bar{a}_{ij}$ and right-hand side $\\bar{b}_i$ into an **integer part** ($\\lfloor \\cdot \\rfloor$) and a **strictly non-negative fractional part** ($0 \\le f < 1$):
    $$\\bar{a}_{ij} = \\lfloor \\bar{a}_{ij} \\rfloor + f_{ij}, \\qquad \\bar{b}_i = \\lfloor \\bar{b}_i \\rfloor + f_i$$
    
    Rearranging integers to the left and fractions to the right:
    $$x_i + \\sum_{j \\in NB} \\lfloor \\bar{a}_{ij} \\rfloor x_j - \\lfloor \\bar{b}_i \\rfloor = f_i - \\sum_{j \\in NB} f_{ij} x_j$$
    
    * For any **integer solution**, the left-hand side must be an **integer**.
    * Therefore, the right-hand side $f_i - \\sum f_{ij} x_j$ must also be an integer.
    * Since $f_{ij} \\ge 0$ and $x_j \\ge 0$, we know $\\sum f_{ij} x_j \\ge 0$, which means $f_i - \\sum f_{ij} x_j \\le f_i < 1$.
    * The largest integer strictly less than $1$ is **$0$**! Thus:
    $$f_i - \\sum_{j \\in NB} f_{ij} x_j \\le 0 \\implies \\sum_{j \\in NB} f_{ij} x_j \\ge f_i$$
    
    **Geometric Punchline:** At the current LP optimum, all non-basic variables $x_j = 0$, making the left side $0 \\ge f_i$ (impossible since $f_i > 0$), so the fractional vertex is chopped off. Meanwhile, any integer point satisfies the inequality automatically!
    """)
