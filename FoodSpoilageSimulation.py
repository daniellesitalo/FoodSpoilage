import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

def simple_model_forward(t_array, y0, alpha, beta, step_size):
    F, S, R = y0
    results = []
    results.append([F, S, R])  # Add initial values first
    for t in t_array[1:]: 
        F_next = F - (alpha * F) * step_size
        S_next = S + (alpha * F - beta * S) * step_size
        R_next = R + (beta * S) * step_size
        F_next = np.clip(F_next, 0, 1)
        S_next = np.clip(S_next, 0, 1)
        R_next = np.clip(R_next, 0, 1)
        results.append([F_next, S_next, R_next])
        F, S, R = F_next, S_next, R_next
    return np.array(results)

def complex_model_forward(t_array, y0, alpha_1, alpha_2, gamma, delta, Beta_1, Beta_2, step_size):
    F, S1, S2, R = y0
    results = []
    results.append([F, S1, S2, R])  # Add initial values first
    for t in t_array[1:]:
        F_next = F - (alpha_1 * F + alpha_2 * F) * step_size
        S1_next = S1 + (alpha_1 * F - gamma * S1 + delta * S2 - Beta_1 * S1) * step_size
        S2_next = S2 + (alpha_2 * F + gamma * S1 - delta * S2 - Beta_2 * S2) * step_size
        R_next = R + (Beta_1 * S1 + Beta_2 * S2) * step_size
        F_next = np.clip(F_next, 0, 1)
        S1_next = np.clip(S1_next, 0, 1)
        S2_next = np.clip(S2_next, 0, 1)
        R_next = np.clip(R_next, 0, 1)
        results.append([F_next, S1_next, S2_next, R_next])
        F, S1, S2, R = F_next, S1_next, S2_next, R_next
    return np.array(results)

st.set_page_config(
    page_title="Simple vs Complex Mathematical Model for Food Spoilage",
    page_icon="🍅",
    layout="wide",
)

st.title("Simple vs Complex Mathematical Model for Food Spoilage")

tab_model, tab_help = st.tabs(["Model Simulation", "Help"])

with tab_model:
    model_type = st.selectbox("Select Model Type", ["Simple", "Complex"])

    n_days = st.number_input("Number of days", min_value=1, max_value=366, value=30, step=1)
    step_size = st.number_input("Step-size", min_value=0.01, max_value=1.0, value=1.0, step=0.01)

    t_train = np.arange(0, n_days + step_size, step_size)

    if model_type == "Simple":
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            alpha = st.number_input(
                "Alpha", min_value=0.0, max_value=1.0, value=0.4, step=0.01,
                help="**Alpha (α):** Spoilage rate. Fraction of fresh food that spoils per time step."
            )
        with col2:
            beta = st.number_input(
                "Beta", min_value=0.0, max_value=1.0, value=0.7, step=0.01,
                help="**Beta (β):** Removal rate. Fraction of spoiled food that is removed per time step."
            )
        with col3:
            F_0 = st.number_input(
                "F₀", min_value=0.0, max_value=1.0, value=0.99, step=0.01,
                help="**F₀:** Initial fraction of fresh food."
            )
        with col4:
            S_0 = st.number_input(
                "S₀", min_value=0.0, max_value=1.0, value=0.01, step=0.01,
                help="**S₀:** Initial fraction of spoiled food."
            )
        with col5:
            R_0 = st.number_input(
                "R₀", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                help="**R₀:** Initial fraction of removed food."
            )
        y0 = [F_0, S_0, R_0]

    else:
        # First row: model parameters (excluding initial values)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            alpha_1 = st.number_input(
                "Alpha₁", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                help="**Alpha₁ (α₁):** Decay rate. Fraction of fresh food decaying to S₁ per time step."
            )
        with col2:
            alpha_2 = st.number_input(
                "Alpha₂", min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                help="**Alpha₂ (α₂):** Cross-contamination rate. Fraction of fresh food decaying to S₂ per time step."
            )
        with col3:
            gamma = st.number_input(
                "Gamma", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                help="**Gamma (γ):** Rate from S₁ to S₂. Fraction of S₁ becoming S₂ per time step."
            )
        with col4:
            delta = st.number_input(
                "Delta", min_value=0.0, max_value=1.0, value=0.2, step=0.01,
                help="**Delta (δ):** Rate from S₂ to S₁."
            )
        with col5:
            Beta_1 = st.number_input(
                "Beta₁", min_value=0.0, max_value=1.0, value=1.0, step=0.01,
                help="**Beta₁ (β₁):** Removal rate for S₁. Fraction of S₁ removed per time step."
            )
        with col6:
            Beta_2 = st.number_input(
                "Beta₂", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                help="**Beta₂ (β₂):** Removal rate for S₂. Fraction of S₂ removed per time step."
            )
        # Second row: initial values
        col7, col8, col9, col10 = st.columns(4)
        with col7:
            F_0 = st.number_input(
                "F₀", min_value=0.0, max_value=1.0, value=0.98, step=0.01,
                help="**F₀:** Initial fraction of fresh food."
            )
        with col8:
            S1_0 = st.number_input(
                "S₁₀", min_value=0.0, max_value=1.0, value=0.01, step=0.01,
                help="**S₁₀:** Initial fraction of spoiled food in compartment S₁."
            )
        with col9:
            S2_0 = st.number_input(
                "S₂₀", min_value=0.0, max_value=1.0, value=0.01, step=0.01,
                help="**S₂₀:** Initial fraction of spoiled food in compartment S₂."
            )
        with col10:
            R_0 = st.number_input(
                "R₀", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                help="**R₀:** Initial fraction of removed food."
            )
        y0 = [F_0, S1_0, S2_0, R_0]

    if st.button("Run simulation"):
        with st.spinner('Simulating...'):
            if model_type == "Simple":
                y_sol = simple_model_forward(t_train, y0, alpha, beta, step_size)
                columns = ["F", "S", "R"]
            else:
                y_sol = complex_model_forward(t_train, y0, alpha_1, alpha_2, gamma, delta, Beta_1, Beta_2, step_size)
                columns = ["F", "S1", "S2", "R"]

            data_real = (
                pd.DataFrame(y_sol, columns=columns)
                .assign(time=t_train)
                .melt(id_vars="time", var_name="status", value_name="population")
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(
                data=data_real,
                x="time",
                y="population",
                hue="status",
                legend=True,
                linestyle="dashed",
                ax=ax
            )
            ax.set_title(f"{model_type} model - Forward Simulation")
            st.pyplot(fig)

with tab_help:
    st.header("Instructions")
    st.write(
        "Select either the Simple or Complex model, adjust parameters using the controls above, and run the simulation. "
        "Hover over the ? icon next to each parameter for a description. "
        "The Simple model uses three compartments (F, S, R), while the Complex model uses four (F, S1, S2, R) with additional parameters. "
    )
