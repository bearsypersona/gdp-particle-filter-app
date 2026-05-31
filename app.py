from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from particle_model import (
    prepare_gdp_data,
    estimate_proxy_parameters,
    particle_filter,
    kalman_filter,
    mle_state_path,
    rmse,
)

DATA_PATH = Path(__file__).parent / "data" / "GDP.csv"

st.set_page_config(page_title="GDP Particle Filter Model", layout="wide")

st.title("GDP Output Gap: Particle Filter Model")
st.caption("Interactive state-space model using U.S. real GDP data, a polynomial trend proxy, and hidden-state filtering.")

st.markdown(
    r"""
## Model shown up front

This app estimates a hidden output gap state, $x_t$, from observed GDP growth, $y_t$.

**State equation:**

$$x_t = \phi x_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma^2)$$

**Observation equation:**

$$y_t = \alpha x_t + \nu_t, \qquad \nu_t \sim N(0, \tau^2)$$

**What the parameters mean:**

- $\phi$: persistence of the hidden output gap. Higher $\phi$ means recessions/booms fade more slowly.
- $\alpha$: how strongly the hidden output gap moves observed GDP growth.
- $\sigma$: volatility of the hidden state shock.
- $\tau$: measurement noise in GDP growth.

The default U.S. parameters are estimated from a proxy output gap created by comparing actual GDP to a user-selected polynomial trend in log GDP. For another country, choose the polynomial degree that gives the most reasonable-looking smooth long-run GDP trend.
"""
)

with st.expander("Method summary", expanded=False):
    st.markdown(
        """
1. Fit a user-selected polynomial trend to log GDP.  
2. Convert the trend back into GDP levels.  
3. Define the proxy output gap as the percent deviation between actual GDP and trend GDP.  
4. Estimate default parameters from this proxy.  
5. Run the particle filter and Kalman filter on centered GDP growth.  
6. Compare the hidden-state estimates against the proxy output gap.  
"""
    )

st.markdown("""
### Note on using another country's GDP data

The app lets the user choose the polynomial degree. For another country, do not assume one fixed degree is automatically best. Upload a CSV with columns named `date` and `GDP`, then adjust the polynomial degree until the trend line looks like a reasonable long-run path rather than a curve that bends strangely or overfits short-term wiggles.
""")

st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Optional: upload another GDP CSV", type=["csv"])

st.sidebar.markdown("""
**CSV upload instructions**

If you upload GDP data for another country, rename the columns before uploading:

- the date column must be named `date`
- the real GDP level column must be named `GDP`

Example accepted format:

| date | GDP |
|---|---:|
| 1990-01-01 | 1234.5 |
| 1990-04-01 | 1251.8 |

The `date` column should be readable as dates, and `GDP` should contain numeric real GDP values. The app will use these two columns to build GDP growth, the polynomial trend, and the proxy output gap.
""")

degree = st.sidebar.slider(
    "Polynomial trend degree",
    min_value=1,
    max_value=8,
    value=5,
    step=1,
    help="For countries other than the default U.S. data, choose the degree that gives the best-looking smooth long-run trend. Higher is not always better."
)
st.sidebar.caption("For another country, visually check the GDP trend graph. If the polynomial bends wildly or misses the broad GDP path, try another degree.")

csv_source = uploaded_file if uploaded_file is not None else DATA_PATH

try:
    base_model_df, poly_model = prepare_gdp_data(csv_source, degree=degree)
    defaults = estimate_proxy_parameters(base_model_df)
except Exception as e:
    st.error(f"Could not load or process the GDP data: {e}")
    st.stop()

st.sidebar.header("Parameter Mode")
param_mode = st.sidebar.radio(
    "Choose parameters",
    ["Use default estimated parameters", "Use my own parameters"],
    index=0,
)

if param_mode == "Use default estimated parameters":
    phi = defaults["phi"]
    alpha = defaults["alpha"]
    sigma = defaults["sigma"]
    tau = defaults["tau"]
else:
    phi = st.sidebar.number_input("phi", min_value=-0.999, max_value=0.999, value=float(defaults["phi"]), step=0.01, format="%.6f")
    alpha = st.sidebar.number_input("alpha", min_value=-2.0, max_value=2.0, value=float(defaults["alpha"]), step=0.01, format="%.6f")
    sigma = st.sidebar.number_input("sigma", min_value=0.0001, max_value=20.0, value=float(defaults["sigma"]), step=0.05, format="%.6f")
    tau = st.sidebar.number_input("tau", min_value=0.0001, max_value=20.0, value=float(defaults["tau"]), step=0.05, format="%.6f")

st.sidebar.header("Filter Settings")
N = st.sidebar.slider("Number of particles", min_value=500, max_value=20000, value=5000, step=500)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=100, step=1)
run_mle = st.sidebar.checkbox("Also compute MLE state path", value=True)

run_button = st.sidebar.button("Run model", type="primary")

left, mid, right = st.columns(3)
left.metric("Observations", f"{len(base_model_df):,}")
mid.metric("Start date", str(base_model_df["date"].iloc[0].date()))
right.metric("End date", str(base_model_df["date"].iloc[-1].date()))

st.subheader("Parameter Values")
param_table = pd.DataFrame(
    {
        "Parameter": ["phi", "alpha", "sigma", "tau"],
        "Value Used": [phi, alpha, sigma, tau],
        "Default Estimate": [defaults["phi"], defaults["alpha"], defaults["sigma"], defaults["tau"]],
        "Meaning": [
            "Persistence of hidden output gap",
            "Relationship between output gap and centered GDP growth",
            "State-shock volatility",
            "Observation/measurement noise",
        ],
    }
)
st.dataframe(param_table, use_container_width=True, hide_index=True)

if "results" not in st.session_state:
    st.session_state.results = None

if run_button:
    with st.spinner("Running particle filter, Kalman filter, and charts..."):
        np.random.seed(int(seed))
        model_df = base_model_df.copy()
        Y = model_df["Y_t_gdp_growth"]
        Y_centered = Y - Y.mean()

        init_mean = float(model_df["X_t_proxy_output_gap"].iloc[0])
        init_sd = float(model_df["X_t_proxy_output_gap"].std())
        init_var = init_sd**2

        x_pf, y_pf, particles_pre, particles_post, pf_loglik = particle_filter(
            Y=Y_centered,
            phi=phi,
            alpha=alpha,
            sigma=sigma,
            tau=tau,
            N=int(N),
            init_mean=init_mean,
            init_sd=init_sd,
            store_paths=True,
        )

        x_kf, y_kf, P_kf, kf_loglik = kalman_filter(
            Y=Y_centered,
            phi=phi,
            alpha=alpha,
            sigma=sigma,
            tau=tau,
            init_mean=init_mean,
            init_var=init_var,
        )

        if run_mle:
            x_mle_path, y_mle_path, mle_path_loss, mle_success = mle_state_path(
                Y=Y_centered,
                phi=phi,
                alpha=alpha,
                sigma=sigma,
                tau=tau,
                init_mean=init_mean,
                init_sd=init_sd,
                maxiter=250,
            )
        else:
            x_mle_path = np.full(len(model_df), np.nan)
            y_mle_path = np.full(len(model_df), np.nan)
            mle_path_loss = np.nan
            mle_success = False

        model_df["X_t_particle_filter"] = x_pf
        model_df["Y_t_particle_filter_pred"] = y_pf
        model_df["X_t_kalman_filter"] = x_kf
        model_df["Y_t_kalman_filter_pred"] = y_kf
        model_df["P_t_kalman_filter"] = P_kf
        model_df["X_t_mle_path"] = x_mle_path
        model_df["Y_t_mle_path_pred"] = y_mle_path

        proxy = model_df["X_t_proxy_output_gap"]
        benchmark_rows = [
            {
                "Method": "Particle Filter",
                "RMSE vs Proxy Output Gap": rmse(proxy, model_df["X_t_particle_filter"]),
                "Log Likelihood / Objective": pf_loglik,
                "Notes": "Sequential Monte Carlo estimate using particles",
            },
            {
                "Method": "Kalman Filter",
                "RMSE vs Proxy Output Gap": rmse(proxy, model_df["X_t_kalman_filter"]),
                "Log Likelihood / Objective": kf_loglik,
                "Notes": "Linear-Gaussian benchmark filter",
            },
        ]
        if run_mle:
            benchmark_rows.append(
                {
                    "Method": "MLE State Path",
                    "RMSE vs Proxy Output Gap": rmse(proxy, model_df["X_t_mle_path"]),
                    "Log Likelihood / Objective": -mle_path_loss,
                    "Notes": "Optimized most likely hidden state path",
                }
            )
        benchmark_table = pd.DataFrame(benchmark_rows)

        st.session_state.results = {
            "model_df": model_df,
            "particles_pre": particles_pre,
            "particles_post": particles_post,
            "pf_loglik": pf_loglik,
            "kf_loglik": kf_loglik,
            "mle_path_loss": mle_path_loss,
            "mle_success": mle_success,
            "benchmark_table": benchmark_table,
            "params": {"phi": phi, "alpha": alpha, "sigma": sigma, "tau": tau, "N": int(N), "seed": int(seed)},
            "run_mle": run_mle,
        }

if st.session_state.results is None:
    st.info("Choose parameter settings in the sidebar, then click **Run model**.")
    st.stop()

results = st.session_state.results
model_df = results["model_df"]
particles_pre = results["particles_pre"]
particles_post = results["particles_post"]
run_mle = results["run_mle"]

st.divider()
st.header("Model Output")

c1, c2, c3 = st.columns(3)
c1.metric("Particle Filter Log Likelihood", f"{results['pf_loglik']:.2f}")
c2.metric("Kalman Filter Log Likelihood", f"{results['kf_loglik']:.2f}")
if run_mle:
    c3.metric("MLE State Path Success", str(results["mle_success"]))
else:
    c3.metric("MLE State Path", "Skipped")

# 1 Actual GDP vs trend
st.subheader("Actual GDP vs Polynomial Trend GDP")
fig1, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(model_df["date"], model_df["GDP"], label="Actual GDP")
ax1.plot(model_df["date"], model_df["trend_GDP_poly"], label=f"Polynomial Trend GDP, degree {degree}")
ax1.set_title("Actual GDP vs Polynomial Trend GDP")
ax1.set_xlabel("Date")
ax1.set_ylabel("GDP")
ax1.legend()
ax1.grid(True, alpha=0.25)
st.pyplot(fig1)
st.caption("This graph creates the proxy benchmark. The polynomial trend is treated as a smooth estimate of potential GDP, and the gap between actual GDP and this trend becomes the proxy output gap used for comparison.")

# 2 Proxy output gap
st.subheader("Proxy Output Gap")
fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(model_df["date"], model_df["X_t_proxy_output_gap"], label="Proxy Output Gap")
ax2.axhline(0, linestyle="--")
ax2.set_title("Proxy Output Gap: Actual GDP vs Polynomial Trend GDP")
ax2.set_xlabel("Date")
ax2.set_ylabel("Output Gap Proxy (%)")
ax2.legend()
ax2.grid(True, alpha=0.25)
st.pyplot(fig2)
st.caption("This shows the percent deviation of actual GDP from the polynomial trend. Positive values mean actual GDP is above trend; negative values mean actual GDP is below trend.")

# 3 PF vs proxy
st.subheader("Proxy Output Gap vs Particle Filter Estimate")
fig3, ax3 = plt.subplots(figsize=(11, 5))
ax3.plot(model_df["date"], model_df["X_t_proxy_output_gap"], label="Proxy Output Gap")
ax3.plot(model_df["date"], model_df["X_t_particle_filter"], label="Particle Filter Estimate")
ax3.axhline(0, linestyle="--")
ax3.set_title("Proxy Output Gap vs Particle Filter Estimate")
ax3.set_xlabel("Date")
ax3.set_ylabel("Output Gap (%)")
ax3.legend()
ax3.grid(True, alpha=0.25)
st.pyplot(fig3)
st.caption("The particle filter estimates the hidden output gap using centered GDP growth. This graph compares that model-based estimate against the proxy output gap from the polynomial trend.")

# 4 Particle distribution over time
st.subheader("Particle Filter Distribution Over Time")
q05 = np.percentile(particles_post, 5, axis=1)
q25 = np.percentile(particles_post, 25, axis=1)
q50 = np.percentile(particles_post, 50, axis=1)
q75 = np.percentile(particles_post, 75, axis=1)
q95 = np.percentile(particles_post, 95, axis=1)
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.fill_between(model_df["date"], q05, q95, alpha=0.20, label="5% to 95% particle range")
ax4.fill_between(model_df["date"], q25, q75, alpha=0.35, label="25% to 75% particle range")
ax4.plot(model_df["date"], q50, label="Particle median")
ax4.plot(model_df["date"], model_df["X_t_particle_filter"], label="Particle filter mean estimate")
ax4.plot(model_df["date"], model_df["X_t_proxy_output_gap"], label="Proxy Output Gap", alpha=0.8)
ax4.axhline(0, linestyle="--")
ax4.set_title("Particle Filter Distribution Over Time")
ax4.set_xlabel("Date")
ax4.set_ylabel("Output Gap (%)")
ax4.legend()
ax4.grid(True, alpha=0.25)
st.pyplot(fig4)
st.caption("The shaded regions show uncertainty across particles. A wide band means the filter is less certain about the hidden state; a narrow band means the particles are more concentrated around one estimate.")

# 5 Kalman interval
st.subheader("Kalman Filter Estimated Output Gap with Interval")
kf_upper = model_df["X_t_kalman_filter"] + 1.96 * np.sqrt(model_df["P_t_kalman_filter"])
kf_lower = model_df["X_t_kalman_filter"] - 1.96 * np.sqrt(model_df["P_t_kalman_filter"])
fig5, ax5 = plt.subplots(figsize=(12, 6))
ax5.plot(model_df["date"], model_df["X_t_kalman_filter"], label="Kalman Filter Estimate", linewidth=2)
ax5.fill_between(model_df["date"], kf_lower, kf_upper, alpha=0.2, label="95% Kalman Confidence Interval")
ax5.axhline(0, linestyle="--")
ax5.set_title("Kalman Filter Estimated Output Gap with Interval")
ax5.set_xlabel("Date")
ax5.set_ylabel("Estimated Hidden State")
ax5.legend()
ax5.grid(True, alpha=0.25)
st.pyplot(fig5)
st.caption("The Kalman filter is the linear-Gaussian benchmark. The confidence interval comes from the filter variance, showing how uncertain the Kalman estimate is at each date.")

# 6 Comparison all paths
st.subheader("Proxy Output Gap vs Estimated Hidden State Paths")
fig6, ax6 = plt.subplots(figsize=(12, 6))
ax6.plot(model_df["date"], model_df["X_t_proxy_output_gap"], label="Proxy Output Gap", linewidth=2)
ax6.plot(model_df["date"], model_df["X_t_particle_filter"], label="Particle Filter Estimate", linewidth=2)
ax6.plot(model_df["date"], model_df["X_t_kalman_filter"], label="Kalman Filter Estimate", linewidth=2)
if run_mle:
    ax6.plot(model_df["date"], model_df["X_t_mle_path"], label="MLE Most Likely State Path", linewidth=2)
ax6.axhline(0, linestyle="--")
ax6.set_title("Proxy Output Gap vs Estimated Hidden State Paths")
ax6.set_xlabel("Date")
ax6.set_ylabel("Output Gap / Hidden State")
ax6.legend()
ax6.grid(True, alpha=0.25)
st.pyplot(fig6)
st.caption("This compares all hidden-state estimates on the same scale. The particle filter and Kalman filter update sequentially through time, while the MLE path optimizes the entire state sequence at once.")

# 7 Pre vs post particles
st.subheader("Particles Before vs After Resampling")
max_t = len(model_df) - 1
t_default = min(50, max_t)
t_idx = st.slider("Select time index for particle histogram", min_value=0, max_value=max_t, value=t_default, step=1)
pre = particles_pre[t_idx]
post = particles_post[t_idx]
fig7, ax7 = plt.subplots(figsize=(11, 5))
ax7.hist(pre, bins=50, density=True, alpha=0.5, label="Before resampling")
ax7.hist(post, bins=50, density=True, alpha=0.5, label="After resampling")
ax7.axvline(model_df.loc[t_idx, "X_t_particle_filter"], linestyle="-.", label="PF estimate")
ax7.axvline(model_df.loc[t_idx, "X_t_proxy_output_gap"], linestyle="--", label="Proxy output gap")
ax7.set_title(f"Particles Before vs After Resampling | Date: {model_df.loc[t_idx, 'date'].date()}")
ax7.set_xlabel("Possible hidden state value x_t")
ax7.set_ylabel("Density")
ax7.legend()
ax7.grid(True, alpha=0.25)
st.pyplot(fig7)
st.caption("Before resampling, particles represent many possible hidden states after prediction and weighting. After resampling, particles with higher likelihood are copied more often, so the distribution concentrates around values that better explain the observed GDP growth.")

# Benchmark table
st.subheader("Benchmark Table")
st.dataframe(results["benchmark_table"], use_container_width=True, hide_index=True)
st.caption("RMSE compares each estimated hidden state to the proxy output gap. Log likelihood/objective values are included as model fit diagnostics, but they are not all directly comparable because the methods optimize different objects.")

st.subheader("Download Results")
csv = model_df.to_csv(index=False).encode("utf-8")
st.download_button("Download model results as CSV", data=csv, file_name="particle_filter_results.csv", mime="text/csv")
