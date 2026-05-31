import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize


def prepare_gdp_data(csv_file, degree=5):
    """Read GDP data, estimate polynomial log-GDP trend, and build proxy output gap."""
    gdp = pd.read_csv(csv_file)
    if "observation_date" not in gdp.columns or "GDP" not in gdp.columns:
        raise ValueError("CSV must contain columns named 'observation_date' and 'GDP'.")

    gdp["date"] = pd.to_datetime(gdp["observation_date"])
    gdp["GDP"] = pd.to_numeric(gdp["GDP"], errors="coerce")
    gdp = gdp[["date", "GDP"]].dropna().sort_values("date").reset_index(drop=True)

    gdp["t"] = np.arange(len(gdp))
    gdp["log_GDP"] = np.log(gdp["GDP"])

    X = pd.DataFrame({f"t_power_{i}": gdp["t"] ** i for i in range(1, degree + 1)})
    X = sm.add_constant(X)
    y = gdp["log_GDP"]
    poly_model = sm.OLS(y, X).fit()

    gdp["log_trend_GDP_poly"] = poly_model.fittedvalues
    gdp["trend_GDP_poly"] = np.exp(gdp["log_trend_GDP_poly"])

    gdp["X_t_proxy_output_gap"] = 100 * (gdp["GDP"] - gdp["trend_GDP_poly"]) / gdp["trend_GDP_poly"]
    gdp["Y_t_gdp_growth"] = 100 * gdp["GDP"].pct_change()
    gdp["Y_t_gdp_growth_annualized"] = 100 * ((gdp["GDP"] / gdp["GDP"].shift(1)) ** 4 - 1)

    model_df = gdp[[
        "date",
        "GDP",
        "trend_GDP_poly",
        "X_t_proxy_output_gap",
        "Y_t_gdp_growth",
        "Y_t_gdp_growth_annualized",
    ]].dropna().reset_index(drop=True)

    return model_df, poly_model


def estimate_proxy_parameters(model_df):
    """Estimate phi, alpha, sigma, and tau using the proxy output gap approach."""
    x_phi = model_df["X_t_proxy_output_gap"].dropna().reset_index(drop=True)
    ar1_model = AutoReg(x_phi, lags=1, trend="n", old_names=False)
    ar1_result = ar1_model.fit()
    phi = float(ar1_result.params.iloc[0])

    alpha_df = model_df[["X_t_proxy_output_gap", "Y_t_gdp_growth"]].dropna()
    x_alpha = alpha_df["X_t_proxy_output_gap"]
    y_alpha = alpha_df["Y_t_gdp_growth"]
    y_alpha_centered = y_alpha - y_alpha.mean()
    alpha_model = sm.OLS(y_alpha_centered, x_alpha)
    alpha_result = alpha_model.fit()
    alpha = float(alpha_result.params.iloc[0])

    sigma2 = float(np.mean(ar1_result.resid ** 2))
    sigma = float(np.sqrt(sigma2))
    tau2 = float(np.mean(alpha_result.resid ** 2))
    tau = float(np.sqrt(tau2))

    return {
        "phi": phi,
        "alpha": alpha,
        "sigma": sigma,
        "tau": tau,
        "sigma2": sigma2,
        "tau2": tau2,
    }


def particle_filter(Y, phi, alpha, sigma, tau, N=5000, init_mean=0.0, init_sd=5.0, store_paths=False):
    """Sequential Monte Carlo particle filter for the linear state-space model."""
    Y = pd.Series(Y).reset_index(drop=True)
    T = len(Y)
    tau = max(float(tau), 1e-8)
    sigma = max(float(sigma), 1e-8)
    init_sd = max(float(init_sd), 1e-8)

    particles = np.random.normal(init_mean, init_sd, N)
    x_hat = np.zeros(T)
    y_hat = np.zeros(T)
    loglik = 0.0

    particles_pre = [] if store_paths else None
    particles_post = [] if store_paths else None

    for t in range(T):
        if store_paths:
            particles_pre.append(particles.copy())

        particles = phi * particles + np.random.normal(0, sigma, N)
        residual = Y.iloc[t] - alpha * particles
        likelihoods = (1 / (np.sqrt(2 * np.pi) * tau)) * np.exp(-0.5 * (residual / tau) ** 2)
        likelihoods += 1e-300

        loglik += np.log(np.mean(likelihoods))
        weights = likelihoods / np.sum(likelihoods)

        x_hat[t] = np.sum(weights * particles)
        y_hat[t] = alpha * x_hat[t]

        idx = np.random.choice(np.arange(N), size=N, replace=True, p=weights)
        particles = particles[idx]

        if store_paths:
            particles_post.append(particles.copy())

    if store_paths:
        particles_pre = np.vstack(particles_pre)
        particles_post = np.vstack(particles_post)

    return x_hat, y_hat, particles_pre, particles_post, float(loglik)


def kalman_filter(Y, phi, alpha, sigma, tau, init_mean=0.0, init_var=1.0):
    """Kalman filter benchmark for the same linear Gaussian model."""
    Y = pd.Series(Y).dropna().reset_index(drop=True)
    T = len(Y)
    sigma = max(float(sigma), 1e-8)
    tau = max(float(tau), 1e-8)
    init_var = max(float(init_var), 1e-8)

    x_filt = np.zeros(T)
    y_pred = np.zeros(T)
    P_filt = np.zeros(T)

    x_prev = init_mean
    P_prev = init_var
    loglik = 0.0

    for t in range(T):
        x_pred = phi * x_prev
        P_pred = phi**2 * P_prev + sigma**2
        y_hat = alpha * x_pred
        S = alpha**2 * P_pred + tau**2
        S = max(float(S), 1e-12)

        K = P_pred * alpha / S
        innovation = Y.iloc[t] - y_hat
        x_new = x_pred + K * innovation
        P_new = (1 - K * alpha) * P_pred
        P_new = max(float(P_new), 1e-12)

        x_filt[t] = x_new
        y_pred[t] = alpha * x_new
        P_filt[t] = P_new

        loglik += -0.5 * (np.log(2 * np.pi) + np.log(S) + innovation**2 / S)

        x_prev = x_new
        P_prev = P_new

    return x_filt, y_pred, P_filt, float(loglik)


def mle_state_path(Y, phi, alpha, sigma, tau, init_mean=0.0, init_sd=5.0, maxiter=250):
    """Estimate the most likely full hidden state path for fixed parameters."""
    Y = pd.Series(Y).dropna().reset_index(drop=True)
    T = len(Y)
    sigma = max(float(sigma), 1e-8)
    tau = max(float(tau), 1e-8)
    init_sd = max(float(init_sd), 1e-8)

    if abs(alpha) < 1e-8:
        x0_guess = np.zeros(T) + init_mean
    else:
        x0_guess = Y / alpha

    def negative_loglik_state_path(x_path):
        obs_residuals = Y.values - alpha * x_path
        obs_loss = np.sum(0.5 * np.log(2 * np.pi * tau**2) + 0.5 * (obs_residuals / tau) ** 2)
        init_loss = 0.5 * np.log(2 * np.pi * init_sd**2) + 0.5 * ((x_path[0] - init_mean) / init_sd) ** 2
        state_residuals = x_path[1:] - phi * x_path[:-1]
        state_loss = np.sum(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * (state_residuals / sigma) ** 2)
        return obs_loss + init_loss + state_loss

    result = minimize(negative_loglik_state_path, x0_guess, method="BFGS", options={"maxiter": maxiter})
    x_mle = result.x
    y_mle = alpha * x_mle
    return x_mle, y_mle, float(result.fun), bool(result.success)


def rmse(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))
