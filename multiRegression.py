import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dragon_data (1).csv")

# Clean
df = df.drop(df[df["AGE"] < 0].index)
df["SPC"] = df["SPC"].replace({"Wyvernn": "Wyvern"})
df = pd.get_dummies(df, columns=["SPC"], dtype=int)

# ── Pre-transformation (log / sqrt on raw features) ───────────────────────────
df["log_MASS"]   = np.log1p(df["MASS"])
df["log_AGE"]    = np.log1p(df["AGE"])
df["log_SPD"]    = np.log1p(df["SPD"])
df["sqrt_WSP"]   = np.sqrt(df["WSP"])
df["AGE_x_WSP"]  = df["log_AGE"] * df["sqrt_WSP"]   # interaction term
df["mass_per_wsp"] = df["log_MASS"] / (df["WSP"] + 1)  # body density proxy

y = df["FHO"]

subsets = {
    # ── Raw feature models ────────────────────────────────────────────────────
    "All features":               ["AGE", "MASS", "WSP", "HID", "SPD",
                                   "SPC_Dragon", "SPC_Hydra", "SPC_Wyvern"],
    "AGE, WSP, MASS, SPD":        ["AGE", "MASS", "WSP", "SPD"],
    "AGE, WSP, MASS, SPD, HID":   ["AGE", "MASS", "WSP", "SPD", "HID"],
    "AGE, WSP, MASS, SPD, SPC":   ["AGE", "MASS", "WSP", "SPD",
                                   "SPC_Dragon", "SPC_Hydra", "SPC_Wyvern"],
    "WSP, MASS, SPD, HID":        ["MASS", "WSP", "HID", "SPD"],
}

post_subsets = {
    # ── Transformed feature models ────────────────────────────────────────────
    "Log transforms only":        ["log_AGE", "log_MASS", "sqrt_WSP",
                                   "HID", "log_SPD",
                                   "SPC_Dragon", "SPC_Hydra", "SPC_Wyvern"],
    "Log + interaction":          ["log_AGE", "log_MASS", "sqrt_WSP",
                                   "HID", "log_SPD", "AGE_x_WSP",
                                   "SPC_Dragon", "SPC_Hydra", "SPC_Wyvern"],
    "Log + interaction + density":["log_AGE", "log_MASS", "sqrt_WSP",
                                   "HID", "log_SPD", "AGE_x_WSP",
                                   "mass_per_wsp",
                                   "SPC_Dragon", "SPC_Hydra", "SPC_Wyvern"],
    "Log, no SPC":                ["log_AGE", "log_MASS", "sqrt_WSP",
                                   "HID", "log_SPD", "AGE_x_WSP"],
}

def run_models(subsets_dict, label):
    print(f"\n{'='*20} {label} {'='*20}")
    for name, features in subsets_dict.items():
        X = df[features]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ── Post-transformation: StandardScaler applied after split ───────────
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)      # fit on train only

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        train_r2 = r2_score(y_train, model.predict(X_train_scaled))
        test_r2  = r2_score(y_test,  model.predict(X_test_scaled))
        rmse     = np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled)))

        print(f"  {name:<45}  Train R²: {train_r2:.4f}  Test R²: {test_r2:.4f}  RMSE: {rmse:.1f}")

run_models(subsets,      "Pre-Transformation (raw features, scaled)")
run_models(post_subsets, "Post-Transformation (log/sqrt/interaction, scaled)")