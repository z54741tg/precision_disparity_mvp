import numpy as np
import pandas as pd


# ============================================================
# Synthetic dataset generator for precision disparity testing
# ============================================================

def generate_synthetic_linkage_data(N=2000, seed=123, scenario="ethnicity_bias"):
    """
    Generate synthetic dataset for testing precision disparity.
    Only TP and FP outcomes are generated.

    Scenarios:
        - baseline_fair
        - ethnicity_bias
        - age_bias
        - sex_bias
        - interaction_bias
    """

    np.random.seed(seed)

    # -----------------------
    #   Subgroup variables
    # -----------------------
    sex = np.random.choice(["Male", "Female"], N, p=[0.48, 0.52])

    age_band = np.random.choice(
        ["18–29", "30–44", "45–64", "65+"],
        N,
        p=[0.20, 0.35, 0.30, 0.15]
    )

    ethnicity = np.random.choice(
        ["White", "Black", "Asian", "Mixed", "Other"],
        N,
        p=[0.70, 0.10, 0.10, 0.05, 0.05]
    )

    id_quality = np.random.choice(
        ["High", "Low"],
        N,
        p=[0.7, 0.3]
    )

    # Output list of TP / FP classifications
    link_truth = []


    # ============================================================
    # Loop over each record and assign TP or FP
    # ============================================================
    for i in range(N):

        s = sex[i]
        a = age_band[i]
        e = ethnicity[i]
        q = id_quality[i]

        # Baseline precision ~85%
        p_tp = 0.85
        p_fp = 0.15


        # ------------------------------------------
        # Scenario-based FP/TP adjustments
        # ------------------------------------------
        if scenario == "baseline_fair":
            pass

        elif scenario == "ethnicity_bias":
            if e == "Asian":
                p_fp += 0.10
                p_tp -= 0.10
            elif e == "Black":
                p_fp += 0.05
                p_tp -= 0.05

        elif scenario == "age_bias":
            if a == "18–29":
                p_fp -= 0.05
                p_tp += 0.05
            elif a == "65+":
                p_fp += 0.07
                p_tp -= 0.07

        elif scenario == "sex_bias":
            if s == "Male":
                p_fp += 0.06
                p_tp -= 0.06

        elif scenario == "interaction_bias":
            if (s == "Male" and a == "18–29" and e == "Asian" and q == "Low"):
                p_fp += 0.15
                p_tp -= 0.15


        # ------------------------------------------
        # All scenarios: low ID quality → more FP
        # ------------------------------------------
        if q == "Low":
            p_fp += 0.05
            p_tp -= 0.05


        # ------------------------------------------
        # Normalise probabilities
        # ------------------------------------------
        total = p_tp + p_fp
        p_tp = p_tp / total
        p_fp = p_fp / total


        # ------------------------------------------
        # Draw the truth label
        # ------------------------------------------
        outcome = np.random.choice(["TP", "FP"], p=[p_tp, p_fp])
        link_truth.append(outcome)


    # ============================================================
    # Build final dataframe
    # ============================================================
    df = pd.DataFrame({
        "sex": sex,
        "age_band": age_band,
        "ethnicity": ethnicity,
        "id_quality": id_quality,
        "link_truth": link_truth
    })

    return df



# ============================================================
#                 Test run
# ============================================================

df_test = generate_synthetic_linkage_data(20, scenario="ethnicity_bias")

# Force output in ALL environments (CML/Jupyter/Terminal)
print("\nGenerated dataset:")
print(df_test)

print("\nFirst 5 rows:")
print(df_test.head())
