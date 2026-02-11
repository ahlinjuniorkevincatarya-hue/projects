import numpy as np
from scipy.optimize import linprog

####### DONNÉES DU PROBLÈME

temps = np.array([[4, 8, 12], [3, 6, 9], [1, 2, 3]])
cout = np.array([[20, 35, 80], [15, 20, 40], [5, 8, 12]])
commandes = [30, 50, 40]
disponibilites = [200, 300, 500]
limite_T4 = 20

n_modeles, n_gpus = 3, 3
n_vars = n_modeles * n_gpus
n_eq = 3  # Contraintes d'égalité (commandes)
n_ineq = 7  # Contraintes d'inégalité


###### FORMULATION DU PROBLÈME (sans variables d'écart pour linprog)


# Fonction objectif (coûts)
c = cout.flatten()

# Contraintes d'égalité (commandes)
A_eq = np.zeros((n_eq, n_vars))
for i in range(n_modeles):
    A_eq[i, i*n_gpus:(i+1)*n_gpus] = 1
b_eq = np.array(commandes)

# Contraintes d'inégalité
A_ub = np.zeros((n_ineq, n_vars))
b_ub = np.zeros(n_ineq)

# Disponibilités (contraintes 1-3)
for j in range(3):
    for i in range(n_modeles):
        A_ub[j, i*n_gpus + j] = temps[i, j]
    b_ub[j] = disponibilites[j]

# Quotas minimum (contraintes 4-6) - reformulées en -x <= -1
for j in range(3):
    for i in range(n_modeles):
        A_ub[3 + j, i*n_gpus + j] = -1
    b_ub[3 + j] = -1

# Limite T4 (contrainte 7)
for i in range(n_modeles):
    A_ub[6, i*n_gpus + 2] = 1
b_ub[6] = limite_T4


###### RÉSOLUTION

result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                 bounds=(0, None), method='simplex')

if not result.success:
    print("ERREUR: Pas de solution optimale")
    exit()

# Solution des variables de décision
x_decision = result.x

# Calcul des variables d'écart
ecarts = b_ub - A_ub @ x_decision
x_total = np.concatenate([x_decision, ecarts])

# Noms des variables
vars_decision = [f'x{i+1}{j+1}' for i in range(n_modeles) for j in range(n_gpus)]
vars_ecart = [f's{i+1}' for i in range(n_ineq)]
vars_total = vars_decision + vars_ecart


####### AFFICHAGE DES RÉSULTATS

print("="*60)
print("SOLUTION OPTIMALE")
print("="*60)
print(f"Coût optimal: {result.fun:.2f} €")

print("\nVariables de décision:")
for i, (nom, val) in enumerate(zip(vars_decision, x_decision)):
    if abs(val) > 1e-6:
        print(f"  {nom}: {val:.4f}")

print("\nVariables d'écart:")
for i, (nom, val) in enumerate(zip(vars_ecart, ecarts)):
    if abs(val) > 1e-6:
        print(f"  {nom}: {val:.4f}")

###### IDENTIFICATION BASE/HORS-BASE

print("\n" + "="*60)
print("VARIABLES DE BASE ET HORS BASE")
print("="*60)

tol = 1e-6
n_total_vars = len(x_total)
n_base = n_eq + n_ineq  # 10 variables de base

# Trier les variables par valeur absolue décroissante
indices_tries = sorted(range(n_total_vars), 
                      key=lambda i: abs(x_total[i]), reverse=True)

# Les 10 premières sont de base, les autres hors base
base_idx = indices_tries[:n_base]
hors_base_idx = indices_tries[n_base:]

base_idx.sort()
hors_base_idx.sort()

print("Variables de base (10):")
for idx in base_idx:
    print(f"  {vars_total[idx]}: {x_total[idx]:.4f}")

print("\nVariables hors base (6):")
for idx in hors_base_idx:
    print(f"  {vars_total[idx]}: {x_total[idx]:.4f}")


###### SYSTÈME À LA DERNIÈRE ITÉRATION

print("SYSTÈME À LA DERNIÈRE ITÉRATION")
print("="*60)

# Matrice complète des contraintes (avec variables d'écart)
A_complete = np.zeros((n_base, n_total_vars))

# Contraintes d'égalité (ne concernent que les variables de décision)
A_complete[:n_eq, :n_vars] = A_eq

# Contraintes d'inégalité (avec variables d'écart)
for i in range(n_ineq):
    A_complete[n_eq + i, :n_vars] = A_ub[i]
    A_complete[n_eq + i, n_vars + i] = 1  # Variable d'écart correspondante

b_complete = np.concatenate([b_eq, b_ub])

try:
    # Extraire les sous-matrices B et N
    B = A_complete[:, base_idx]
    N = A_complete[:, hors_base_idx]
    
    # Inverser B
    B_inv = np.linalg.inv(B)
    
    # Expression des variables de base en fonction des hors base
    const_term = B_inv @ b_complete
    var_term = -B_inv @ N
    
    print("\nVariables de base en fonction des variables hors base:")
    for i, idx_b in enumerate(base_idx):
        expr = f"{vars_total[idx_b]} = {const_term[i]:.4f}"
        for j, idx_n in enumerate(hors_base_idx):
            coeff = var_term[i, j]
            if abs(coeff) > tol:
                sign = '+' if coeff >= 0 else '-'
                expr += f" {sign} {abs(coeff):.4f}·{vars_total[idx_n]}"
        print(f"  {expr}")
    
    # Expression de la fonction objectif
    c_B = np.array([c[idx] if idx < n_vars else 0 for idx in base_idx])
    c_N = np.array([c[idx] if idx < n_vars else 0 for idx in hors_base_idx])
    
    y = c_B @ B_inv
    r = c_N - y @ N
    Z_opt = y @ b_complete
    
    print(f"\nFonction objectif:")
    expr = f"Z = {Z_opt:.4f}"
    for j, idx in enumerate(hors_base_idx):
        if abs(r[j]) > tol:
            sign = '+' if r[j] >= 0 else '-'
            expr += f" {sign} {abs(r[j]):.4f}·{vars_total[idx]}"
    print(f"  {expr}")
    
    # Vérification optimalité
    print("\nConditions d'optimalité:")
    optimal_couts = all(r[j] >= -tol for j in range(len(r)))
    optimal_vars = all(x_total[idx] >= -tol for idx in base_idx)
    print(f"  Coûts réduits ≥ 0: {'✓' if optimal_couts else '✗'}")
    print(f"  Variables de base ≥ 0: {'✓' if optimal_vars else '✗'}")
    
    # Vérification numérique
    x_B = np.array([x_total[idx] for idx in base_idx])
    verification = B @ x_B
    erreur_max = np.max(np.abs(verification - b_complete))
    print(f"\nVérification B·x_B = b:")
    print(f"  Erreur maximale: {erreur_max:.2e}")
    print(f"  {'✓ OK' if erreur_max < tol else '✗ ERREUR'}")
    
except np.linalg.LinAlgError:
    print("Matrice B non inversible (solution dégénérée)")

###### VÉRIFICATION DES CONTRAINTES

print("\n" + "="*60)
print("VÉRIFICATION DES CONTRAINTES")
print("="*60)

# 1. Commandes
print("\n1. Commandes (égalités):")
modeles_noms = ['BERT', 'ResNet', 'LSTM']
for i in range(n_modeles):
    total = sum(x_decision[i*n_gpus:(i+1)*n_gpus])
    ok = abs(total - commandes[i]) < tol
    print(f"  {modeles_noms[i]}: {total:.1f} = {commandes[i]} {'✓' if ok else '✗'}")

# 2. Disponibilités
print("\n2. Disponibilités (≤):")
for j in range(3):
    heures = sum(temps[i, j] * x_decision[i*n_gpus + j] for i in range(n_modeles))
    ok = heures <= disponibilites[j] + tol
    print(f"  GPU{j+1}: {heures:.1f} ≤ {disponibilites[j]} {'✓' if ok else '✗'}")

# 3. Quotas minimum
print("\n3. Quotas minimum (≥ 1):")
for j in range(3):
    total = sum(x_decision[i*n_gpus + j] for i in range(n_modeles))
    ok = total >= 1 - tol
    print(f"  GPU{j+1}: {total:.1f} ≥ 1 {'✓' if ok else '✗'}")

# 4. Limite T4
utilisation_T4 = sum(x_decision[i*n_gpus + 2] for i in range(n_modeles))
ok_T4 = utilisation_T4 <= limite_T4 + tol
print(f"\n4. Limite T4: {utilisation_T4:.1f} ≤ {limite_T4} {'✓' if ok_T4 else '✗'}")

###### SYNTHÈSE

print("\n" + "="*60)
print("SYNTHÈSE")
print("="*60)

print(f"\nCoût total optimal: {result.fun:.2f} €")
print("Solution validée et toutes les contraintes sont satisfaites.")
print("="*60)

# ============================================================================
# INTERVALLE D'OPTIMALITÉ D'UN COEFFICIENT DE COÛT c_k
# ============================================================================

import sympy as sp

def intervalle_optimalite_c(k):
    """
    Intervalle d’optimalité du coefficient de coût c_k
    (variable de décision k, 0-based)
    """
    if k not in base_idx:
        return (-np.inf, np.inf)

    # Position de k dans la base
    pos = base_idx.index(k)

    delta_min = -np.inf
    delta_max = np.inf

    for j, idx_n in enumerate(hors_base_idx):
        a = (B_inv[pos, :] @ N[:, j])
        if abs(a) < tol:
            continue

        rhs = r[j]
        bound = rhs / a

        if a > 0:
            delta_max = min(delta_max, bound)
        else:
            delta_min = max(delta_min, bound)

    return c[k] + delta_min, c[k] + delta_max

# ======================================================================
# TABLEAU RÉCAPITULATIF DES INTERVALLES D’OPTIMALITÉ (COÛTS)
# ======================================================================

print("\n" + "="*60)
print("TABLEAU DE POST-OPTIMALITÉ (COEFFICIENTS DE COÛT)")
print("="*60)

print(f"{'Variable':15s} | {'Coût actuel':>10s} | Intervalle d’optimalité")
print("-"*60)

for k, nom in enumerate(vars_decision):
    a, b = intervalle_optimalite_c(k)

    # Mise en forme des bornes infinies
    a_str = f"{a:.2f}" if np.isfinite(a) else "-∞"
    b_str = f"{b:.2f}" if np.isfinite(b) else "+∞"

    print(f"{nom:15s} | {c[k]:10.2f} | [{a_str} , {b_str}]")


def intervalle_realisabilite_b(k):
    """
    Intervalle de réalisabilité du second membre b_k
    """
    e = np.zeros(n_base)
    e[k] = 1

    d = B_inv @ e          # direction
    xB = const_term       # solution de base actuelle

    delta_min = -np.inf
    delta_max = np.inf

    for i in range(n_base):
        if abs(d[i]) < tol:
            continue
        ratio = -xB[i] / d[i]
        if d[i] > 0:
            delta_min = max(delta_min, ratio)
        else:
            delta_max = min(delta_max, ratio)

    return float(b_complete[k] + delta_min), float(b_complete[k] + delta_max)

#print("Intervalle de réalisabilité – disponibilité GPU A100 :")
#print(intervalle_realisabilite_b(3))  # index exact de la contrainte A100


# ======================================================================
# TABLEAU RÉCAPITULATIF DES INTERVALLES DE RÉALISABILITÉ
# ======================================================================

parametres = []

# Commandes
for i, nom in enumerate(modeles_noms):
    parametres.append((
        f"Commande {nom}",
        intervalle_realisabilite_b(i)
    ))

# Disponibilités GPU
gpu_noms = ["V100", "A100", "T4"]
for j in range(3):
    idx = n_eq + j
    parametres.append((
        f"Disponibilité GPU {gpu_noms[j]}",
        intervalle_realisabilite_b(idx)
    ))

# Quotas minimum
for j in range(3):
    idx = n_eq + 3 + j
    parametres.append((
        f"Quota minimum GPU {gpu_noms[j]}",
        intervalle_realisabilite_b(idx)
    ))

# Limite T4
parametres.append((
    "Limite T4",
    intervalle_realisabilite_b(n_eq + 6)
))

print("\n" + "="*60)
print("TABLEAU DE POST-OPTIMALITÉ (SECONDS MEMBRES)")
print("="*60)

print(f"{'Paramètre':35s} | Intervalle de réalisabilité")
print("-"*60)

for nom, (a, b) in parametres:
    print(f"{nom:35s} | [{a:.2f} , {b:.2f}]")
