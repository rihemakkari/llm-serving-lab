import argparse
import csv
import json
import os
import time

from openai import OpenAI
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# ---------- 1. Paramètres de l'expérience (ligne de commande) ----------
parser = argparse.ArgumentParser()
parser.add_argument("--quant", required=True, help="étiquette du modèle, ex. q4 ou q8")
parser.add_argument("--run", type=int, default=1, help="numéro de passage (1, 2...)")
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--codebook", default="complet", choices=["complet", "minimal"])
args = parser.parse_args()

CATEGORIES = ["confiance", "méfiance politique", "méfiance économique", "indifférence"]

# ---------- 2. Le livre de codes (codebook) donné au modèle ----------
SYSTEM_PROMPT = """Tu codes des réponses ouvertes à la question d'enquête :
« Pourquoi faites-vous ou non confiance aux institutions ? »
Attribue UNE seule catégorie :
- confiance : la personne exprime de la confiance envers les institutions.
- méfiance politique : la méfiance vise les élus, les partis, le vote, la corruption, la représentation.
- méfiance économique : la méfiance vient des impôts, des prix, des retraites, des banques, des entreprises, du budget.
- indifférence : la personne ne s'y intéresse pas ou n'a pas d'avis.
Réponds uniquement en JSON."""
SYSTEM_PROMPT_MINIMAL = """Classe cette réponse d'enquête dans une catégorie parmi :
confiance, méfiance politique, méfiance économique, indifférence.
Réponds uniquement en JSON."""

if args.codebook == "minimal":
    SYSTEM_PROMPT = SYSTEM_PROMPT_MINIMAL

# ---------- 3. Le schéma JSON qui CONTRAINT la sortie ----------
# llama.cpp transforme ce schéma en grammaire : le modèle ne peut produire
# qu'un JSON valide dont la catégorie est l'une des quatre (enum).
SCHEMA = {
    "type": "object",
    "properties": {"categorie": {"type": "string", "enum": CATEGORIES}},
    "required": ["categorie"],
}

client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="pas-de-cle")

with open("reponses.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

gold, pred, out_rows = [], [], []
total_completion_tokens = 0
start = time.perf_counter()

for row in rows:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=args.quant,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["reponse"]},
        ],
        temperature=0,
        seed=42,
        response_format={"type": "json_schema", "json_schema": {"name": "codage", "schema": SCHEMA}},
    )
    latency = time.perf_counter() - t0
    raw = resp.choices[0].message.content
    try:
        label = json.loads(raw)["categorie"]
    except (json.JSONDecodeError, KeyError):
        label = "INVALIDE"           # ne devrait pas arriver grâce au schéma
    total_completion_tokens += resp.usage.completion_tokens

    gold.append(row["etiquette"])
    pred.append(label)
    ok = "✔" if label == row["etiquette"] else "✘"
    print(f"{ok} #{row['id']:>2}  attendu={row['etiquette']:<20} modèle={label:<20} {latency:.2f}s")
    out_rows.append({**row, "prediction": label, "latence_s": round(latency, 3)})

elapsed = time.perf_counter() - start

# ---------- 4. Les métriques ----------
accuracy = accuracy_score(gold, pred)
kappa = cohen_kappa_score(gold, pred)
metrics = {
    "accuracy": round(accuracy, 3),
    "kappa": round(kappa, 3),
    "latence_moyenne_s": round(elapsed / len(rows), 3),
    "duree_totale_s": round(elapsed, 2),
    "tokens_generes": total_completion_tokens,
}

print("\n--- Résultats", args.quant, "passage", args.run, "---")
for k, v in metrics.items():
    print(f"{k:<20}: {v}")
print("\nMatrice de confusion (lignes = attendu, colonnes = modèle) :")
print("ordre :", CATEGORIES)
print(confusion_matrix(gold, pred, labels=CATEGORIES))

# ---------- 5. Sauvegarde (pour comparer les passages et pour MLflow) ----------
os.makedirs("resultats", exist_ok=True)
base = f"resultats/{args.quant}_{args.codebook}_run{args.run}"
with open(base + "_predictions.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
    w.writeheader()
    w.writerows(out_rows)
with open(base + "_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"quant": args.quant, "codebook": args.codebook, "run": args.run, **metrics}, f, ensure_ascii=False, indent=2)
print(f"\nSauvegardé : {base}_predictions.csv et {base}_metrics.json")