import glob
import hashlib
import json
import os

import mlflow
from mlflow.tracking import MlflowClient

# ---------- 1. Où MLflow stocke tout : une base SQLite locale ----------
# (le Model Registry a besoin d'une base de données, pas seulement de fichiers)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("progedo-codage-enquete")

# ---------- 2. Ce qu'on sait de chaque modèle (sa « fiche ») ----------
MODELES = {
    "q4": {"fichier": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "quantification": "Q4_K_M"},
    "q8": {"fichier": "qwen2.5-1.5b-instruct-q8_0.gguf", "quantification": "Q8_0"},
}
COMMUN = {
    "modele_base": "Qwen2.5-1.5B-Instruct",
    "source": "huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "licence": "Apache-2.0",
    "moteur": "llama.cpp b11946 (Vulkan)",
    "gpu": "GTX 1650 Ti 4 Go",
    "contexte": 4096,
    "temperature": 0,
    "seed": 42,
}


def sha256(path):
    """Empreinte du fichier : prouve exactement quel modèle a produit les résultats."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloc in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloc)
    return h.hexdigest()


empreintes = {q: sha256(m["fichier"]) for q, m in MODELES.items() if os.path.exists(m["fichier"])}

# ---------- 3. Un « run » MLflow par expérience déjà réalisée ----------
for chemin in sorted(glob.glob("resultats/*_metrics.json")):
    with open(chemin, encoding="utf-8") as f:
        res = json.load(f)
    quant = res["quant"]
    codebook = res.get("codebook", "complet")   # les premiers runs n'avaient pas ce champ
    nom = f"{quant}-{codebook}-run{res['run']}"

    with mlflow.start_run(run_name=nom):
        mlflow.log_params({**COMMUN, **MODELES[quant], "codebook": codebook, "passage": res["run"],
                           "sha256": empreintes.get(quant, "inconnu")})
        mlflow.log_metrics({k: res[k] for k in
                            ["accuracy", "kappa", "latence_moyenne_s", "duree_totale_s", "tokens_generes"]})
        mlflow.log_artifact(chemin.replace("_metrics.json", "_predictions.csv"))
        if os.path.exists("bench_q4_vs_q8.md"):
            mlflow.log_artifact("bench_q4_vs_q8.md")
    print("run enregistré :", nom)


# ---------- 4. Le Model Registry : versionner le modèle « servi » ----------
# On n'enregistre pas le fichier de 1 Go : on enregistre une référence (fiche + empreinte)
# et un petit client qui appelle le serveur. C'est ce qu'un registre de modèles doit tracer.
class ClientCodage(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        from openai import OpenAI
        client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="pas-de-cle")
        sorties = []
        for texte in model_input["reponse"]:
            r = client.chat.completions.create(
                model="qwen", temperature=0, seed=42,
                messages=[{"role": "user", "content": texte}])
            sorties.append(r.choices[0].message.content)
        return sorties


fiche = {**COMMUN, **MODELES["q4"], "sha256": empreintes.get("q4", "inconnu"),
         "justification": "même accuracy que Q8 sur le test, 40 % de VRAM en moins, génération 53 % plus rapide"}
with open("fiche_modele_q4.json", "w", encoding="utf-8") as f:
    json.dump(fiche, f, ensure_ascii=False, indent=2)

NOM_REGISTRE = "qwen2.5-1.5b-codage-enquete"
with mlflow.start_run(run_name="enregistrement-q4"):
    mlflow.log_params({k: v for k, v in fiche.items() if k != "justification"})
    info = mlflow.pyfunc.log_model(
        name="modele",
        python_model=ClientCodage(),
        artifacts={"fiche": "fiche_modele_q4.json"},
        pip_requirements=["openai"],
        registered_model_name=NOM_REGISTRE,
    )

client = MlflowClient()
version = max(int(v.version) for v in client.search_model_versions(f"name='{NOM_REGISTRE}'"))
client.set_registered_model_alias(NOM_REGISTRE, "production", str(version))
print(f"\nModèle {NOM_REGISTRE} version {version} enregistré avec l'alias 'production'.")
print("Ouvre l'interface :  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")