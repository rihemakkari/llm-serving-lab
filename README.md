\# llm-serving-lab



Servir un petit modèle de langue en local, l'interroger comme un chercheur et mesurer ce qu'on obtient : vitesse, qualité de codage d'enquete et reproductibilité, le tout tracé dans MLflow.



Projet réalisé en octobre 2026 pour apprendre llama.cpp et MLflow à partir de zéro.



\## Ce qui a été fait

\- Serving : `llama-server` (llama.cpp b11946, backend Vulkan) sert \*\*Qwen2.5-1.5B-Instruct\*\* (GGUF) sur une GTX 1650 Ti de 4 Go, via une \*\*API compatible OpenAI\*\*.

\- Deux quantifications comparées : `Q4\_K\_M` (1,04 Go) et `Q8\_0` (1,76 Go). Licence vérifiée : \*\*Apache 2.0\*\* (attention, dans la même famille, le 3B est sous licence de recherche Qwen).

\- Benchmark avec `llama-bench` et un script de streaming (TTFT, tokens/s).

\- Tâche SHS: codage de 20 réponses ouvertes fictives à « Pourquoi faites-vous ou non confiance aux institutions ? » en 4 catégories, avec sortie JSON contrainte par schéma, comparée à un codage de référence (accuracy, kappa de Cohen).

\- MLflow : un run par expérience (paramètres, métriques, prédictions, empreinte SHA-256 du modèle) et le modèle retenu enregistré dans le Model Registry avec l'alias `production`.



\## Résultats



\### Vitesse (llama-bench, 5 répétitions, tout sur GPU)

| | Q4\_K\_M | Q8\_0 |

|---|---|---|

| Taille | 1,04 Go | 1,76 Go |

| Génération (tg128) | \*\*98 t/s\*\* | 64 t/s |

| Traitement du prompt (pp512) | 1 037 t/s | 883 t/s |



TTFT mesuré avec le client OpenAI en streaming : \*\*1,04 s à froid\*\*, \*\*0,30 s à chaud\*\*.



\### Codage des réponses d'enquête (20 réponses, température 0, seed 42)

| Modèle | Livre de codes | Accuracy | Kappa | Latence / réponse |

|---|---|---|---|---|

| Q4\_K\_M | complet (×2) | 1,00 | 1,00 | 0,45 s |

| Q8\_0 | complet (×2) | 1,00 | 1,00 | 0,42 à 0,49 s |

| Q4\_K\_M | minimal | 0,95 | 0,93 | 0,59 s |

| Q8\_0 | minimal | 0,95 | 0,93 | 0,73 s |



!\[Comparaison des runs dans MLflow](docs/mlflow-compare.png)



\## Ce que j'en retiens

\- La génération est limitée par la bande passante mémoire : Q4 et Q8 lisent la mémoire au même débit (\~110 à 120 Go/s) ; le Q4 est 53 % plus rapide parce que ses poids sont plus légers. Le traitement du prompt, parallèle, est limité par le calcul : l'écart n'est que de 17 %.

\- La métrique utile dépend de la tâche : pour le codage d'enquête (long prompt, sortie courte), l'avantage du Q4 en génération se voit peu ; son vrai gain est la mémoire (40 % de VRAM en moins).

\- Même score, erreurs différentes : avec le livre de codes minimal, Q8 code « Je n'ai plus confiance depuis que ma retraite a été repoussée… » en \*confiance\* (négation ratée), Q4 en \*méfiance politique\* (erreur défendable). Il faut lire les erreurs, pas seulement la métrique.

\- Les consignes comptent autant que le modèle : avec un livre de codes explicite, les deux modèles ne se trompent plus.

\- Reproductibilité : sorties courtes et contraintes identiques d'un passage à l'autre ; en génération libre, la sortie varie même à température 0. D'où l'intérêt de tracer modèle, quantification, paramètres et sorties.

\- Démarrage à froid : la première requête est 3 à 4 fois plus lente ; en production, on envoie une requête de chauffe.



\## Limites

\- 20 réponses fictives, rédigées en même temps que le livre de codes : le test est trop facile et ne mesure pas la fiabilité sur de vraies données. Une vraie évaluation demanderait un double codage humain indépendant et la comparaison du modèle à l'accord inter-codeurs.

\- Un seul petit modèle, une seule carte graphique grand public, un seul utilisateur à la fois.

\- Pour un service partagé entre laboratoires, il faudrait vLLM sur GPU serveur (PagedAttention, continuous batching, métriques Prometheus), une clé d'API par utilisateur, des quotas, un reverse proxy HTTPS et des logs d'accès.



\## Incidents rencontrés

\- Téléchargement automatique (`-hf`) qui échouait sans message → téléchargement manuel avec `curl`.

\- Binaire mis en quarantaine par l'antivirus → exception sur le dossier du projet.

\- Port 8080 occupé par le listener Oracle (TNSLSNR) → service déplacé sur 8081.



\## Reproduire

```powershell

winget install llama.cpp

curl.exe -L -o qwen2.5-1.5b-instruct-q4\_k\_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4\_k\_m.gguf

llama-server -m .\\qwen2.5-1.5b-instruct-q4\_k\_m.gguf -ngl 99 -c 4096 --port 8081



python -m venv .venv; .venv\\Scripts\\Activate.ps1; pip install -r requirements.txt

python stream\_test.py

python classify.py --quant q4 --run 1

python log\_mlflow.py

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1

```



\## Fichiers

\- `stream\_test.py` : appel en streaming, mesure du TTFT et des tokens/s

\- `classify.py` : codage des réponses, sortie JSON contrainte, accuracy, kappa, matrice de confusion

\- `log\_mlflow.py` : runs MLflow, empreintes SHA-256, Model Registry et alias `production`

\- `reponses.csv` : les 20 réponses fictives et leur codage de référence

\- `resultats/` : prédictions et métriques de chaque passage

