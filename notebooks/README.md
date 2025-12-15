# Notebooks

Ce dossier contient les notebooks utilisés pour l'extraction et le calcul des métriques.

## Ordre d'exécution
1. `1.0-gep-data-extraction-osm.ipynb`  
   - Récupère les données brutes depuis la source externe OSM (OpenStreetMap).  
   - Génère un fichier JSON dans `data/raw/` avec un timestamp "%Y%m%d_%H%M%S".

2. `2.0-gep-data-exploration.ipynb`  
   - Lit le fichier généré par `1.0-gep-data-extraction-osm.ipynb`.  
   - Calcule les métriques (pourcentage de données manquantes, histogrammes, etc.).  
   - Produit des visualisations interactives avec Plotly.

## Données
- Les fichiers volumineux sont stockés dans `data/raw/` (non versionnés, voir `.gitignore`).  


## Environnement
Voir le fichier [text](../requirements.txt) pour recréer l'environnement.

### Installation avec pip

1. Créez un environnement virtuel :
   ```bash
   python -m venv venv

2. Activez l'environnement :
   ```bash
   source venv/bin/activate # Linux/Mac
   venv\Scripts\activate # Windows

3. Installez les packages :
   ```bash
   pip install -r requirements.txt

4. relier avec jupyter :
   ```bash
   python -m ipykernel install --user --name=venv --display-name venv

5. Lance Jupyter et choisi le kernel venv

6. Désactivez l'environnement :
   ```bash
   deactivate