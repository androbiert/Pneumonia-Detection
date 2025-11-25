# 🩺 Pneumonia Detection Project

This project uses **Deep Learning (PyTorch)** to detect pneumonia from chest X-ray images.  


---

##  Dataset Download (IMPORTANT)

** Les données ne sont PAS incluses dans ce repository GitHub** (trop volumineuses ~2.5 GB)

Vous devez télécharger le dataset depuis Kaggle:
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Instructions détaillées**: [`data_setup/DATASET_DOWNLOAD.md`](data_setup/DATASET_DOWNLOAD.md)

**Options de téléchargement**:
1. **Manuel**: Télécharger depuis Kaggle et extraire dans `data/`
2. **Automatique**: Utiliser Kaggle API: `python data_setup/download_data.py`

---

## 📁 Structure du Projet

```
Pneumonia-Detection/
├── data/                      # ⚠️ À télécharger séparément (voir ci-dessus)
│   ├── train/                 # 5,216 images d'entraînement
│   ├── val/                   # 16 images de validation
│   └── test/                  # 624 images de test
├── src/                       # Code source
│   ├── model.py               # Architecture des modèles (CNN, ResNet)
│   ├── dataset.py             # Chargement des données
│   ├── train.py               # Pipeline d'entraînement
│   ├── evaluate.py            # Évaluation et métriques
│   └── utils.py               # Utilitaires
├── notebooks/                 # Jupyter notebooks
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_model_building.ipynb
│   ├── 3_training.ipynb
│   └── 4_evaluation_metrics.ipynb
├── outputs/                   # Résultats d'entraînement
│   ├── models/                # Modèles sauvegardés (.pth)
│   └── logs/                  
├── data_setup/                # Scripts de téléchargement
│   ├── DATASET_DOWNLOAD.md    #  Instructions détaillées
│   └── download_data.py       # Script automatique
├── docs/                      # Documentation
├── train_model.py             # Script principal d'entraînement
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

---

##  Quick Start

### 1. Installation
```bash
# Cloner le repository
git clone https://github.com/androbiert/Pneumonia-Detection.git
cd Pneumonia-Detection

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Télécharger le Dataset
Suivez les instructions dans [`data_setup/DATASET_DOWNLOAD.md`](data_setup/DATASET_DOWNLOAD.md)

### 3. Entraîner le Modèle
```bash
python train_model.py
```

Le modèle sera sauvegardé dans `outputs/models/best_model.pth`

---

##  Modèles Disponibles

### 1. PneumoniaCNN (Custom)
- Architecture CNN à 4 couches
- ~776K paramètres
- Bon pour datasets de petite taille

### 2. PneumoniaResNet (Transfer Learning)
- Basé sur ResNet18 pré-entraîné
- ~11M paramètres (backbone gelé)
- Recommandé pour le dataset complet

**Changer de modèle** dans `train_model.py`:
```python
MODEL_TYPE = "CNN"      # ou "ResNet"
```

---




##  Configuration

Modifiez `train_model.py` pour ajuster:
- `BATCH_SIZE`: Taille du batch (défaut: 16)
- `EPOCHS`: Nombre d'époques (défaut: 20)
- `LEARNING_RATE`: Taux d'apprentissage (défaut: 0.001)
- `PATIENCE`: Early stopping (défaut: 5)

---

##  Améliorations Récentes

✅ Fix deprecation warning (ResNet18_Weights)  
✅ Gestion robuste des chemins (Path)  
✅ Normalisation des images  
✅ Data augmentation  
✅ Early stopping  
✅ Logs UTF-8  


---

## 📄 License

Ce projet utilise le dataset sous licence **CC BY 4.0**.

---

## 👨‍💻 Auteur

**OUADDANE MOHAMED**  
