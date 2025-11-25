#  Dataset Download Instructions

## ⚠️ IMPORTANT

Les données de ce projet sont **volumineuses** (~2.5 GB) et ne sont **PAS incluses dans GitHub**.

Vous devez télécharger le dataset depuis Kaggle.

---

##  Source du Dataset

**Chest X-Ray Images (Pneumonia)**
- **URL Kaggle**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Taille**: ~2.5 GB
- **License**: CC BY 4.0
- **Images**: 5,856 radiographies thoraciques

---

##  Option 1: Téléchargement Manuel (Recommandé)

### Étape 1: Télécharger depuis Kaggle
1. Allez sur: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Connectez-vous à votre compte Kaggle (créez-en un si nécessaire)
3. Cliquez sur **"Download"** (bouton en haut à droite)
4. Le fichier `archive.zip` (~2.5 GB) sera téléchargé

### Étape 2: Extraire les données
1. Extraire le contenu de `archive.zip`
2. Vous obtiendrez un dossier `chest_xray/` contenant:
   - `train/`
   - `val/`
   - `test/`

### Étape 3: Organiser dans le projet
Copiez les dossiers dans la structure suivante:

```
Pneumonia-Detection/
└── data/
    ├── train/
    │   ├── NORMAL/    
    │   └── PNEUMONIA/   
    ├── val/
    │   ├── NORMAL/      
    │   └── PNEUMONIA/   
    └── test/
        ├── NORMAL/     
        └── PNEUMONIA/   
```

### Étape 4: Vérifier l'installation
```bash
python -c "from src.dataset import ChestXRayDataset; train = ChestXRayDataset('train'); print(f'✓ {len(train)} images chargées')"
```

---

##  Option 2: Téléchargement Automatique via Kaggle API

### Prérequis
1. Compte Kaggle
2. API Token Kaggle

### Configuration de l'API

#### 1. Obtenir votre API Token
- Allez sur: https://www.kaggle.com/settings/account
- Section "API" → Cliquez sur **"Create New Token"**
- Un fichier `kaggle.json` sera téléchargé

#### 2. Installer Kaggle CLI
```bash
pip install kaggle
```

#### 3. Configurer le Token

**Windows**:
```powershell
mkdir %USERPROFILE%\.kaggle
move Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

**Linux/Mac**:
```bash
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 4. Télécharger automatiquement
Depuis la racine du projet:
```bash
python data_setup/download_data.py
```

OU manuellement:
```bash
cd data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d .
mv chest_xray/* .
rm -rf chest_xray chest-xray-pneumonia.zip
```

---


### Description
- **Type**: Radiographies thoraciques (Chest X-Ray)
- **Format**: JPEG (niveaux de gris)
- **Classes**: 
  - `NORMAL`: Poumons sains
  - `PNEUMONIA`: Détection de pneumonie
- **Traitement**: Images redimensionnées à 224×224 pixels par le code

---

##  Note importante pour Git

Le dossier `data/` est dans `.gitignore` pour éviter de commit les fichiers volumineux.

**NE PAS essayer de push les données sur GitHub** - GitHub a une limite de 100MB par fichier.

Les utilisateurs doivent télécharger les données séparément en suivant ces instructions.

---

## ❓ Problèmes Courants

### "Data directory not found"
➡️ **Solution**: Vérifiez que le dossier `data/` existe et contient `train/`, `val/`, `test/`

### "Class folder not found: NORMAL"
➡️ **Solution**: Chaque split doit contenir les sous-dossiers `NORMAL/` et `PNEUMONIA/`

### Erreur Kaggle API "401 Unauthorized"
➡️ **Solution**: Vérifiez que `kaggle.json` est bien placé dans `~/.kaggle/` avec les bonnes permissions

### Dataset trop volumineux
➡️ **Solution**: Utilisez l'option 2 (Kaggle API) ou téléchargez sur un disque avec au moins 3GB d'espace libre

---

## 📌 Référence

**Citation du Dataset**:
```
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018). 
Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. 
Mendeley Data, v2. http://dx.doi.org/10.17632/rscbjbr9sj.2
```

**Kaggle**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
