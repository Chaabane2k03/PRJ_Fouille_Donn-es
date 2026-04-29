# Commentaires et Interprétations du Code — Détection d'Anomalies Réseau

Ce document accompagne le notebook `prj-python.ipynb` et fournit des commentaires détaillés sur chaque étape du pipeline, ainsi que les interprétations des résultats obtenus.

---

## 1. Téléchargement et Copie du Dataset

### Code : Téléchargement via KaggleHub
```python
import kagglehub
path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
```
**Commentaire :** Le dataset **CIC-IDS2017** est récupéré depuis Kaggle. Il contient 8 fichiers CSV correspondant à différents jours et types de trafic réseau (lundi normal, mardi avec attaques FTP/SSH, etc.).

### Code : Copie vers un répertoire de travail
```python
shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
```
**Commentaire :** Les données sont copiées vers `/kaggle/working/dataset` (répertoire en écriture). Cela garantit que les fichiers sources restent intacts et que le pipeline peut écrire librement.

**Résultat :** 8 fichiers CSV sont confirmés présents.

---

## 2. Chargement et Fusion des Données

```python
all_files = glob.glob(os.path.join(path, "*.csv"))
li = []
for filename in all_files:
    df_temp = pd.read_csv(filename, index_col=None, header=0)
    li.append(df_temp)
df = pd.concat(li, axis=0, ignore_index=True)
df.columns = df.columns.str.strip()
```

**Commentaire :**
- `glob` récupère tous les fichiers `.csv` du répertoire.
- Chaque fichier est lu avec `pd.read_csv` et ajouté à une liste.
- `pd.concat` fusionne les 8 DataFrames en un seul, avec réindexation (`ignore_index=True`).
- `str.strip()` nettoie les espaces dans les noms de colonnes (problème courant dans ce dataset).

**Résultat :** `(2 830 743, 79)` — soit ≈2,83 millions de flux réseau × 79 colonnes (78 features + 1 label).

**Interprétation :** Ce volume est conséquent et justifie des choix prudents en termes de batch size et d'optimisation Optuna.

---

## 3. Exploration des Données

### `df.head()`, `df.shape`, `df.info()`, `df.describe()`

**Commentaire :** Ces cellules exploratoires permettent de :
- Vérifier la structure tabulaire (types, taille)
- Identifier les colonnes numériques (78) et la colonne cible (`Label`, type `object`)
- Repérer les valeurs extrêmes, les échelles hétérogènes, et même des valeurs négatives atypiques (ex: `Flow Duration` min = -13)
- Confirmer l'utilisation mémoire (~1.7 GB)

**Interprétation :** Les échelles très hétérogènes et la présence de valeurs aberrantes justifient pleinement le nettoyage (remplacement `inf`/`NaN`) et la normalisation `MinMaxScaler`.

### Distribution des classes

```python
label_counts = df['Label'].value_counts()
```

**Résultat :**
| Classe | Nombre |
|--------|--------|
| BENIGN | 2 273 097 |
| DoS Hulk | 231 073 |
| PortScan | 158 930 |
| DDoS | 128 027 |
| ... (11 autres types) | ... |

**Interprétation :** Forte majorité de trafic `BENIGN` (~80%). Le déséquilibre renforce l'intérêt d'une approche par **Autoencoder non supervisé** : le modèle apprend uniquement le comportement normal et détecte les anomalies par écart de reconstruction.

---

## 4. Prétraitement des Données

```python
features = [col for col in df.columns if col != 'Label']
train_df = df[df['Label'] == 'BENIGN'].copy()
train_features = train_df[features].copy()
test_features = df[features].copy()

# Nettoyage des valeurs inf et NaN
train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
train_features.fillna(train_features.median(numeric_only=True), inplace=True)
test_features.replace([np.inf, -np.inf], np.nan, inplace=True)
test_features.fillna(test_features.median(numeric_only=True), inplace=True)

# Normalisation
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)
```

**Commentaires détaillés :**
1. **Séparation Train/Test** : Le train utilise **uniquement les données BENIGN** (2 273 097 lignes). C'est le cœur de l'approche : l'Autoencoder apprend exclusivement à reconstruire le trafic normal.
2. **Nettoyage** : `np.inf` et `-np.inf` → `NaN` → remplacement par la **médiane** (plus robuste que la moyenne face aux outliers).
3. **MinMaxScaler** : Ramène toutes les features dans `[0, 1]`. Indispensable pour la convergence du réseau de neurones — sans normalisation, les features à grande échelle domineraient l'apprentissage.

**Résultat :** `X_train: (2 273 097, 78)`, `X_test: (2 830 743, 78)`

---

## 5. Configuration GPU et Imports ML

```python
gpu_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
```

**Commentaire :** La détection GPU (Tesla T4) et l'activation du *memory growth* évitent l'allocation préemptive de toute la mémoire GPU. Les warnings CUDA sont normaux en environnement notebook.

---

## 6. Fonction Objective pour Optuna

```python
def objective(trial):
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    activation = trial.suggest_categorical('activation', ['relu'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    encoder_units = []
    for i in range(2):  # 2 couches encodeur
        units = trial.suggest_categorical(
            f'encoder_units_layer_{i+1}',
            [64, 96, 128] if i == 0 else [16, 32, 48, 64]
        )
        encoder_units.append(units)

    latent_dim = trial.suggest_int('latent_dim', 4, 16)
    # ... construction du modèle et entraînement
```

**Commentaires détaillés :**
- **`dropout_rate`** (0.0–0.3) : Régularisation — désactive aléatoirement un % de neurones pour éviter le surapprentissage.
- **`activation`** : Fixé à `relu` (la seule option explorée — par contrainte de temps).
- **`learning_rate`** (1e-4 à 1e-2, échelle log) : Vitesse de convergence de l'optimiseur Adam.
- **`encoder_units`** : 2 couches avec des choix discrets. La première couche est plus large (64–128) car elle doit capturer les motifs bruts des 78 features. La seconde est plus petite (16–64) pour commencer la compression.
- **`latent_dim`** (4–16) : Dimension de l'espace latent, cœur du bottleneck.

### Pourquoi seulement 2 couches dans l'encodeur ?

> **Contrainte de temps et hardware.** Sur Kaggle (GPU Tesla T4 avec limite de session), chaque trial Optuna prend ~2 minutes. Avec 20 trials et 30 epochs max, l'optimisation dure déjà ~34 minutes. Ajouter des couches multiplierait le temps par un facteur significatif, rendant l'optimisation impraticable dans les délais.

---

## 7. Exécution de l'Optimisation Optuna

```python
X_search, _ = train_test_split(X_train_scaled, train_size=0.2, random_state=42)
study = optuna.create_study(direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
study.optimize(objective, n_trials=20)
```

**Commentaires :**
- **20% des données** (`train_size=0.2`) pour accélérer la recherche (≈450 000 échantillons au lieu de 2,3M).
- **MedianPruner** : Arrête prématurément les trials dont la performance intermédiaire est inférieure à la médiane des trials précédents. Économise du temps.
- **20 trials** : Compromis entre exploration et temps disponible.

**Résultat (Best Trial #3) :**
| Paramètre | Valeur |
|-----------|--------|
| `dropout_rate` | 0.005 |
| `activation` | relu |
| `learning_rate` | 0.001506 |
| `encoder_units_layer_1` | 128 |
| `encoder_units_layer_2` | 32 |
| `latent_dim` | 15 |
| **val_loss** | **4.124e-05** |

**Interprétation :**
- Le **dropout très faible** (0.5%) indique que le modèle ne souffre pas de surapprentissage majeur sur ces données.
- Le **learning rate** ~0.0015 est dans une plage typique pour Adam.
- La structure **128 → 32 → 15** crée une compression progressive : 78 features → 128 (expansion) → 32 → 15 (bottleneck). L'expansion initiale permet au réseau d'apprendre des combinaisons non linéaires avant de compresser.

### Erreur `KeyError: 'encoder_units_layer_3'`
Le code tente d'accéder à des clés `encoder_units_layer_3` et `encoder_units_layer_4` dans `study.best_trial.params`, mais l'architecture n'a que 2 couches. **C'est un résidu de code** d'une version antérieure qui prévoyait plus de couches. L'erreur est sans conséquence car les paramètres sont correctement extraits juste avant.

---

## 8. Construction du Modèle Final

```python
# Encodeur : 78 → 128 → Dropout → 32 → Dropout → 15 (latent)
# Décodeur : 15 → 32 → Dropout → 128 → Dropout → 78 (sortie)
autoencoder_model = Model(input_data, reconstructed_data)
autoencoder_model.compile(optimizer=Adam(learning_rate=0.001506), loss='mse')
```

**Architecture confirmée par `model.summary()` :**

| Couche | Output Shape | Params |
|--------|-------------|--------|
| encoder_input (Input) | (None, 78) | 0 |
| encoder_1 (Dense, relu) | (None, 128) | 10 112 |
| dropout (Dropout 0.5%) | (None, 128) | 0 |
| encoder_2 (Dense, relu) | (None, 32) | 4 128 |
| dropout (Dropout 0.5%) | (None, 32) | 0 |
| latent_encoding (Dense, linear) | (None, 15) | 495 |
| decoder_1 (Dense, relu) | (None, 32) | 512 |
| dropout (Dropout 0.5%) | (None, 32) | 0 |
| decoder_2 (Dense, relu) | (None, 128) | 4 224 |
| dropout (Dropout 0.5%) | (None, 128) | 0 |
| reconstructed_data (Dense, linear) | (None, 78) | 10 062 |
| **Total** | | **29 533** |

**Commentaire :** ~29K paramètres — modèle léger, adapté aux données tabulaires. L'activation **linéaire** pour l'espace latent et la sortie est standard : l'espace latent doit pouvoir prendre n'importe quelle valeur réelle, et la sortie doit reconstruire des valeurs continues normalisées.

---

## 9. Entraînement Final

```python
history = autoencoder_model.fit(
    X_train_scaled, X_train_scaled,  # input = target (autoencoder)
    epochs=30, batch_size=1024,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5,
               mode='min', restore_best_weights=True)]
)
```

**Commentaires :**
- **`X_train_scaled` comme input ET target** : c'est le principe de l'Autoencoder — le modèle apprend à reconstruire ses propres entrées.
- **`batch_size=1024`** : Batch élevé pour profiter du GPU et stabiliser le gradient sur ce grand volume de données.
- **`EarlyStopping(patience=5)`** : Arrêt si la `val_loss` ne s'améliore pas pendant 5 epochs. `restore_best_weights=True` restaure les poids du meilleur epoch.

**Résultat :** L'entraînement converge en 27 epochs (arrêt anticipé). La `val_loss` finale est ~2.3e-05, en baisse constante sans divergence → **pas de surapprentissage**.

---

## 10. Détection des Anomalies

```python
# Reconstruction et calcul MSE
reconstructions = autoencoder_model.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)

# Seuil = 90e percentile des erreurs sur les données normales
train_mse = np.mean(np.power(X_train_scaled - train_reconstructions, 2), axis=1)
threshold = np.percentile(train_mse, 90)

# Classification
y_pred = [1 if e > threshold else 0 for e in mse]
y_true = [1 if label != 'BENIGN' else 0 for label in df['Label']]
```

**Commentaires :**
- **Erreur de reconstruction (MSE)** : Pour chaque échantillon, on calcule la différence au carré entre l'entrée et la sortie du modèle.
- **Seuil au 90e percentile** : 10% des données normales sont "mal reconstruites" → seuil `2.21e-05`. Ce choix est un compromis : un seuil plus bas détecterait plus d'anomalies mais génèrerait plus de faux positifs.
- **Logique de détection** : Si l'erreur de reconstruction > seuil → anomalie.

**Résultat :**
```
              precision    recall  f1-score   support
      Normal       0.89      0.90      0.89   2 273 097
     Anomaly       0.57      0.54      0.55     557 646
    accuracy                           0.83   2 830 743
```

**Interprétation :**
- **Normal** : Bonne détection (precision 89%, recall 90%) — le modèle reconstruuit bien le trafic normal.
- **Anomaly** : Détection plus difficile (precision 57%, recall 54%, F1 55%) — classique pour les Autoencoders sur ce type de dataset.
- **525 941 anomalies détectées** sur 2 830 743 échantillons.
- La matrice de confusion montre que le modèle est conservateur : il privilégie la détection correcte du trafic normal au détriment de certaines attaques subtiles.

### Pourquoi la détection des anomalies est-elle modeste ?
1. **Architecture simplifiée** (2 couches) : manque de profondeur pour capturer les motifs complexes.
2. **Seuil fixe (percentile)** : un seuil adaptatif ou un clustering (ex: KMeans) sur l'espace latent pourrait améliorer les résultats.
3. **Pas de sélection de features** : certaines des 78 features sont redondantes ou peu informatives.
4. **Certaines attaques sont très similaires au trafic normal** (ex: Infiltration, 36 échantillons seulement).
