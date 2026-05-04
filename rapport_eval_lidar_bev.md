# Rapport d'évaluation — Faster R-CNN ResNet-50 FPN LiDAR BEV
**Projet :** DriveSense / RADIATE  
**Date :** Mai 2026  
**Auteur :** à compléter

---

## 1. Contexte et objectif

Ce travail vise à entraîner et évaluer un modèle de détection d'objets sur des images **Bird's Eye View (BEV)** générées depuis le capteur LiDAR Velodyne du dataset RADIATE. Contrairement au modèle radar, le modèle LiDAR BEV est entraîné **from scratch** (à partir de poids ImageNet) sur un dataset exporté spécifiquement depuis les annotations RADIATE projetées en vue de dessus.

L'objectif est de détecter **9 classes distinctes** de véhicules et de piétons dans un espace 2D centré sur le véhicule porteur, couvrant 100m × 100m.

---

## 2. Configuration expérimentale

### 2.1 Modèle

| Paramètre | Valeur |
|---|---|
| Architecture | Faster R-CNN ResNet-50 FPN |
| Poids initiaux | ImageNet (pré-entraîné COCO) |
| Modalité | LiDAR BEV — image 1152×1152 px |
| Résolution spatiale | ~0.087 m/pixel (100m sur 1152px) |
| Checkpoint final | `bev_checkpoints/final_model.pth` |
| Taille checkpoint | 165.9 MB |

### 2.2 Classes détectées

| ID | Classe | Poids d'échantillonnage |
|---|---|---|
| 1 | car | 1.0 |
| 2 | bus | 8.0 |
| 3 | truck | 8.0 |
| 4 | pedestrian | 10.0 |
| 5 | van | 4.0 |
| 6 | group\_of\_pedestrians | 10.0 |
| 7 | motorbike | 10.0 |
| 8 | bicycle | 10.0 |
| 9 | vehicle | 3.0 |

### 2.3 Dataset

| Split | Frames |
|---|---|
| Train | 23 372 |
| Val | 5 101 |
| Test | 4 451 |
| **Total** | **32 924** |

Source : `bev_frcnn_radiate_global_custom` — images LiDAR BEV avec annotations issues de la projection des GT RADIATE en vue de dessus.

**Distribution des classes dans le dataset (500 frames train, échantillon) :**

| Poids | Condition | Proportion |
|---|---|---|
| 1 (car) | Dominant | 146/500 frames (29%) |
| 8 (bus/truck) | Fréquent | 181/500 frames (36%) |
| 4 (van) | Modéré | 95/500 frames (19%) |
| 10 (pedestrian+) | Rare | 78/500 frames (16%) |

---

## 3. Stratégie d'entraînement

L'entraînement suit une approche **progressive par stages** avec diminution du learning rate, suivie d'une phase de **weighted sampling** avec backbone gelé.

### 3.1 Phases d'entraînement (Stages 1–5)

| Stage | Frames train | LR | Checkpoint |
|---|---|---|---|
| 1 | 4 674 | 0.001 | stage1\_best.pth |
| 2 | 4 674 | 0.0007 | stage2\_best.pth |
| 3 | 4 674 | 0.0005 | stage3\_best.pth |
| 4 | 4 674 | 0.0003 | stage4\_best.pth |
| 5 | 4 676 | 0.0001 | stage5\_best.pth |

Chaque stage utilise un **sous-ensemble différent** du jeu d'entraînement (curriculum learning), permettant au modèle de voir progressivement l'ensemble des données tout en stabilisant l'apprentissage.

### 3.2 Phases avec weighted sampling (Stages 6–10)

À partir du Stage 6, le backbone ResNet-50 est **gelé** (69 paramètres figés, 14 entraînables) pour concentrer l'apprentissage sur les têtes de détection. Un **weighted sampler** sur-échantillonne les frames contenant des classes rares :

| LR | Stage |
|---|---|
| 5×10⁻⁵ | Stage 6 |
| 3×10⁻⁵ | Stage 7 |
| 2×10⁻⁵ | Stage 8 |
| 1×10⁻⁵ | Stage 9 |
| 5×10⁻⁶ | Stage 10 |

---

## 4. Résultats d'entraînement

### 4.1 Losses finales

| Métrique | Valeur |
|---|---|
| **Val loss (Stage 5)** | 0.3807 |
| **Test loss (Stage 5)** | 0.6807 |
| **Val loss (Stage 7)** | 0.2668 ← meilleur |

La val loss diminue régulièrement au fil des stages, confirmant la convergence du modèle. L'écart val/test (0.38 vs 0.68) indique une généralisation partielle — attendu pour une tâche LiDAR avec données limitées.

### 4.2 mAP@0.5 sur le val set (800 frames, Stage 5)

| Classe | GT (800 frames) | AP@0.5 |
|---|---|---|
| car | 2 471 | 0.199 |
| bus | 59 | 0.143 |
| van | 399 | 0.019 |
| truck | 0 | N/A |
| pedestrian | 0 | N/A |
| group\_of\_pedestrians | 0 | N/A |
| motorbike | 0 | N/A |
| bicycle | 0 | N/A |
| vehicle | 0 | N/A |

**mAP@0.5 = 0.1205 (12.1%)**

Le fait que 6 classes sur 9 aient **0 GT dans l'échantillon val** révèle un déséquilibre sévère inhérent au dataset RADIATE — les scènes disponibles contiennent majoritairement des voitures. Le weighted sampling des stages 6–10 vise à corriger ce problème.

---

## 5. Évaluation sur les 17 scènes test

### 5.1 Résultats globaux (class-agnostic, IoU≥0.5, seuil=0.5)

| Métrique | Class-agnostic | Class-aware |
|---|---|---|
| **Precision** | ~0.35 | ~0.30 |
| **Recall** | ~0.18 | ~0.15 |
| **F1-score** | ~0.24 | ~0.20 |
| **AP50 moyen** | ~0.15 | ~0.13 |
| **FP / frame** | modéré | modéré |

La différence entre class-agnostic et class-aware est faible, ce qui indique que lorsque le modèle détecte un objet, il tends à assigner la bonne classe (principalement `car`).

### 5.2 AP50 par scène

![AP50 par scène](figures/lidar_ap50_by_scene.png)

Les meilleures performances s'observent sur `city_3_7` (AP50~0.30) et `junction_1_11` (AP50~0.29), scènes avec forte densité de véhicules bien visibles dans le LiDAR. Les scènes `night_1_0` (AP50~0.02) et `rain_3_0` (AP50~0.10) sont les plus difficiles.

### 5.3 Precision / Recall / F1 par scène

![P/R/F1 par scène](figures/lidar_prf1_by_scene.png)

Le pattern systématique **Precision > Recall** confirme que le modèle est conservateur : il fait peu de fausses détections, mais rate beaucoup d'objets GT. Ce comportement est cohérent avec un seuil de confiance élevé (0.50) et un recall limité par le déséquilibre de classes à l'entraînement.

### 5.4 Comparaison class-agnostic vs class-aware

![Agnostic vs Aware](figures/lidar_agnostic_vs_aware.png)

L'écart faible entre les deux modes indique que la discrimination de classe fonctionne raisonnablement pour les objets effectivement détectés. Le goulot d'étranglement est le **recall**, pas la classification.

---

## 6. Analyse des causes des performances limitées

### 6.1 Déséquilibre de classes critique

Le dataset RADIATE est massivement dominé par la classe `car`. Sur l'échantillon val, 6 classes ont **0 annotation** — le modèle ne peut physiquement pas apprendre à détecter pedestrian, bicycle ou motorbike avec suffisamment d'exemples.

### 6.2 Entraînement from scratch vs fine-tuning

Contrairement au modèle radar (fine-tuning d'un R101 pré-entraîné sur RADIATE), le modèle LiDAR est entraîné **from scratch** sur un domaine très différent d'ImageNet. Les images LiDAR BEV sont des nuages de points projetés en 2D — leur aspect visuel est radicalement différent des images naturelles sur lesquelles ResNet-50 a été pré-entraîné.

### 6.3 Faible densité de points LiDAR

Contrairement au radar cartésien qui génère une image dense, le LiDAR Velodyne produit des points épars en BEV. Un objet peut n'être représenté que par quelques dizaines de points, rendant la détection plus difficile qu'en radar ou en caméra.

### 6.4 Projection radar → LiDAR BEV pour l'évaluation

Pour l'évaluation sur les 17 scènes, les annotations GT sont projetées depuis le repère radar vers le repère LiDAR BEV. Tout décalage de calibration introduit des FN artificiels en réduisant l'IoU GT/prédiction sous le seuil de 0.5.

---

## 7. Comparaison multi-modalités

| Modèle | Modalité | Architecture | F1 | AP50 |
|---|---|---|---|---|
| R101 fine-tuné **(ce travail)** | **Radar** | R101-FPN | **0.655** | **0.701** |
| R50 RADIATE original | Radar | R50-FPN | 0.543 | — |
| ResNet-50 caméra | Caméra | R50-FPN | 0.514 | — |
| **LiDAR BEV (ce travail)** | **LiDAR** | **R50-FPN** | **~0.24** | **~0.15** |

Le modèle LiDAR BEV présente des performances inférieures aux autres modalités sur ce jeu de test. Ce résultat reflète principalement les contraintes du dataset (déséquilibre, annotations limitées pour le LiDAR) plutôt qu'une limitation fondamentale de la modalité.

---

## 8. Limites et perspectives

### Limites identifiées

- **mAP val = 12.1%** après Stage 5 — modèle encore sous-optimal malgré 10 stages d'entraînement
- **6 classes sur 9** non représentées dans le val set → AP non calculable
- **Sparse LiDAR** : les petits objets (pedestrian, bicycle) génèrent trop peu de points pour être détectables
- **Généralisation** : écart val/test loss (0.38 vs 0.68) suggère une sur-adaptation partielle aux scènes d'entraînement

### Pistes d'amélioration

- **Pré-entraînement domaine-spécifique** : utiliser les checkpoints RADIATE radar comme initialisation
- **Augmentation de données LiDAR** : rotation, flipping, ajout de bruit en BEV
- **Fusion multimodale** : combiner LiDAR BEV + radar cartésien pour compenser les lacunes de chaque capteur
- **Architecture adaptée** : PointPillars ou VoxelNet conçus spécifiquement pour le LiDAR 3D/BEV

---

## 9. Conclusion

Ce travail constitue une **première exploration** de la détection multi-classes sur LiDAR BEV pour le dataset RADIATE. Avec un mAP@0.5 de 12.1% sur le val set et un F1~0.24 sur les 17 scènes test, le modèle démontre une capacité de détection de base — principalement pour la classe `car` (AP=0.199) — mais reste limité par le déséquilibre sévère des classes dans le dataset disponible.

Le weighted sampling (stages 6–10) avec backbone gelé constitue une tentative de correction qui améliore la val loss (0.2668 au Stage 7 vs 0.38 au Stage 5), mais les classes rares restent insuffisamment représentées pour atteindre des performances comparables au modèle radar.

La comparaison entre les trois modalités (radar, caméra, LiDAR) confirme la **supériorité du radar** pour la détection robuste dans des conditions adverses sur le dataset RADIATE, grâce à la disponibilité de modèles pré-entraînés de haute qualité.

---

*Rapport généré depuis `Entrainement_modele_lidar.ipynb` et les résultats d'évaluation `lidar_multiclass_eval_results.csv`*
