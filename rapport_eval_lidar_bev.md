# Rapport d'évaluation — Faster R-CNN ResNet-50 FPN LiDAR BEV

**Projet :** DriveSense / RADIATE  
**Date :** Mai 2026  
**Auteur :** à compléter

---

## 1. Contexte et objectif

Ce travail vise à entraîner et évaluer un modèle de détection d'objets sur des images **Bird's Eye View (BEV)** générées depuis le capteur LiDAR Velodyne du dataset RADIATE.

Contrairement au modèle radar officiel, qui regroupe plusieurs classes en une seule classe générique `vehicle`, le modèle LiDAR BEV est ici entraîné pour détecter directement les classes séparées issues des annotations RADIATE.

Le modèle est entraîné sur un dataset exporté spécifiquement à partir des annotations RADIATE projetées en vue de dessus.

L'objectif est de détecter **8 classes distinctes** de véhicules et de piétons dans un espace 2D centré sur le véhicule porteur.

---

## 2. Configuration expérimentale

### 2.1 Modèle

| Paramètre | Valeur |
|---|---|
| Architecture | Faster R-CNN ResNet-50 FPN |
| Poids initiaux | Pré-entraînement ImageNet / COCO |
| Modalité | LiDAR BEV — image 1152×1152 px |
| Représentation | Vue de dessus générée à partir du LiDAR |
| Checkpoint final | `bev_checkpoints/final_model.pth` |
| Taille checkpoint | 165.9 MB |

---

### 2.2 Classes détectées

Le modèle est entraîné sur **8 classes RADIATE**.

| ID | Classe | Poids d'échantillonnage |
|---|---|---:|
| 1 | car | 1.0 |
| 2 | bus | 8.0 |
| 3 | truck | 8.0 |
| 4 | pedestrian | 10.0 |
| 5 | van | 4.0 |
| 6 | group_of_pedestrians | 10.0 |
| 7 | motorbike | 10.0 |
| 8 | bicycle | 10.0 |

> Important : la classe `vehicle` ne fait pas partie du modèle LiDAR.  
> Elle correspond plutôt à une fusion de classes utilisée dans certains modèles radar baseline.

---

### 2.3 Dataset

| Split | Frames |
|---|---:|
| Train | 23 372 |
| Val | 5 101 |
| Test | 4 451 |
| **Total** | **32 924** |

Source : `bev_frcnn_radiate_global_custom`

Le dataset contient des images LiDAR BEV associées à des annotations issues de la projection des ground truth RADIATE en vue de dessus.

---

### 2.4 Distribution des classes

Sur un échantillon de 500 frames du train set, la distribution reste fortement déséquilibrée.

| Groupe | Condition | Proportion |
|---|---|---:|
| car | Classe dominante | 146 / 500 frames |
| bus / truck | Classes fréquentes mais minoritaires | 181 / 500 frames |
| van | Classe modérément représentée | 95 / 500 frames |
| pedestrian / group / motorbike / bicycle | Classes rares | 78 / 500 frames |

Ce déséquilibre explique en partie les difficultés du modèle à apprendre correctement toutes les classes.

---

## 3. Stratégie d'entraînement

L'entraînement suit une approche progressive par stages, avec diminution du learning rate, puis une phase de weighted sampling pour mieux prendre en compte les classes rares.

---

### 3.1 Phases d'entraînement — Stages 1 à 5

| Stage | Frames train | Learning rate | Checkpoint |
|---|---:|---:|---|
| Stage 1 | 4 674 | 0.001 | `stage1_best.pth` |
| Stage 2 | 4 674 | 0.0007 | `stage2_best.pth` |
| Stage 3 | 4 674 | 0.0005 | `stage3_best.pth` |
| Stage 4 | 4 674 | 0.0003 | `stage4_best.pth` |
| Stage 5 | 4 676 | 0.0001 | `stage5_best.pth` |

Chaque stage utilise un sous-ensemble différent du jeu d'entraînement. Cette stratégie permet au modèle de voir progressivement l'ensemble des données tout en stabilisant l'apprentissage.

---

### 3.2 Phases avec weighted sampling — Stages 6 à 10

À partir du Stage 6, le backbone ResNet-50 est gelé afin de concentrer l'apprentissage sur les têtes de détection.

Un weighted sampler est utilisé pour sur-échantillonner les frames contenant des classes rares.

| Stage | Learning rate |
|---|---:|
| Stage 6 | 5×10⁻⁵ |
| Stage 7 | 3×10⁻⁵ |
| Stage 8 | 2×10⁻⁵ |
| Stage 9 | 1×10⁻⁵ |
| Stage 10 | 5×10⁻⁶ |

---

## 4. Résultats d'entraînement

### 4.1 Losses finales

| Métrique | Valeur |
|---|---:|
| Val loss — Stage 5 | 0.3807 |
| Test loss — Stage 5 | 0.6807 |
| Meilleure val loss — Stage 7 | 0.2668 |

La validation loss diminue au fil des stages, ce qui montre que l'entraînement converge progressivement.

L'écart entre la validation loss et la test loss indique cependant une généralisation encore limitée, probablement liée à la difficulté du LiDAR BEV et au déséquilibre des classes.

---

### 4.2 mAP@0.5 sur le val set — 800 frames

| Classe | GT sur 800 frames | AP@0.5 |
|---|---:|---:|
| car | 2 471 | 0.199 |
| bus | 59 | 0.143 |
| van | 399 | 0.019 |
| truck | 0 | N/A |
| pedestrian | 0 | N/A |
| group_of_pedestrians | 0 | N/A |
| motorbike | 0 | N/A |
| bicycle | 0 | N/A |

**mAP@0.5 = 0.1205**

Le fait que plusieurs classes aient 0 annotation dans cet échantillon de validation montre un déséquilibre important du dataset. Le modèle apprend surtout les classes les plus représentées, principalement `car`.

---

## 5. Évaluation sur les scènes test

### 5.1 Résultats globaux  
Class-agnostic et class-aware — IoU ≥ 0.5 — seuil = 0.5

| Métrique | Class-agnostic | Class-aware |
|---|---:|---:|
| Precision | ~0.35 | ~0.30 |
| Recall | ~0.18 | ~0.15 |
| F1-score | ~0.24 | ~0.20 |
| AP50 moyen | ~0.15 | ~0.13 |
| FP / frame | modéré | modéré |

L'écart entre class-agnostic et class-aware reste limité. Cela indique que, lorsque le modèle détecte correctement un objet, il attribue souvent une classe cohérente, principalement pour la classe `car`.

Le principal problème reste le **recall faible** : le modèle rate encore beaucoup d'objets.

---

### 5.2 AP50 par scène

![AP50 par scène](figures/lidar_ap50_by_scene.png)

Les meilleures performances apparaissent sur certaines scènes urbaines et de jonction, où les véhicules sont plus nombreux et mieux représentés dans le LiDAR BEV.

Les scènes difficiles, notamment de nuit ou avec pluie, restent plus complexes pour le modèle.

---

### 5.3 Precision / Recall / F1 par scène

![P/R/F1 par scène](figures/lidar_prf1_by_scene.png)

Le modèle présente généralement une précision supérieure au rappel.

Cela signifie qu'il produit relativement peu de fausses détections, mais qu'il manque encore beaucoup d'objets présents dans les ground truth.

Ce comportement est cohérent avec :

- un seuil de confiance assez strict ;
- des objets parfois peu visibles dans le LiDAR BEV ;
- un déséquilibre important des classes pendant l'entraînement.

---

### 5.4 Comparaison class-agnostic vs class-aware

![Agnostic vs Aware](figures/lidar_agnostic_vs_aware.png)

La comparaison entre les deux modes d'évaluation permet de distinguer deux aspects :

- **class-agnostic** : le modèle est récompensé si la boîte est bien placée, peu importe la classe ;
- **class-aware** : la boîte doit être bien placée et la classe doit être correcte.

L'écart limité entre les deux suggère que la classification n'est pas le principal point faible. Le problème principal est plutôt la capacité à détecter tous les objets.

---

## 6. Analyse des performances

### 6.1 Déséquilibre important des classes

Le dataset est dominé par la classe `car`.

Les classes comme `pedestrian`, `bicycle`, `motorbike` ou `group_of_pedestrians` sont beaucoup moins représentées, ce qui limite fortement leur apprentissage.

---

### 6.2 Difficulté du LiDAR BEV

Les images LiDAR BEV sont beaucoup plus éparses que les images radar ou caméra.

Un objet peut être représenté par très peu de points, surtout s'il est petit ou éloigné. Cela rend la détection plus difficile.

---

### 6.3 Projection des annotations

Pour évaluer le modèle LiDAR, les annotations source RADIATE sont projetées depuis le repère radar vers le repère LiDAR BEV.

Cette projection est nécessaire, mais elle peut introduire de petits décalages. Si ces décalages réduisent l'IoU sous 0.5, certains objets peuvent être comptés comme faux négatifs.

---

### 6.4 Généralisation limitée

L'écart entre la validation loss et la test loss montre que le modèle n'est pas encore parfaitement généralisable à toutes les scènes.

Cela peut être dû à :

- la diversité des conditions météo ;
- la rareté de certaines classes ;
- la faible densité LiDAR ;
- la différence entre les scènes d'entraînement et de test.

---

## 7. Comparaison avec les autres modalités

| Modèle | Modalité | Architecture | F1 | AP50 |
|---|---|---|---:|---:|
| R101 radar fine-tuné | Radar | Faster R-CNN R101-FPN | ~0.655 | ~0.701 |
| R50 radar RADIATE original | Radar | Faster R-CNN R50-FPN | ~0.543 | — |
| ResNet-50 caméra | Caméra | Faster R-CNN R50-FPN | ~0.514 | — |
| LiDAR BEV | LiDAR | Faster R-CNN R50-FPN | ~0.24 | ~0.15 |

Le modèle LiDAR BEV obtient des performances inférieures aux modèles radar et caméra.

Cela ne signifie pas que le LiDAR est inutile. Cela montre surtout que cette première approche Faster R-CNN sur image BEV LiDAR reste limitée par :

- la faible densité des points ;
- le déséquilibre des classes ;
- la qualité des projections ;
- le manque de pré-entraînement adapté au LiDAR.

---

## 8. Limites et perspectives

### Limites identifiées

- mAP@0.5 encore faible ;
- recall limité ;
- classes rares insuffisamment représentées ;
- dépendance à la projection radar → LiDAR pour les ground truth ;
- difficulté à détecter les petits objets ;
- généralisation encore limitée.

---

### Pistes d'amélioration

- augmenter la quantité de données pour les classes rares ;
- améliorer le weighted sampling ;
- tester des augmentations spécifiques LiDAR BEV ;
- comparer avec des architectures plus adaptées au LiDAR, comme PointPillars ou VoxelNet ;
- exploiter le LiDAR en complément du radar dans la fusion multi-capteurs ;
- améliorer les projections entre radar, LiDAR et caméra.

---

## 9. Conclusion

Ce travail constitue une première exploration de la détection multi-classes sur LiDAR BEV pour le dataset RADIATE.

Le modèle est entraîné sur **8 classes distinctes** :

```text
car, bus, truck, pedestrian, van, group_of_pedestrians, motorbike, bicycle
