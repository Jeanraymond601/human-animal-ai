# Rapport — Détection IA (issue #2)

Modèle : `yolo11n.pt` (CPU, imgsz 640, IoU NMS 0,45). Classes ciblées : personne + 10 animaux COCO.

## 1. Fonctionnement du modèle
- Le modèle se charge et l'inférence tourne sur CPU : environ 0,04 s par image (hors premier appel, ~1 s).
- Tests : 9/9 passent (6 unitaires avec faux modèle + 3 d'intégration avec le vrai modèle, `tests/test_real_model.py`).

## 2. Détection personnes / animaux (modèle réel)

| Image | Vérité terrain | Détecté (conf. 0,35) | Verdict |
|---|---|---|---|
| test.jpg | femme + éléphant | Personne 1, Éléphant 1 | correct |
| image2.jpg | 2 personnes | Personne 2 (IoU ≥ 0,5) | correct |
| images.jpg | homme + chien | Personne 1, Chien 1 | correct visuellement |

Effet du seuil de confiance (nombre de détections correctes, sur images.jpg et image2.jpg) :
- 0,10 à 0,35 : toutes les personnes et animaux sont trouvés.
- 0,50 : le chien de images.jpg (confiance 44,8 %) est perdu.
- 0,70 : une personne de image2.jpg est perdue.

**Seuil recommandé : 0,25 à 0,35.** Au-delà de 0,45, on perd des vrais positifs.

## 3. Erreurs identifiées
1. **Annotation erronée `data/validation/labels/images.txt`** : la boîte du chien (x ≈ 65–240 px) ne correspond pas au chien réel (x ≈ 164–295 px). Les détections du modèle sont correctes à l'œil, mais l'évaluation les compte comme faux positif et faux négatif (le chien est jugé raté). Le mAP/recall du module d'évaluation était donc sous-estimé. **Corrigé** : les boîtes du chien et de la personne ont été recalées sur l'image (la boîte de la personne était aussi trop étroite).
2. **Collision pytest** : deux fichiers `test_detector.py` (racine et `tests/`) faisaient échouer `pytest`. Corrigé par `pytest.ini` (`testpaths = tests`). Le script de la racine est un script de démonstration, pas un test.
3. **Couverture** : les tests existants n'utilisaient que de faux modèles. Ajout de 3 tests avec le vrai modèle.
4. **Jeu de validation trop petit** (2 images) : les métriques ne sont pas représentatives. Il faut davantage d'images (cas difficiles : occlusion, nuit, petits objets, groupes).

## 4. Pistes pour améliorer la fiabilité
- Corriger les annotations puis élargir le jeu de validation avant de chiffrer mAP/précision/rappel.
- ~~Fixer le seuil par défaut autour de 0,30~~ : fait, `DEFAULT_CONFIDENCE = 0.30` dans `detect_objects` (était 0,20).
- Si la précision reste insuffisante, essayer `yolo11s`/`yolo11m` (plus précis, plus lents) et comparer sur le même jeu.
- Le chien est détecté avec une confiance faible (45 %) alors que l'image est nette : à surveiller sur des images plus nombreuses.
