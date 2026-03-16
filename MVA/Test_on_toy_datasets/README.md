# Test on toy datasets

Ce dossier contient une experience autonome 2D pour reproduire le protocole jouet GMM de Flower sans utiliser la formule analytique de la velocite.

## Contenu

- `gmm_2d_common.py` : briques communes GMM / reseau / checkpoint.
- `train_gmm_2d_model.py` : entraine le reseau et sauvegarde un checkpoint reutilisable.
- `gmm_2d_flower.py` : script principal.
- `configs/gmm_2d_default.yaml` : configuration de l'experience.
- `run_train_gmm_2d.sh` : commande de training.
- `run_gmm_2d.sh` : commande de lancement.

## Ce que fait le script

Le script :

- definit un prior GMM en 2D ;
- charge un reseau fully-connected de flow matching deja entraine sur `p0 = N(0, I)` vers `p1 = pX` ;
- lance Flower avec `gamma = 0` et `gamma = 1` ;
- compare les particules finales a des echantillons du posterior analytique ;
- sauve les chemins de solution des etapes 1, 2, 3 a plusieurs iterations avec une presentation proche des figures du papier.

## Workflow

Il faut commencer par lancer le training, puis lancer les tests/experiences Flower.

1. Entrainer une fois le reseau et sauvegarder le checkpoint.
```bash
python MVA/Test_on_toy_datasets/train_gmm_2d_model.py --config MVA/Test_on_toy_datasets/configs/gmm_2d_default.yaml
```

2. Reutiliser ensuite ce checkpoint pour plusieurs experiences Flower.
```bash
python MVA/Test_on_toy_datasets/gmm_2d_flower.py --config MVA/Test_on_toy_datasets/configs/gmm_2d_default.yaml
```

## Scenarios inclus

- `scenario_1` : `h = [1.5, 1.5]`, `sigma_n = 0.25`, `y = 1`
- `scenario_2` : `h = [1.5, -1.5]`, `sigma_n = 0.75`, `y = 1`

## Sorties attendues

Quand tu executeras le script, il creera notamment :

- `training_curve.png`
- `training_summary.yaml`
- `checkpoints/gmm_2d_velocity.pt`
- `scenario_1_posterior_comparison.png`
- `scenario_1_flower_paths_gamma_0.png`
- `scenario_1_flower_paths_gamma_1.png`
- `scenario_1_single_trajectory_gamma_0.png`
- `scenario_1_single_trajectory_gamma_1.png`
- `scenario_2_posterior_comparison.png`
- `scenario_2_flower_paths_gamma_0.png`
- `scenario_2_flower_paths_gamma_1.png`
- `scenario_2_single_trajectory_gamma_0.png`
- `scenario_2_single_trajectory_gamma_1.png`
- `run_summary.yaml`

Les figures de comparaison et de chemins contiennent directement dans leur titre le nom du scenario ainsi que les parametres `h`, `y` et `sigma_n`.
