# market_implied_taylor_rule

Ce dossier contient une réplication, adaptée à la zone euro, de la logique de
Hamilton, Pruitt & Borger (2011) pour estimer une règle de Taylor « perçue par
le marché » à partir de mouvements de taux OIS autour de publications
macroéconomiques.

L’idée générale du projet est la suivante :

1. Construire une base d’événements de marché à partir des annonces HICP, PMI
   et, selon les scripts, d’autres séries macro.
2. Relier chaque événement à un mois de référence `tau`, puis raccrocher les
   cibles futures d’inflation et d’activité.
3. Estimer, pour plusieurs maturités OIS, la sensibilité des taux aux surprises
   de données et déduire les paramètres d’une règle de Taylor implicite.
4. Proposer plusieurs variantes de la même logique d’estimation :
   GMM, OLS simple, orthogonalisation des surprises PMI, et versions
   simplifiées centrées uniquement sur l’inflation.

## Organisation des fichiers

### `GMM_estimation.py`

Fichier central du projet. Il contient :

- la construction des tables d’événements ;
- l’inférence du mois de référence `tau` ;
- le raccordement aux séries mensuelles d’inflation et d’output ;
- la construction des panels mensuels par maturité ;
- l’estimation GMM en deux étapes ;
- les fonctions de diagnostic et de restitution des résultats.

Ce script est la version la plus complète et la plus proche de la réplication
conceptuelle de HPB. Il sert de socle aux autres fichiers.

### `OLS_HPB.py`

Version OLS simplifiée du pipeline, construite sur la même préparation des
données que le script GMM.

Spécificités :

- estimation en trois étapes ;
- régressions séparées pour l’équation d’inflation et l’équation d’output ;
- régression principale finale avec régresseurs générés ;
- bootstrap par blocs mensuels pour corriger les écarts-types ;
- adaptation « daily » avec maturités mensuelles de 1M à 12M.

Cette version est utile pour comparer la logique GMM avec une approche OLS plus
simple et plus lisible.

### `ortho_OLS.py`

Variante OLS avec orthogonalisation de la surprise PMI par rapport à la
surprise HICP.

Spécificités :

- étape préalable d’orthogonalisation `PMI ⊥ HICP` ;
- meilleure séparation empirique des deux facteurs de surprise ;
- diagnostic explicite de l’identification via `det(Γ)` et le nombre de
  conditionnement ;
- bootstrap de l’estimation principale ;
- résumé par maturité pour comparer la qualité d’identification.

Ce script sert surtout à traiter les problèmes de colinéarité entre HICP et
PMI.

### `Inflation_only.py`

Version réduite centrée uniquement sur l’HICP.

Spécificités :

- pas de PMI ;
- pas de variable d’output ;
- estimation en deux étapes seulement ;
- calcul direct de l’effet OIS sur l’inflation observée ;
- conversion vers `beta` par méthode delta ;
- bootstrap pour obtenir des écarts-types corrigés.

Cette variante fournit une lecture plus parcimonieuse du mécanisme de
réaction des taux, au prix d’un modèle moins riche.

### `inflation_no_output.py`

Autre version simplifiée, également centrée sur l’HICP, mais structurée de
façon encore plus directe.

Spécificités :

- régression de la réaction OIS sur la surprise HICP ;
- estimation séparée de l’empreinte inflation `gamma_pi` ;
- calcul de `beta = psi / gamma_pi` ;
- utilisation d’une méthode delta pour les écarts-types ;
- bootstrap mensuel pour fiabiliser l’inférence.

Ce fichier est surtout utile comme baseline minimale pour comparer les
résultats des versions plus complètes.

### `panel_data.ipynb`

Notebook de travail pour explorer les données, tester les hypothèses
d’alignement temporel et inspecter les panels avant estimation.

## Logique commune

Quel que soit le script, la logique générale reste la même :

- transformer les annonces en événements datés ;
- construire des surprises `actual - consensus` ;
- associer ces surprises aux cibles futures et aux contrôles ;
- estimer des coefficients d’empreinte pour l’inflation et l’activité ;
- remonter ensuite à la réaction implicite des taux OIS.

## Remarque

Plusieurs fichiers implémentent des variantes proches du même cadre, avec des
hypothèses différentes sur la disponibilité des variables, la structure du
panel, ou la façon de traiter l’identification. Le bon point d’entrée dépend
donc de l’objectif :

- `GMM_estimation.py` pour la version la plus complète ;
- `OLS_HPB.py` pour une version OLS plus simple ;
- `ortho_OLS.py` pour tester l’effet de l’orthogonalisation PMI/HICP ;
- `Inflation_only.py` ou `inflation_no_output.py` pour des versions
  simplifiées.
