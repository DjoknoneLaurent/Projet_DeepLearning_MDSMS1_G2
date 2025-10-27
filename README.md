Projet_DeepLearning_MDSMS1_G2
Projet de Deep Learning (Reconnaissance Faciale - MDSMS12025 G2)



Master Data Science & Modélisation Statistique (MDSMS1 – Groupe 2)
Encadrant : [MELATAGIA YONTA Paulin]
Auteurs :DJOKNONE Laurent, EKWANE Franck, MATSING MEVOUNGOU Orlane, OTABELA ANGAYENE Stéphane


Résumé du projet

Ce projet vise à concevoir, entraîner et évaluer un modèle de Deep Learning capable de reconnaître automatiquement des visages humains à partir d’images.
Le modèle repose sur une architecture CNN (Convolutional Neural Network) implémentée sous TensorFlow / Keras, avec un pipeline complet allant de la préparation des données à la prédiction finale et à la visualisation des performances.



Objectifs du projet

1. Construire un jeu de données de visages, structuré par individu.
2. Appliquer un prétraitement complet (redimensionnement, normalisation, conversion RGB).
3. Mettre en œuvre une augmentation de données (Data Augmentation) pour enrichir l’apprentissage.
4. Concevoir un modèle CNN régularisé et robuste.
5. Évaluer les performances à l’aide de métriques visuelles et quantitatives.
6. Tester le modèle sur des images individuelles et analyser les erreurs via une matrice de confusion.



Technologies et bibliothèques utilisées

Python 3.10+
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Scikit-learn
TQDM



Pipeline général du projet:

1-Connexion au Drive et exploration des données

Les données sont stockées dans Google Drive, sous :

python
base_dir = "/content/drive/MyDrive/Projet_DeepLearning"


Chaque sous-dossier du répertoire dataset correspond à une personne (classe).

2-Prétraitement des images

Conversion BGR → RGB
Redimensionnement à 128×128 pixels
Normalisation des valeurs de pixels entre 0 et 1
Sauvegarde dans un dossier processed_data/

Outil principal : OpenCV (cv2)



3-Chargement et préparation des données

-Chargement des images depuis processed_data/
-Construction des matrices X (images) et y (étiquettes)
-Conversion des étiquettes en one-hot encoding
-Séparation 80% / 20% en jeu d’entraînement et de test

Outil principal : Scikit-learn (train_test_split)


 4-Augmentation des données

Pour renforcer la robustesse du modèle, les images sont augmentées par :

rotations aléatoires (±20°)
translations horizontales/verticales
 zooms aléatoires
retournements horizontaux

Cette étape utilise ImageDataGenerator de Keras pour générer dynamiquement de nouvelles images à chaque époque d’entraînement.


5-Construction du modèle CNN

Le réseau de neurones convolutif que nous avons proposé comprend:

  La première partie du réseau correspond à la phase d’extraction des caractéristiques. Elle débute avec une première couche de convolution 2D comportant 32 filtres de taille 3×3, associée à une régularisation L2 fixée à 0.002. Cette couche est suivie d’une opération de pooling (MaxPooling2D) avec une fenêtre 2×2, qui permet de réduire la dimension spatiale tout en conservant les informations essentielles. Un Dropout de 0.3 est ensuite appliqué pour limiter le risque de surapprentissage en désactivant aléatoirement certains neurones pendant l’entraînement.

   La deuxième partie approfondit l’apprentissage des caractéristiques avec une deuxième couche de convolution 2D comprenant 64 filtres, toujours accompagnée d’une régularisation L2 de 0.003. Elle est suivie d’un MaxPooling2D (2×2) et d’un Dropout de 0.4, renforçant la capacité du modèle à généraliser sur de nouvelles images.

   La troisième partie constitue la couche de convolution la plus profonde du réseau. Elle contient 128 filtres de taille 3×3, également régularisés avec L2=0.003, suivis d’un pooling 2×2 et d’un Dropout de 0.5. Cette dernière étape d’extraction permet au modèle de capturer des motifs plus complexes et spécifiques aux visages individuels.
  
  Après l’extraction des caractéristiques, le réseau passe à la phase de classification. Les cartes de caractéristiques issues des convolutions sont aplaties (Flatten) puis envoyées dans une couche Dense (fully connected) de 128 neurones. Cette couche est également régularisée par un Dropout de 0.4 pour garantir une meilleure robustesse du modèle. Enfin, la couche de sortie est une couche Dense comportant un nombre de neurones égal au nombre total de classes (num_classes), avec une activation Softmax qui permet de produire des probabilités de prédiction pour chaque individu.



6-Entraînement du modèle

-50 époques
-Batch size = 32
-Données augmentées à chaque itération
-Enregistrement automatique de l’historique (loss, accuracy, etc.)

Les performances sont visualisées via les courbes :


Précision (accuracy) : entraînement vs validation
Perte (loss) : entraînement vs validation


7-Sauvegarde du modèle

Le modèle entraîné est sauvegardé dans :


/content/drive/MyDrive/Projet_DeepLearning/modele_visage.h5, Cela permet une réutilisation future sans réentraînement.



8-Évaluation du modèle

-Évaluation quantitative :

python: test_eval = model.evaluate(X_test, y_test, verbose=0)


Sortie :

=====================================
      ÉVALUATION DU MODÈLE CNN
=====================================
 Test Loss (fonction de perte)     : 2.2245
 Test Accuracy (précision globale) : 67.57%
=====================================
 Modèle encore faible : pense à ajouter plus de données ou à ajuster les hyperparamètres.


Interprétation automatique :

Le modèle final de reconnaissance faciale présente une perte (Test Loss) de 2.2245 et une précision globale (Test Accuracy) de 67.57%. Ces résultats montrent que le modèle a bien appris à reconnaître une partie des visages, mais qu’il reste encore des marges d’amélioration pour mieux généraliser. En d’autres termes, le réseau parvient à identifier correctement environ deux visages sur trois, ce qui est déjà une bonne performance pour une première version construite sur un petit jeu de données.

Tout au long du projet, plusieurs efforts importants ont été fournis pour surmonter les difficultés rencontrées. Nous avons commencé par préparer et nettoyer les données en redimensionnant, convertissant et normalisant toutes les images de manière systématique. Chaque étape du prétraitement a été soigneusement testée et vérifiée visuellement pour s’assurer de la qualité des entrées. Ensuite, un réseau de neurones convolutif (CNN) a été construit à partir de zéro, avec plusieurs couches de convolution, de pooling, de régularisation et de dropout. Des ajustements progressifs ont été réalisés sur la structure du modèle et sur les hyperparamètres afin de stabiliser l’entraînement et d’éviter le surapprentissage.

Malgré tous ces efforts, certains défis subsistent. Le nombre d’images par personne (environ 20) reste faible et limite la diversité des données (expressions, angles, luminosité…). De plus, les paramètres de régularisation relativement forts (Dropout à 0.5 et L2=0.003). e nombre d’époques (50), bien qu’adéquat pour un plusieurs essais, pourrait être augmenté pour permettre une convergence plus complète du réseau.

Cependant, les résultats obtenus sont encourageants : le modèle a démontré une capacité d’apprentissage réelle et une distinction claire entre plusieurs individus. Cela prouve que l’architecture que nous avons mise en place fonctionne correctement.

Pour améliorer la performance, nous estimons pertinent d’ajouter davantage d’images par classe, d’ajuster les hyperparamètres (Dropout entre 0.3 et 0.4, L2 autour de 0.001), d’utiliser des couches de normalisation (BatchNormalization), ou encore de recourir à des modèles préentraînés comme VGG16, ResNet50 ou MobileNetV2 afin de profiter du transfert d’apprentissage. Ces optimisations nous permettraient d’atteindre une précision supérieure à 90%.

En résumé, notre modèle atteint une performance moyenne mais prometteuse (≈67%), tout à fait acceptable pour un prototype académique. Ce projet constitue ainsi une base solide pour des versions futures plus puissantes et plus précises que nous envisageons.



9-Prédiction sur une image individuelle

Chargement d’une image au hasard :

python:img = cv2.imread("processed_data/DJOKNONE/laurent_1.jpg")

Le modèle prédit la classe correspondante et affiche :


Prédiction : DJOKNONE avec un affichage visuel via matplotlib.



10-Matrice de confusion

Une matrice de confusion est générée pour évaluer la performance du modèle par classe individuelle.
L’affichage se fait avec ConfusionMatrixDisplay de Scikit-learn.

PISTE D'AMELIORATION:

1. Ajouter davantage d’images par individu (≥ 50 par classe).
2. Explorer des architectures plus profondes (ResNet, VGG16, MobileNet).
3. Optimiser les hyperparamètres (batch size, learning rate, régularisation).
4. Implémenter un système de **détection automatique de visage** avant classification.

