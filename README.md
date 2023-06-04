# Projet Long : Protein Units Recognition from sequence using Transformers

NB : Du fait d'un taille importante de donnée pour la réalisation du projet, un jeu de donnée test a été créé pour GitHub

Le code en Jupyter Notebook correspond à l'ensemble du script utilisé alors que le code en python correspond à l'utilisation partielle du Jupyter Notebook afin de lancer l'apprentissage sur la plateforme iPOP-UP.

Pour lancer le code python :
```
python pu.py \
   --test categorical \ #ou binary
   --learning_rate 0.00001 \
   --batch_size 4 \
   --path_data dataset_trim \
   --path_save save \
   --transf binary_adjusted_batch4_lr-05.h5 \ #uniquement pour la classification catégorique
   --epoch 20
```
Le code python génèrera les plots de la précision et de la _loss_ dans le dossier `save`
