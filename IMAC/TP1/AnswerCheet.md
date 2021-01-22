# Feuille de réponses TP1

## Exercice 2 

### Question 3

Le programme fonctionnait toujours pour moi, mais je suppose que c'est parce que j'avais vérifié que l'index ne dépassait pas la taille du tableau. On doit choisir un nombre global de threads à exécuter qui est en accord avec les tailles de vecteurs.

### Question 5 

Il ne faut plus coder sur un seul thread mais sur plusieurs (Ici, j'ai 256 threads par blocs). On a aussi besoin de blocs. Un groupe de threads est appelé un bloc. Les blocs sont regroupés dans une grille. Un noyau est exécuté comme une grille de blocs de threads. Pour calculer le nombre de blocs à avoir dans la grille, on divise la taille du vecteur par le nombre de thread par blocs.

Pour rappel, le maximum de thread par bloc est de 1024.



| Taille vecteur | Thread | Bloc | Image                                                        |
| -------------- | ------ | ---- | ------------------------------------------------------------ |
| 256            | 256    | 1    | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/256-256T-1B.png) |
| 512            | 256    | 1    | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/512-256T-1B.png) |
| 512            | 256    | 2    | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/512-256T-2B.png) |
| 4096           | 256    | 16   | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/4096-256T-8B.png) |
| 4096           | 512    | 8    | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/4096-256T-16B.png) |



## Exercice 3

### Question 3

| Nom image | Thread    | Bloc    | Image                                                        |
| --------- | --------- | ------- | ------------------------------------------------------------ |
| Chuck     | (38,50)   | (16,16) | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/Chuck_T16.png) |
| Lena      | (32,32)   | (16,16) | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/Lena_T16.png) |
| Rose      | (308,204) | (16,16) | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/Rose_T16.png) |
| Chuck     | (19,25)   | (32,32) | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/Chuck_T32.png) |
| Lena      | (16,16)   | (32,32) | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/Lena_T32.png) |
| Rose      | (154,102) | (32,32) | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/Rose_T32.png) |



### Question 4 

Si l'on inverse les boucles (largeur/hauteur) du kernel, on obtient une différence dans le chargement de données. C'est ce qu'on appelle une transaction de mémoire coalescée, qui est une transaction dans laquelle tous les threads d'un demi-flux accèdent à la mémoire globale en même temps. La bonne façon de procéder est de faire en sorte que des threads consécutifs accèdent à des adresses mémoire consécutives.

## Exercice 4

### Question 2



| Taille matrice | Thread    | Bloc  | Image                                                        |
| -------------- | --------- | ----- | ------------------------------------------------------------ |
| 32             | (32,32)   | (1,1) | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/M32-32x2T-1B.png) |
| 128            | (128,128) | (1,1) | ![](/home/laurine/Documents/MASTER/learn-cuda/IMAC/TP1/images/M128-128x2T-1B.png) |
|                |           |       |                                                              |



