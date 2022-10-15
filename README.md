# Hackathon42
 
Le Hackathon EffiSciences x Ecole42, en partenariat avec 42AI, sera axé sur l'intelligence artificielle bénéfique : le week-end du vendredi 14 au samedi 16 Octobre, nous proposerons des présentations, des ateliers et des formations sur la sécurité de l'IA.
 
![alt text](assets/hackathon.png "Hackathon IA Safety")
 
## Le Sujet en quelques mots : 
 
Certains jeux de données ne sont pas bien spécifiés : prenons l'exemple d'un jeu de données qui contiendrait des images de chameaux dans le désert, ainsi que des images de vaches dans des prairies. Le classificateur doit classer les images de chameaux et les images de vaches. Mais tel qu'il est formulé, le classificateur pourrait apprendre à classer les images non pas en fonction de l'animal, mais en fonction du paysage : le jeu de données est sous-spécifié car nous avons deux caractéristiques qui sont parfaitement corrélées (l'animal et le paysage). En d'autres termes, le classificateur peut décider de classer soit vache/chameau, soit prairie/désert. Et il y a ambiguïté lorsque nous essayons de classifier l'image d'un chameau dans une prairie. L'objectif de ce hackathon est de résoudre ce type d'ambiguïtés.
 
Le Hackathon consiste en une série de jeux de données (Toy dataset, MNIST, embeddings, ...).
 
Chaque ensemble de données contient :
- Un jeu de donné étiqueté : qui contient des images avec deux ou plus de deux caractéristiques parfaitement corrélées.
- Un jeu de données non étiqueté : qui contient un mélange d'images avec des caractéristiques parfaitement corrélées et des images avec des caractéristiques non corrélées.
- un jeu de données de validation : qui doit être étiqueté par les participant-es, et qui contient un mélange d'images avec des caractéristiques corrélées et des images avec des caractéristiques non corrélées.
 
Vous devez utiliser les jeux de données non étiquetés pour révéler l'ambiguïté.
 
Pour résoudre l'ambiguïté, les participant-es peuvent demander les étiquettes d'un maximum de 5 images des ensembles cibles, en choisissant judicieusement les images les plus pertinentes pour résoudre l'ambiguïté.
Les participant-es ont accès à une API, et ils peuvent interroger l'API pour obtenir les étiquettes des images, en posant la question image par image.
Il n'y a pas de pénalité pour avoir demandé 5 étiquettes au lieu d'une.
 
!! Attention !! Les participant-es ne peuvent faire qu'une seule soumission par ensemble de données !
 
Le leaderboard du Hackathon est accessible ici : https://leaderboard42.herokuapp.com/
 
## Motivations
 
### Pourquoi qu'une seule soumission ?
 
En effet, l'un des objectifs de ce hackathon est de faire prendre conscience aux participant-es la difficulté de mettre en production un système d'intelligence artificielle avancé. 
Une fois que le système est déployé, il est très difficile de revenir en arrière. De plus, nous aimerions que les futures intelligences artificielles ou modèles de langage avancés prennent le temps de poser des questions en cas de doute, prennent le temps de remarquer les ambiguïtés et n'agissent qu'après s'être parfaitement assurés de ce qu'on leur demande. La possibilité de demander 5 étiquettes simule cette situation de manière simple.
 
### Pourquoi trouve-t-on ce sujet intéressant ? 
 
Pourquoi un classificateur entraîné à identifier des poumons affaissés a-t-il fini par détecter des drains thoraciques (les cables) ?

![alt text](assets/lungs.jpg "Poumon avec un drain")
 
En effet, les données d'apprentissage ne permettaient pas de distinguer les véritables poumons affaissés des drains thoraciques - un traitement pour les poumons affaissés. Les drains thoraciques sont visuellement beaucoup plus simples que les poumons affaissés et les deux caractéristiques étaient corrélées, de sorte que l'algorithme a pu obtenir de bons résultats en apprenant à identifier la caractéristique la plus simple.
 
Les classificateurs apprennent généralement la caractéristique la plus simple qui permet de prédire l'étiquette, qu'elle corresponde ou non à ce que les humains avaient en tête. La surveillance humaine peut parfois détecter cette erreur, mais elle est lente, coûteuse et n'est pas totalement fiable (car l'homme peut ne pas se rendre compte de ce que fait l'algorithme avant qu'une erreur potentiellement dangereuse ne soit commise).
 
La détection de la "mauvaise" caractéristique signifie que le classificateur ne parviendra pas à généraliser comme prévu - lorsqu'il est déployé sur des radiographies de vrais humains avec de vrais poumons affaissés, non traités, il les classera comme sains, puisqu'ils n'ont pas de drain thoracique.
 
Ce défi est lié aux problèmes de sous-spécification (D'Amour et al., 2020) dans lesquels plusieurs hypothèses peuvent expliquer les données. Ainsi qu'au problème de robustesse aux changements de distribution (Amodei et al., 2016). Par exemple, des classificateurs entraînés à reconnaître les poumons de patients hospitalisés avec et sans pneumothorax ne peuvent pas être utilisés de manière préventive sur des patients non traités car le classificateur reconnaîtra le drain thoracique (une ligne droite facilement identifiable) et non les caractéristiques causales de la maladie (Oakden-Rayner et al., 2020). Ce problème est assez général et est susceptible de se poser dès qu'un algorithme ML doit être utilisé sur des données différentes des données d'entraînement (biais de sélection : les données étiquetées sont généralement plus simples que les données non étiquetées et plus simples que les données rencontrées en production). Par exemple, dans le domaine du développement durable, la plupart des modèles ML sont entraînés sur un échantillon de pays riches très différents des pays où le modèle sera déployé. En général, nous voulons utiliser les données du passé pour prédire l'avenir, mais l'avenir n'est pas le passé.
 
 
## Règles du jeu
 
Il y a deux types de prix : Score au leaderboard et prix du Jury. 

Vous devrez rassembler toutes vos explication pour chaque problème dans un google form qu'on vous fournira. Vous devrez soumettre votre code zippé et vos explication pour chaque probleme dans ce google form.
 
### Prix de Score au Leaderboard (premier prix 700 € + deuxième prix 400 €)
 
Le score total est la somme de la précision obtenue dans les target set de chaque jeu de données. Si les participant-es ne soumettent pas de données, ils ont un score par défaut de 80 % pour chaque jeu de données. Soumettre un ensemble de données, c'est donc prendre un risque. Il est préférable de ne rien soumettre que de soumettre quelque chose de mauvais.
 
Il est possible pour une même équipe de gagner à la fois le premier prix du classement et le premier prix du jury.
Nous examinerons le code des meilleures équipes du classement.
 
### Prix du jury (premier prix 600 € + deuxième prix 300 €)
 
Pour les prix du jury, les participant-es devront montrer leur code au jury et sont libres de demander au jury s'ils veulent présenter une bonne idée, même si leur idée ne réalise pas un bon score au classement, le jury prendra ces éléments en considération.
 
Critères d'évaluation :
- Entretiens avec les 20 meilleures équipes pour comprendre leurs approches.
- Les nouvelles méthodes seront fortement favorisées.
- Une belle méthode sera fortement privilégiée.
- Les approches créatives qui n'ont pas de bons résultats seront appréciées.
- Nous vérifierons le code :
   - L'entraînement d'un jeu de données ne doit pas utiliser les autres jeux de données.
   - Ne pas utiliser un modèle pré-entraîné.
   - Il est permis de regarder le target set, mais pas de classifier à la main.
 
Toute technique d'interprétabilité des réseaux neuronaux utilisée pour comprendre le calcul des réseaux neuronaux sera fortement valorisée.
Si votre solution est généralisable, et fonctionne sur plusieurs ensembles de données, elle sera valorisée par le prix du jury.
 
L'implication générale dans le hackathon et la volonté d'aider les autres participant-es sera prise en compte.
 
## Les deux phases du Hackathon
 
Du vendredi soir au dimanche à 14h, les participants travailleront sur des jeux de données fictifs.
Et le dimanche à 14h :
- les vrais jeux de données seront révélés et seront disponibles sur ce GitHub.
- Toutes les anciennes soumissions au classement seront supprimées. Les participants repartiront de zéro.
La permière phase sert donc à vous approprier le sujet et peaufinner vos stratégies et modèles, pour pouvoir les déployer en phase 2.
 
Nous procédons ainsi pour encourager l'écriture de code reproductible et généralisable, et décourager une solution de fortune adaptée à un jeu de données spécifique.
En substance, nous allons simplement régénérer les ensembles de données avec une autre graine aléatoire. Il n'y aura pas de nouveaux types de jeux de données. Les participants sont donc encouragés à écrire le code le plus automatique possible.
 
## Installation
 
Veuillez installer git large file system avant de cloner le git. La taille du repo est d'environ 1GB.
 
```
git lfs install
git clone https://github.com/EffiSciencesResearch/hackathon42.git
```
 
Notez qu'un GPU avec CUDA n'est pas indispensable pour ce tutoriel, un simple CPU fonctionnera très bien.
 
## Détails des jeux de données.
 
### INITIATION
 
#### 00_toy_dataset (1pts)
 
Cet ensemble de données est une régression linéaire simple. Ce jeu de données correspond à l'illustration la plus simple possible de notre problème. Nous avons deux caractéristiques (axe des x et axe des y) qui sont corrélées dans l'ensemble étiqueté. Les caractéristiques ne sont pas corrélées dans l'ensemble non étiqueté et dans le target set (ces deux éléments sont en gris sur la figure).
 
#### 01_mnist_cc (1pt)
 
Nous utilisons l'ensemble de données mnist pour simuler des ensembles de données mal spécifiés.
Nous avons maintenant deux caractéristiques : gauche et droite.
Tous les autres jeux de données sont une variation de ce jeu de données.
 
![alt text](assets/1_mnist_cc_labeled.png "1_mnist_cc_labeled.png")
![alt text](assets/1_mnist_cc_unlabeled.png "1_mnist_cc_unlabeled.png")
 
Dans ce jeu de données et dans les suivants, nous ajoutons un peu de bruit aux images.
 
#### 02_mnist_constant_image (1pt)
 
Dans cette tâche, nous introduisons le concept de biais de simplicité.  Le biais de simplicité (SB) est la tendance des procédures d'entrainement standardes telles que la descente de gradient stochastique (SGD) à trouver des modèles simples.
 
Selon [1], le SB de la SGD et de ses variantes peut être extrême : les réseaux neuronaux peuvent s'appuyer exclusivement sur la caractéristique la plus simple et rester invariants à toutes les caractéristiques prédictives complexes.
 
Dans cet exercice, il y a deux caractéristiques :
- L'image de gauche est une image provenant de MNIST.
- L'image de droite est une image constante (mais toujours un peu bruyante) en fonction de la classe, c'est-à-dire qu'elle est toujours le même 1 ou le même 0.
 
Il est beaucoup plus facile pour le classificateur d'utiliser l'image constante de droite que l'image de gauche. Cependant, dans le target_set, seule l'image de gauche sera prédictive.
 
02 signifie qu'il y a des zéros et des deux dans cet ensemble de données.
(L'id de cet ensemble de données est 2)
 
### RANDOM POSITION
 
#### 03_mnist_constant_image_random_row (2pts)
 
Même chose que 02_mnist_constant_image, mais on randomise les images de gauche et de droite.
 
#### 04_mnist_uniform_color_random_row (1pt)
 
Dans cette tâche, nous exacerbons le biais de simplicité en utilisant une image dont la couleur est constante en fonction de l'étiquette.
 
#### 05_mnist_uniform_color_low_mix_rate (2pts)
 
Certaines approches fonctionnent bien lorsque l'ensemble de données non étiquetées est équilibré entre toutes les catégories d'images - mais nous ne pouvons pas supposer que cela soit vrai pour les ensembles de données non étiquetées arbitraires dans la nature. Les ensembles de données non équilibrés peuvent, bien sûr, être rééquilibrés - cependant, cela s'apparente à un étiquetage manuel et, en tant que tel, est prohibitif et difficile à mettre à l'échelle.
 
Ainsi, dans cet exercice, nous cherchons une méthode qui fonctionne même avec un faible taux de mélange. Le taux de mélange est un nombre réel entre 0 et 1 qui indique la proportion de types de croisement (0/1 et 1/0) dans l'ensemble de données non étiquetées. Un taux de mélange de 0 ne comporte que 0/0 et 1/1 (comme dans l'ensemble de données étiquetées), un taux de mélange de 0,5 comporte des quantités égales dans chaque catégorie, tandis qu'un taux de mélange de 1 ne comporte que 0/1 et 1/0 types de croisement.
 
### SUM
 
#### 06_mnist_sum (1pts)

Identique à 01_mnist_cc mais on additionne les images de gauche et de droite.
 
#### 07_mnist_sum_bis (1pts)
 
Identique à 06_mnist_sum mais nous additionnons 3 images.
 
#### 08_mnist_sum_noise_level (1pts)
 
Nous utilisons le niveau du bruit gaussien comme biais de simplicité.
 
 
### Mysterious datasets (3pts each)
 
En plus des autres jeux de données, nous ajoutons 2 jeux de données (12 et 13) qui peuvent être traités indépendamment du reste du hackathon.
 
Vous ne pourrez soumettre et collecter les 5 étiquettes sur le dataset 12 que pendant la deuxième phase du hackathon.
 
### Embedding datasets (3pts each)
 
Les jeux de données (23 et 456) contiennent les embeddings des chiffres mnists. Vous ne serez pas en mesure d'inspecter ces jeux de données ^^.
(Les identifiants de ces jeux de données sont 23 et 456)

### Vehicle-Animal (5pts)

L'animal se trouve soit à gauche, soit à droite du véhicule.
L'"oiseau" ou le "chat" à côté d'un "avion" ou d'une "voiture".
(id=888)

Cet exercice et le suivant sont plus durs que les autres. Le score par défaut est donc de 0.5 et non de 0.8.
### Human datasets (6pts)
 
Ce jeu de données (id=999) contient des images d'humains. Vous devrez partir d'un réseau neuronal pré-entraîné pour améliorer vos chances.
Ce jeu de données n'est pas dans le même format que les deux autres pour des raisons de taille mémoire. Vous devrez dézipper le jeu de données pour commencer à travailler.
Ce jeu de données nécessite l'utilisation d'un gpu (contrairement aux autres jeux de données), par exemple via google colab. Vous pouvez sauvegarder votre gpu colab pour l'utiliser sur ce jeu de données.

Un dataset qui ressemble beaucoup beacuoup à human dataset sera rajouté à 15h dimanche.
 
### Comment obtenir les 5 étiquettes
 
Comment accéder à l'API :
 
POST https://leaderboard42.herokuapp.com/reveal/
 
with the following form data:
- `username`: `awesome_team`
- `password`: `secret_password`
- `exercise_id`: 3
- `datum_id`: 456
 
Example request with cURL:
 
```bash
$ curl -F username=awesome_team -F password=secret_password -F exercise_id=3 -F datum_id=456 https://leaderboard42.herokuapp.com/reveal/
```
 
Example request with [Requests](https://requests.readthedocs.io/en/latest/) in Python:
 
```python
import requests
import json
 
res = requests.post("https://leaderboard42.herokuapp.com/reveal/", data={
       'username': 'my_awesome_team',
       'password': "my_password",
       'exercise_id': 0,
       'datum_id': 4
   })
 
try:
   res = json.loads(res.content)
   print(res)
except:
   print("Error")
   print(res.content)
 
# {'exercise_id': 0, 'datum_id': 4, 'label': 0, 'previously revealed': [12]}
```
Attention, le score final entre les gagnants sera probablement très serré ! Chaque étiquette est précieuse !
 
 
### Comment soumettre votre solution

Pour pouvoir participer au classement du hackathon, une équipe doit être validée par les organisateurs.
 
Allez sur https://leaderboard42.herokuapp.com/
 
Et cliquez sur un exercice, et soumettez votre solution.
 
Un exemple de format de soumission est [here](exemple_submision.csv). La soumission est un csv sans en-tête et sans colonne d'index. Il s'agit simplement de la liste des étiquettes de l'ensemble de validation. Vous devez soumettre un .csv et non un .txt.
 
### Résolution de problèmes
 
Si vous voyez "ValueError : Cannot load file containing pickled data when allow_pickle=False", c'est probablement parce que vous n'avez pas installé git LFS.
 
## Remerciements
 
- Nous remercions l'Ecole42 et le Club AI de l'Ecole42 pour leur collaboration à l'organisation du hackathon.
- Manuel Bimich pour l'idée du hackathon et les lourdes tâches administratives.
- Quentin Didier et Charbel Cegerie pour la préparation et l'organisation du Hackathon.
- Quentin Feuillat, Mathieu David et le club AI de 42 pour leur formation !
- Esaïe Bauer et Joseph Barbier pour l'animation !
- Symphony, pour leur incroyable cuisine !
- Laszlo pour le développement du leaderboard.
- Lola Elisalde pour l'énorme gestion de la logistique.
- Timothée Chauvin, Elias Schmidt et Gautier Ducurtil pour le beta-testing du hackathon.
- Merci à Alexandre, JS et toutes les autres personnes qui nous ont aidé à développer le sujet.
- Diego et son frère, pour avoir fourni un sujet de secours.
- L'administration de l'Ecole42 qui a aidé à la logistique.
 
[1] Shah, Harshay, et al. "The pitfalls of simplicity bias in neural networks." Advances in Neural Information Processing Systems 33 (2020): 9573-9585.
 




===== ENGLISH VERSION ======
# hackathon42
 
The EffiSciences x Ecole42 hackathon will focus on Beneficial Artificial Intelligence : The weekend of Friday, October 14-16 will host presentations, workshops and training in AI Safety.
 
![alt text](assets/hackathon.png "Hackathon IA Safety")
 
## The topic in a few words
 
Some datasets are not well specified: let's take an example of a dataset that would contain images of camels in the desert, as well as images of cows in grasslands. The classifier must classify between images of camels and images of cows. But formulated as it is, the classifier could learn to classify the images not according to the animal, but according to the landscape: the dataset is underspecified because we have two features that are perfectly correlated (animal and landscape). In other words, the classifier can either decide to classify cow/camel or grassland/desert. And there is ambiguity when we try to classify an image of a camel in a grassland. The goal of this hackathon is to resolve these types of ambiguities.
 
The hackathon consists of a series of datasets (Toy dataset, MNIST, embeddings, ...).
 
Each dataset contains:
- A labeled set: which contains images with two or more perfectly correlated features.
- An unlabeled set: which contains a mixture of images with perfectly correlated features and images with uncorrelated features
- a validation set: which must be labeled by the participants, and which contains a mixture of images with correlated features and images with uncorrelated assets.
 
You must use the unlabeled sets to notice the ambiguity.
 
To resolve the ambiguity, participants can ask for the labels of up to 5 images from the target sets, choosing the most relevant images to resolve the ambiguity.
The participants have access to an API, and they can query the API to get the labels of the images, by asking the question image by image.
There is no penalty for requesting 5 labels instead of 1.
 
Warning. Participants can only make one submission per dataset!
 
The hackathon leaderboard can be found here: https://leaderboard42.herokuapp.com/
 
## Motivations
 
### Why only one submission?
 
Indeed, one of the goals of this hackathon is to make students aware of the difficulty of putting an advanced artificial intelligence system into production. Once the system is in production, it is very difficult to go back. In addition, we would like future artificial intelligences or advanced language models to take the time to ask questions when in doubt, take the time to notice ambiguity, and only act after being sure of the request. The opportunity to ask for 5 labels simulates this situation in a simple way.
 
### Why do we think this topic is interesting?
 
Why did a classifier that was trained to identify collapsed lungs end up detecting chest drains instead?
 
![alt text](assets/lungs.jpg "lungs with a drain")
 
Because the training data was insufficient to distinguish actual collapsed lungs from chest drains – a treatment for collapsed lungs. Chest drains are visually far simpler than collapsed lungs and the two features were correlated, so the algorithm was able to perform well by learning to identify the simplest feature.
 
Classifiers will generally learn the simplest feature that predicts the label, whether it is what we humans had in mind, or not. Human oversight can sometimes catch this error, but human oversight is slow, expensive, and not fully reliable (as the humans may not even realize what the algorithm is actually doing before a potentially dangerous mistake is made).
 
Detecting the ‘wrong’ feature means that the classifier will fail to generalize as expected – when deployed on X-rays of real humans with real, untreated, collapsed lungs, it will classify them as healthy, since they don’t have a chest drain.
 
This challenge is related to underspecification problems (D'Amour et al., 2020) in which several hypotheses can explain the data. As well as the problem of robustness to distributional changes (Amodei et al., 2016). For example, classifiers trained to recognize the lungs of hospitalized patients with and without pneumothorax cannot be used preemptively on untreated patients because the classifier will recognize the chest drain (an easily identifiable straight line) and not the causative features of the disease (Oakden-Rayner et al., 2020). This problem is quite general and is likely to arise as soon as an ML algorithm is to be used on data different from the training data (selection bias: labeled data are generally simpler than unlabeled data and simpler than the data encountered in production). For example, in the area of sustainable development, most ML models are trained on a sample of rich countries very different from the countries where the model will be deployed. In general, we want to use past data to predict the future, but the future is not the past.
 
 
## Rules of the game
 
There are two types of prizes: leaderboard maximization awards, and jury awards.
 
 
### Leaderboard Maximization prize (first prize €700 + second prize €400)
 
The total score is the sum of the accuracy achieved in the target sets of each data set. If participants do not submit data, they have a default score of 80% for each dataset. So submitting a dataset is taking a risk. It is better to submit nothing than to submit something bad.
 
It is possible for the same team to win both the first leaderboard prize and the first jury prize.
We will review the code of the top teams of the leaderboards.
 
### Jury prize (first prize €600 + second prize €300)
 
For the jury prizes, participants will have to show their code to the jury and are free to ask the jury if they want to pitch a good idea. Even if they don't have a good score on the leaderboard, the jury will take these elements into consideration.
 
Evaluation criteria:
- Interviews with the top 20 teams to understand their approaches
- New methods will be strongly favored.
- A beautiful ML method will be strongly preferred.
- Creative approaches that do not have good results will be valued
- We will check the code:
   - The training of a dataset must not use the other datasets
   - Do not use a pretrained model
   - It is allowed to look at the target set, but not allowed to classify by hand.
 
Any neural network interpretability techniques used to understand neural network computation will be highly valued.
If your solution is generic, and works across datasets, it will be valued by the jury prize.

Involvement and ideas proposed during the conferences and workshops will also be a criterion.
 
## The two phases of the hackathon
 
From Friday evening to Sunday at 2pm, participants will work on mock datasets. On Sunday at 2pm:
- the real datasets will be revealed and will be available on this GitHub.
- all old submissions to the leaderboard will be deleted. Participants will start from scratch again.
 
We are doing this procedure to encourage writing replicable code.
In essence, we're just going to regenerate the datasets with another random seed. There will be no new dataset types. Participants are thus encouraged to write the most automatic code possible.
 
## Installation
 
Please install git large file system before cloning the git. The size of the repo is approximately 1GB.
 
```
git lfs install
git clone https://github.com/EffiSciencesResearch/hackathon42.git
```
 
Note, a GPU with CUDA is not critical for this tutorial, as a CPU will not take much time.
 
## Dataset details
 
### INITIATION
 
#### 00_toy_dataset (1pts)
 
This dataset is a simple linear regression. This dataset corresponds to the simplest possible illustration of our problem. We have two features which are correlated in the labeled set. The features are not correlated in the unlabeled set and in the target set.

#### 01_mnist_cc (1pt)
 
We use the mnist dataset to simulate misspecified datasets.
We now have two features: left and right.
All other datasets are a variation of this dataset.
 
![alt text](assets/1_mnist_cc_labeled.png "1_mnist_cc_labeled.png")
![alt text](assets/1_mnist_cc_unlabeled.png "1_mnist_cc_unlabeled.png")
 
In this dataset and in the following ones, we add a bit of noise to the images.
 
#### 02_mnist_constant_image (1pt)
 
In this task, we introduce the concept of Simplicity bias.  Simplicity Bias (SB) -- the tendency of standard training procedures such as stochastic gradient descent (SGD) to find simple models.
 
According to [1], the SB of SGD and its variants can be extreme: neural networks can rely exclusively on the simplest feature and remain invariant to all complex predictive features.
 
In this exercise, there are two features:
- The left image is an image from MNIST
- The image on the right is a constant image (but still a bit noisy) depending on the class, i.e. it is always the same 1 or the same 0.
 
It is much easier for the classifier to use the constant image on the right than the image on the left. However, in the target_set, only the left image will be predictive.
 
02 means that there are zeros and twos in this dataset.
(The id of this dataset is 2)
 
### RANDOM POSITION
 
#### 03_mnist_constant_image_random_row (2pts)
 
Same thing as 02_mnist_constant_image, but we randomize the left and right images.
 
#### 04_mnist_uniform_color_random_row (1pt)
 
In this task, we exacerbate the simplicity bias by using an image that is a constant color depending on the label.
 
#### 05_mnist_uniform_color_low_mix_rate (2pts)
 
Some approaches work well when the unlabeled dataset is balanced across all image categories - but we cannot assume this to be true for arbitrary unlabeled datasets in nature. Unbalanced datasets can, of course, be rebalanced - however, this is akin to manual labeling and, as such, is prohibitively expensive and difficult to scale.
 
Thus, in this exercise, we seek a method that works even with a low mixing rate. The mixing ratio is a real number between 0 and 1 that indicates the proportion of cross types (0/1 and 1/0) in the unlabeled data set. A mixture rate of 0 has only 0/0 and 1/1 (as in the labeled data set), a mixture rate of 0.5 has equal amounts in each category, while a mixture rate of 1 has only 0/1 and 1/0 cross types.
 
### SUM
 
#### 06_mnist_sum (1pts)
 
Same as 01_mnist_cc but we sum the left and right images.
 
#### 07_mnist_sum_bis (1pts)
 
Same as 06_mnist_sum but we sum 3 images.
 
#### 08_mnist_sum_noise_level (1pts)
 
We use the level of the gaussian noise as the simplicity bias.
 
 
### Mysterious datasets (3pts each)
 
In addition to the other datasets, we add 2 datasets (12 and 13) that can be processed independently with the rest of the hackathon.
 
You will only be able to submit and collect the 5 labels on dataset 12 only during the second phase of the hackathon.
 
### Embedding datasets (3pts each)
 
Datasets (23 and 456) contain the embeddings of mnist digits. You won't be able to inspect those datasets ^^.
(The id of those datasets are 23 and 456)

### Vehicle-Animal (5pts)

The animal is either to the left or right of the vehicle.
The "bird" or "cat" is next to a "plane" or "car".
(id=888)

This exercise and the next one are harder than the others. The default score is therefore 0.5 and not 0.8.
 
### Human datasets (6pts)
 
This dataset (id=999) contains images of humans. You will have to start from a pre-trained neural network to improve your chances.
This dataset is not in the same format as the two others for memory size reasons. You will need to unzip the dataset to start working.
This dataset requires the use of a gpu (unlike the other datasets), for example google colab. You can save your colab gpu to use it on this dataset.
 
### How to get the 5 labels
 
Revelation:
 
POST https://leaderboard42.herokuapp.com/reveal/
 
with the following form data:
- `username`: `awesome_team`
- `password`: `secret_password`
- `exercise_id`: 3
- `datum_id`: 456
 
Example request with cURL:
 
```bash
$ curl -F username=awesome_team -F password=secret_password -F exercise_id=3 -F datum_id=456 https://leaderboard42.herokuapp.com/reveal/
```
 
Example request with [Requests](https://requests.readthedocs.io/en/latest/) in Python:
 
```python
import requests
import json
 
res = requests.post("https://leaderboard42.herokuapp.com/reveal/", data={
       'username': 'my_awesome_team',
       'password': "my_password",
       'exercise_id': 0,
       'datum_id': 4
   })
 
try:
   res = json.loads(res.content)
   print(res)
except:
   print("Error")
   print(res.content)
 
# {'exercise_id': 0, 'datum_id': 4, 'label': 0, 'previously revealed': [12]}
```
Beware, the final score between the winners will probably be very close! Every label is precious!
 
 
### How to submit your solution
 
In order to participate in the hackathon's leaderboard, a team must be validated by the organizers.
 
Go to https://leaderboard42.herokuapp.com/
 
And click on one exercice, and submit your solution.
 
An example of submission format is [here](example_submision.csv). The submission is a csv with no header and no index column. It's just the list of labels of the validation set. You must submit a .csv and not a .txt.
 
### Troubleshooting
 
If you see “ValueError: Cannot load file containing pickled data when allow_pickle=False”, it's probably because you didn't install git LFS.
 
## Acknowledgements
 
- We thank Ecole42 and the AI Club of Ecole42 for their collaboration in the organization of the hackathon.
- Manuel Bimich for the hackathon idea and the heavy administrative lifting.
- Quentin Didier for the preparation of the Hackathon.
- Quentin Feuillat, Mathieu David and the AI club of 42 for their formation!
- Esaïe Bauer and Joseph Barbier for the animation!
- Symphonie, for their incredible cooking!
- Laszlo for the development of the leaderboard.
- Lola Elisalde for the huge logistics management
- Timothée Chauvin, Elias Schmidt and Gautier Ducurtil for beta-testing the hackathon.
- Thanks to Alexandre, JS and all the other people who helped us to develop the subject.
- Diego and his brother, for providing a backup subject.
- The administration of Ecole42 who helped with the logistics.
 
 
[1] Shah, Harshay, et al. "The pitfalls of simplicity bias in neural networks." Advances in Neural Information Processing Systems 33 (2020): 9573-9585.
 

