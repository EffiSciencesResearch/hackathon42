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
 
### Human datasets (5pts)
 
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
 

