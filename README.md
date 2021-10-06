# Projet de NLP


## Authors
- [Alexis Morton](https://github.com/Ekusukaliba)
- [Clara David](https://github.com/ClaraMMD)
- [Théo Mitrail](https://github.com/TheoMtrl)
- [Victor Dutto](https://github.com/VictorDutto)

## Project setup instructions
To start using this project use the following commands:

- `git clone https://github.com/VictorDutto/nlp_symbolic`

## Install the requirements

The project has been tested with Python 3.8 and 3.9

Thus, it is recommended to run this project with python 3.8 or more
```bash
pip install -r requirements.txt
```

### Project description

This project aims to classify some [IMDB cinema review](https://keras.io/api/datasets/imdb), and to predict whether the written reviews are rather positive or not.

To do so, 3 approaches are implemented, tested with some diverse techniques to verify if the results can be improved.

The first one is the bayesian naive classification

The second one is a logistic regression

The third and last one is a machine learning solution based on FastText and embedding words
