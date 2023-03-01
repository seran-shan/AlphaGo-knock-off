# AlphaGo Knock-off

This is a knock-off of the AlphaGo algorithm, which is a deep learning algorithm that can play the game of Go. The algorithm is described in the paper [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://www.nature.com/articles/nature16961) by Silver et al. The algorithm is implemented in Python using the [Keras](https://keras.io/) deep learning library.

## Authors

- Seran Shanmugathas
- Ferdinand T. Eide

## Requirements

- Python 3.5
- Keras 2.0.6
- Tensorflow 1.1.0
- Numpy 1.12.1

## Usage

### Training

To train the model, run the following command:

    python3 train.py

The training data is stored in the `data` directory. The model is saved in the `model` directory.

### Playing

To play against the model, run the following command:

    python3 play.py

The model is loaded from the `model` directory.

### Folder structure

```bash
alphaGo-knock-off/
├── game/
│ ├── __init__.py
│ ├── board.py
│ ├── moves.py
│ └── win_conditions.py
├── neural_network/
│ ├── __init__.py
│ ├── model.py
│ └── training.py
├── mcts/
│ ├── __init__.py
│ ├── node.py
│ └── search.py
├── reinforcement_learning/
│ ├── __init__.py
│ ├── actor.py
│ ├── critic.py
│ ├── environment.py
│ └── training.py
├── topp/
│ ├── __init__.py
│ ├── competition.py
│ ├── evaluation.py
│ └── ranking.py
├── main.py
└── requirements.txt
```
