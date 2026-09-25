# Chess Engine

A this is a chess engine written from scratch in Python.

## Features

* Alpha-beta search with quiescence search and a transposition table
* Move ordering, SEE pruning, and time management
* Handcrafted evaluation plus an NNUE-style neural network evaluation (work in progress)
* Pygame GUI to play against the engine
* Benchmarking and testing tools

## Technologies

* Python
* bulletchess
* TensorFlow / Keras
* NumPy
* Pygame

## License

GPL-3.0 - see [LICENSE](LICENSE).
## Dataset

Training data is based on **Lichess Chess Positions: ML-Ready Deduplicated Evaluations** by **Mateusz Grzyb**, licensed under **CC BY 4.0**.

The original chess positions and evaluations were converted to the **HalfKP representation** and processed for training the chess evaluation model.

Original dataset: Grzyb, Mateusz (2025), *Lichess Chess Positions: ML-Ready Deduplicated Evaluations*.
URL: https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized
