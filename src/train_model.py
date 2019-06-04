import numpy as np

from __root__ import getPath
from src.SentenceClassifier import SentenceClassifier


def main():
    random_state = 42
    np.random.seed(random_state)

    root_dir = getPath()
    output_dir_path = root_dir + "/model"
    training_data_dir_path = root_dir + "/data"

    classifier = SentenceClassifier()
    batch_size = 64
    epochs = 30

    classifier.fit(training_data_dir_path=training_data_dir_path,
                   model_dir_path = output_dir_path,
                   batch_size=batch_size,
                   epochs=epochs,
                   random_state=random_state)


if __name__ == "__main__":
    main()