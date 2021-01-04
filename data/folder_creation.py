import os


def create_data_folders() -> None:
    """Creates empty folders for data storage."""
    # os.makedirs("raw_data/csv_files/kaggle")
    # os.makedirs("raw_data/numpy_data/kaggle")
    # os.makedirs("raw_data/csv_files/nfl_scrapr")
    # os.makedirs("raw_data/numpy_data/nfl_scrapr")
    os.makedirs("dictionaries/games")
    os.makedirs("dictionaries/plays")


if __name__ == "__main__":
    create_data_folders()
