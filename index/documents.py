import ir_datasets
import string

from sklearn.datasets import fetch_rcv1

def tokenizer(text):
    lower_doc = text.lower()
    without_punct = lower_doc.translate(str.maketrans('', '', string.punctuation))
    return without_punct.split()

def load_dataset(DATASET_NAME):
    """
    Load the dataset using ir_datasets and return the dataset object.
    Args:
        DATASET_NAME (str): The name of the dataset to load. "rcv1" and dataset supported by ir_datasets are accepted.
    Returns:
        dataset: The loaded dataset object. 
    """

    if DATASET_NAME == "rcv1":
        dataset = fetch_rcv1()
    else:
        try:
            dataset = ir_datasets.load(DATASET_NAME)
        except Exception as e:
            raise ValueError(f"Dataset {DATASET_NAME} not found, error: {e}") 

    return dataset
