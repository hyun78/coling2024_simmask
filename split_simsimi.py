import pickle
import random
from dataset import train_test_split
_SAMPLE_PATH = "../teaching_ukusca_19_21_sample10k.pkl"
_SAVE_PATH_ = "../"
if __name__ == "__main__":

    random.seed(0)

    with open(_SAMPLE_PATH, 'rb') as file:
        data = pickle.load(file)

        train_data, _, test_data = train_test_split(data, 0.8, 0, 0.2)

        with open(_SAVE_PATH_+"teaching_train.pkl", 'wb') as train_file:

            pickle.dump(train_data, train_file)
        
        with open(_SAVE_PATH_+"teaching_test.pkl", 'wb') as test_file:

            pickle.dump(test_data, test_file)
