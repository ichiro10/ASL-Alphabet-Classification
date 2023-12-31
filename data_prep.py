import os
import random
import shutil
import cv2

# Define constants for paths
DATASET_PATH = './Dataset/image_dataset'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
VALID_PATH = os.path.join(DATASET_PATH, 'valid')
TEST_PATH = os.path.join(DATASET_PATH, 'test')

def create_test_set(test_path, train_path):
    # Iterate through each directory in the training set
    for letter in os.listdir(train_path):
        test_letter_path = os.path.join(test_path, letter)

        # Create the test category directory if it doesn't exist
        if not os.path.isdir(test_letter_path):
            os.mkdir(test_letter_path)

        # Get a list of files in the current training category
        files = os.listdir(os.path.join(train_path, letter))

        # Move a random selection of 10 files from training to test
        for _ in range(10):
            if files:
                random_file = random.choice(files)
                file_path = os.path.join(train_path, letter, random_file)
                shutil.move(file_path, test_letter_path)
                files.remove(random_file)

def resize_data(test_path, train_path, valid_path):
    for path in [test_path, train_path, valid_path]:
        for letter in os.listdir(path):
            for instance in os.listdir(os.path.join(path, letter)):
                image = os.path.join(path,letter, instance)
                im = cv2.imread(image)
                print(im.shape)
                img = cv2.resize(im, (224,224))
                cv2.imwrite(os.path.join(path,letter, instance), img)

def main():
    # Create the 'test' directory if it doesn't exist
    if not os.path.isdir(TEST_PATH):
        os.mkdir(TEST_PATH)

    # Call the function to create the test set
    create_test_set(TEST_PATH, TRAIN_PATH)

    resize_data(TEST_PATH, TRAIN_PATH, VALID_PATH)   

if __name__ == "__main__":
    main()
