import os
import random
import shutil, errno
import csv
import sys
import datetime

def rename_files(path, domain, class_name):
    os.chdir(path)
    print(os.getcwd())

    for count, f in enumerate(os.listdir()):
        f_name, f_ext = os.path.splitext(f)
        f_name = domain + "_" + class_name + "_" + str(count + 1)
        new_name = f'{f_name}{f_ext}'
        os.rename(f, new_name)


def move_files(source, dest, no_of_files):
    files = os.listdir(source)
    for file_name in random.sample(files, no_of_files):
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.move(os.path.join(source, file_name), dest)
    print("Total {} files moved!", no_of_files)


if __name__ == "__main__":
    
    #################################################
    # Assuming source_data contains DF,             #
    # F2F, FS, NT, Pristine folders.                 #
    # This script generates processed_data and         #
    # csv_files folders based on the requested        #
    # transfer learning task (X-->Y).                #
    #################################################
    
    # get user specified variables
    source = sys.argv[1]
    target = sys.argv[2]
    source_labeled_count = sys.argv[3]
    target_labeled_count = sys.argv[4]
    target_unlabeled_count = sys.argv[5]
    test_count = sys.argv[6]

    print("#################################################")
    print("Source Name: ", source)
    print("Target Name: ", target)
    print("Labeled Source Data Count: ", source_labeled_count)
    print("Labeled Target Data Count: ", target_labeled_count)
    print("Unlabeled Target Data Count: ", target_unlabeled_count)
    print("Test Data Count: ", test_count)
    print("#################################################")

    base_path = "/home/mdshamimseraj/Desktop/FakeImageDetection/UDA"
    
    # move files to intermediate folders
    move_files(base_path + "/source_data/" + source, base_path + "/temp_data/" + source + "/train/fake", int(int(source_labeled_count)/2))
    move_files(base_path + "/source_data/Pristine", base_path + "/temp_data/" + source + "/train/real", int(int(source_labeled_count)/2))
    move_files(base_path + "/source_data/" + source, base_path + "/temp_data/" + source + "/test/fake", int(int(test_count)/2))
    move_files(base_path + "/source_data/Pristine", base_path + "/temp_data/" + source + "/test/real", int(int(test_count)/2))

    move_files(base_path + "/source_data/" + target, base_path + "/temp_data/" + target + "/train/fake", int(int(target_labeled_count)/2))
    move_files(base_path + "/source_data/Pristine", base_path + "/temp_data/" + target + "/train/real", int(int(target_labeled_count)/2))
    move_files(base_path + "/source_data/" + target, base_path + "/temp_data/" + target + "/train_unlabelled/fake", int(int(target_unlabeled_count)/2))
    move_files(base_path + "/source_data/Pristine", base_path + "/temp_data/" + target + "/train_unlabelled/real", int(int(target_unlabeled_count)/2))
    move_files(base_path + "/source_data/" + target, base_path + "/temp_data/" + target + "/test/fake", int(int(test_count)/2))
    move_files(base_path + "/source_data/Pristine", base_path + "/temp_data/" + target + "/test/real", int(int(test_count)/2))

    # rename files in intermediate folders
    rename_files(base_path + "/temp_data/" + source + "/train/fake", source, "fake")
    rename_files(base_path + "/temp_data/" + source + "/train/real", source, "real")
    rename_files(base_path + "/temp_data/" + source + "/test/fake", source, "fake")
    rename_files(base_path + "/temp_data/" + source + "/test/real", source, "real")
    rename_files(base_path + "/temp_data/" + target + "/train/fake", target, "fake")
    rename_files(base_path + "/temp_data/" + target + "/train/real", target, "real")
    rename_files(base_path + "/temp_data/" + target + "/train_unlabelled/fake", target, "fake")
    rename_files(base_path + "/temp_data/" + target + "/train_unlabelled/real", target, "real")
    move_files(base_path + "/temp_data/" + target + "/train_unlabelled/fake", base_path + "/temp_data/" + target + "/unlabelled", int(int(target_unlabeled_count)/2))
    move_files(base_path + "/temp_data/" + target + "/train_unlabelled/real", base_path + "/temp_data/" + target + "/unlabelled", int(int(target_unlabeled_count)/2))
    rename_files(base_path + "/temp_data/" + target + "/unlabelled", target, "u")
    rename_files(base_path + "/temp_data/" + target + "/test/fake", target, "fake")
    rename_files(base_path + "/temp_data/" + target + "/test/real", target, "real")
    
    # move files from intermediate folders to the processed_data folder
    move_files(base_path + "/temp_data/" + source + "/train/fake", base_path + "/processed_data/train", int(int(source_labeled_count)/2))
    move_files(base_path + "/temp_data/" + source + "/train/real", base_path + "/processed_data/train", int(int(source_labeled_count)/2))
    move_files(base_path + "/temp_data/" + source + "/test/fake", base_path + "/processed_data/test_source", int(int(test_count)/2))
    move_files(base_path + "/temp_data/" + source + "/test/real", base_path + "/processed_data/test_source", int(int(test_count)/2))

    move_files(base_path + "/temp_data/" + target + "/train/fake", base_path + "/processed_data/train", int(int(target_labeled_count)/2))
    move_files(base_path + "/temp_data/" + target + "/train/real", base_path + "/processed_data/train", int(int(target_labeled_count)/2))
    move_files(base_path + "/temp_data/" + target + "/test/fake", base_path + "/processed_data/test_target", int(int(test_count)/2))
    move_files(base_path + "/temp_data/" + target + "/test/real", base_path + "/processed_data/test_target", int(int(test_count)/2))

    move_files(base_path + "/temp_data/" + target + "/unlabelled", base_path + "/processed_data/train", int(target_unlabeled_count))

    shutil.rmtree(base_path + "/temp_data")
    #shutil.rmtree(base_path + "/source_data")
    
