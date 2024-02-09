import os
import random
import shutil, errno
import csv
import sys
import datetime
import sys
import os


def generate_train_csv(no_of_batches, batch_size, no_of_classes, per_class_sample_count, per_class_source_count, per_class_target_count, source_name, target_name, class_names):
    curr_time = datetime.datetime.now()
    file_name = 'csv_files/train_data.csv'
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["Full_Path", "Category", "Domain"]
        writer.writerow(field)

        source_counter = [1]*no_of_classes
        target_counter = [1]*no_of_classes
        unlabeled_counter = 1

        for i in range(no_of_batches):

            # fill the labeled elements
            for c in range(no_of_classes):
                # fill the source elements
                for s in range(per_class_source_count):
                    writer.writerow(['processed_data/train/' + source_name + '_' + class_names[c] + '_' + str(source_counter[c]) + '.png', class_names[c], source_name])
                    source_counter[c] += 1


                # fill the target elements
                for t in range(per_class_target_count):
                    writer.writerow(['processed_data/train/' + target_name + '_' + class_names[c] + '_' + str(target_counter[c]) + '.png', class_names[c], target_name])
                    target_counter[c] += 1

            
            # fill the unlabeled elements
            unlabeled_data_count = batch_size - no_of_classes*per_class_sample_count
            for u in range(unlabeled_data_count):
                writer.writerow(['processed_data/train/' + target_name + '_' + 'u' + '_' + str(unlabeled_counter) + '.png', 'unlabeled', target_name])
                unlabeled_counter += 1


def generate_test_csv(no_of_classes, per_class_sample_count, source_name, target_name, class_names):
    
    curr_time = datetime.datetime.now()
    source_file_name = 'csv_files/test_data_source.csv'
    target_file_name = 'csv_files/test_data_target.csv'

    with open(source_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["Full_Path", "Category", "Domain"]
        writer.writerow(field)

        source_counter = [1]*no_of_classes

        for c in range(no_of_classes):

            for i in range(per_class_sample_count):

                writer.writerow(['processed_data/test_source/' + source_name + '_' + class_names[c] + '_' + str(source_counter[c]) + '.png', class_names[c], source_name])
                source_counter[c] += 1

    with open(target_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["Full_Path", "Category", "Domain"]
        writer.writerow(field)

        target_counter = [1]*no_of_classes

        for c in range(no_of_classes):

            for i in range(per_class_sample_count):

                writer.writerow(['processed_data/test_target/' + target_name + '_' + class_names[c] + '_' + str(target_counter[c]) + '.png', class_names[c], target_name])
                target_counter[c] += 1


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
    train_count = sys.argv[3]
    test_count = sys.argv[4]

    print("#################################################")
    print("Source Name: ", source)
    print("Target Name: ", target)
    print("Train Data Count: ", train_count)
    print("Test Data Count Per Domain: ", test_count)
    print("#################################################")

    # create base directory
    directory = "csv_files"
    parent_dir = "/home/mdshamimseraj/Desktop/FakeImageDetection/UDA/"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    
    # generate necessary csv files
    batch_size = 250
    no_of_batches = int(int(train_count)/batch_size)
    generate_train_csv(no_of_batches, batch_size, 2, 110, 105, 5, source, target, ['fake', 'real'])
    generate_test_csv(2, int(int(test_count)/2), source, target, ['fake', 'real'])
    
