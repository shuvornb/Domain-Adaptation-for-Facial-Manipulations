#!/bin/bash

# Echo current time
now=$(date)
echo "Current time: $now"

# Copy 3 subfolders (ex- DF, F2F, Pristine) into a folder named source_data
#python3 create_source_data.py FS DF
mkdir source_data
cp -r videos/manipulated_sequences/Images/F2F source_data
cp -r videos/manipulated_sequences/Images/DF source_data
cp -r videos/manipulated_sequences/Images/Pristine source_data

# Generate processed_data from source_data under UDA directory
python3 create_processed_data.py F2F DF 42000 2000 6000 10000

# Generate necessary csv files under UDA/csv_files
python3 create_csv_data.py F2F DF 50000 10000

# Preprocess images
python3 preprocess_image_data.py F2F DF

# Train UDA model with preprocessed data and save the model
python3 train.py F2F DF 1

# Test UDA model and save the results
python3 test.py F2F DF 1

# Delete unnecessary folders
rm -rf source_data
rm -rf processed_data
rm -rf csv_files

# Echo current time
now=$(date)
echo "Current time: $now"
