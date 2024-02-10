# Domain-Adaptation-for-Facial-Manipulations
With the advent and popularity of generative models such as GANs, synthetic image generation and manipulation has become commonplace. This has promoted active research in the development of effective deepfake detection technology. While existing detection techniques have demonstrated promise, their performance suffers when tested on data generated using a different faking technology, on which the model has not been sufficiently trained. This challenge of detecting new types of deepfakes, without losing its prior knowledge about deepfakes (catastrophic forgetting), is of utmost importance in today's world. In this project, we propose a novel deep domain adaptation framework to address this important problem in deepfake detection research.

## Multi Source Domain Adaptation
### Dataset Structure
For example- if source domains are `DeepFake`, `Face2Face` and target domain is `FaceSwap`, then a directory named `dataset` with three sub-directories are expected with corresponding image files. They are- `DeepFake`, `Face2Face`, and `FaceSwap`. Each of these sub-directories should have three more subdirectories- `train`, `test`, and `val`. `test` and `val` sub-directories should have two more sub-directories- `fake`, and `real`. `train` directory should have the similar structure [`fake`, `real`] for `DeepFake` and `Face2Face` i.e source domains. But for `FaceSwap`, `train` should have `labeled` and `unlabeled` sub-directories and under them [`fake`, `real`] sub-directories.

### How to Run
Here are the instructions on how to train and test the model-
1. Train the model-
```
python3 create_target_data.py
```
2. Test the model-
```
python3 multi_train.py
```

## Single Source Domain Adaptation
### Dataset Structure
For example- if source domain is `DF` and target domain is `FS`, then a directory named `source_data` with three directories are expected with corresponding image files. They are- `DF`, `FS`, `Pristine`.
### How to Run
Here are the instructions on how to train and test the model-
1. Generate processed_data from source_data. The command expects 6 arguments. They are- `source_domain`, `target_domain`, `labeled_source_data_count`, `labeled_target_data_count`, `unlabeled_target_data_count`, `test_data_count`.
```
python3 create_processed_data.py F2F DF 42000 2000 6000 10000
```
2. Generate necessary csv files. The command expects 4 arguments. They are- `source_domain`, `target_domain`, `train_data_count`, `test_data_count`.
```
python3 create_csv_data.py F2F DF 50000 10000
```
3. Preprocess images. The command expects 2 arguments. They are- `source_domain`, `target_domain`.
```
python3 preprocess_image_data.py F2F DF
```
4. Train the model with preprocessed data and save the model. The command expects 3 arguments. They are- `source_domain`, `target_domain`, `trial_count`.
```
python3 train.py F2F DF 1
```
5. Test the model and save the results. The command expects 3 arguments. They are- `source_domain`, `target_domain`, `trial_count`.
```
python3 test.py F2F DF 1
```
6. Delete unnecessary folders
```
rm -rf source_data
rm -rf processed_data
rm -rf csv_files
```
