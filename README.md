# Defect binary segmentation using AttentionUnet

## Environment
- We provided yaml and requirement.txt to makeup the environment
```shell=
# create conda environment
conda env create -f environment.yaml
# activate environment
conda activate unet

# install other package with pip
pip install -r requirements.txt

```

## Training data format
- **Please follow the format we provided.**
- The training data should be format as:
```
training_data_folder
|__ img
|__ mask
|__ testing
```
## How to use?
### Training
- Run `Unet.py` to train your own model.
- We have provide our model in `models/*`
- We have also provided the usage in `Unet.py` line 93
  ![alt text](image.png) 
```shell=
python Unet.py --data_folder path/to/the/training/folder --epochs <training epochs> --batch_size <batch_size> --model_name <the name for the best model to save> 

# train dent dataset
python Unet.py --data_folder ./data/dent --model_name best_dent.keras 
# train scratch dataset
python Unet.py --data_folder ./data/scratch --model_name best_scratch.keras

```
| Argument        | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| `--data_folder` | Path to the training data folder. Ensure it follows the provided format. |
| `--epochs`      | Number of training epochs, default=800                                   |
| `--batch_size`  | Size of each training batch, default=4                                   |
| `--model_name`  | Name for saving the trained model.                                       |
| `--DEBUG`       | Enable debug messages (default: `False`).                                |
| `--TRAIN`       | Whether to train the model or not (default: `True`).                     |

### Testing
- Run `generate.py` to train your own model.
- We have provide our model in `models/*`
- We have also provided the usage in `generate.py` line 59
![alt text](image-1.png)
```shell=
python generate.py --model_name <the name for the best model to save> --data_folder path/to/the/inference/folder --result_folder path/to/the/result/folder

# test dent dataset
python generate.py --model_name best_dent.keras --data_folder ./data/dent/testing --result_folder ./result/inference/dent
# test scratch dataset
python generate.py --model_name best_scratch.keras --data_folder ./data/scratch/testing --result_folder ./result/inference/scratch

```

| Argument          | Type  | Default Value              | Description                                               |
| ----------------- | ----- | -------------------------- | --------------------------------------------------------- |
| `--model_name`    | `str` | `'best_scratch.keras'`     | Name of the model to save or load.                        |
| `--data_folder`   | `str` | `'./data/scratch/testing'` | Path to the folder containing testing data.               |
| `--result_folder` | `str` | `'./result/inference'`     | Path to the folder where inference results will be saved. |
