Guide to recreating results

## Train SimVP
The SimVP training script was setup with a number of options. However, you just need to run with mostly 
default parameters, and just need to point to the dataset. You likely also need to adjust the batch size;
I encountered a max batch size of 3 per GPU before the GPUs would run out of memory.

As an example, running in HPC used the following command.

```
python --train_data /dataset --batch_size 6 
```

Note the code for SimVP was partially sourced from https://www.kaggle.com/code/simuzilisen/simvp


## Train Unet
Unet was trained separately, and the script is less developed, so you need to go in and edit the file
to point to the datset. Specifically the two following lines need to be edited.

```
dataset = LabeledDataset('/scratch/py2050/Dataset_Student/')

val_dataset = ValidationDataset('/scratch/py2050/Dataset_Student/')
```

Then the script can just be run without any arguments
```
python train_unet.py
```

The Unet 

## Predict on Hidden Dataset

For predicting on the hidden dataset, you need to give the script arguments for the training data (root folder), hidden data
(subfolder with `video_*` children) and the two models. The script will predict on both the validation and hidden dataset,
and output a `simvp_unet_hidden.pt` file with the hidden dataset results.

Note the batch_size also needs to be adjusted to prevent running out of memory.

```
python predict_simvp_resnet.py --data /dataset --hidden_data scratch/hidden --simvp simvp.pth --unet best_model_unet_50_py.pth --batch_size 2 
```