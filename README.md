UNET GEDI Forest Canopy Height , AGB Estimation
==============================

An UNET probabilistic deep learning model for estimating  forest structure data such as canopy height/above ground biomass from sentinel 1 , sentinel 2 and GEDI data

Full Documenatation
==============================
https://rsongphon.github.io/Satellite-Based-AGB-CHM-Estimation-Model-

Prerequsite
==============================
- Pytorch
- torchsummary
- torchmetrics
- tqdm
- rasterio
- numpy
- matplotlib
- wandb

Metodology
==============================
![flow_model](/docs/images/flow_model.jpeg)<br>
- Full Information about methodology is describe in paper <br>
- A DEEP LEARNING APPROACH WITH UNCERTAINTY ESTIMATION TO ASSESSMENT THE POTENTIAL OF ABOVEGROUND BIOMASS MAPPING OF TROPICAL RAINFOREST IN THAILAND <br>
- IEEE M2GARSS 2024 <br>
- locate at /docs/papers <br>

Data Preprocessing (Earth Engine)
==============================
### Earth engine implementation 
- Code in javascript file must be use in earth engine code editor <br>
- The script will generate raster file following the paper procedures<br>

##### import target shapefile in earth engine coding environment 
- see documenatation <br>
- https://developers.google.com/earth-engine/guides/table_upload <br>

#### Input : <br>    
- Sentinel 2 + 1 stack image <br>
- 14 channel image numpy .npy format <br> 
- band can be in any order if traing from scratch but to follow the paper's methodology use,<br>
- B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 , VV(ASC) , VH(ASC)<br>
- model support any spatial resolution , input/output must have the same<br>

#### Script Location <br>

- Input Sentinel 1+2 preprocessing script  <br>
    - /src/utils/EE_input_query.js <br>

#### Reference : <br>  
- GEDI rasterize sparse AGB/canopy height image<br>
- leave pixel that have no value as NAN <br>
- 1 channel image<br>

#### Script Location <br>

- Reference GEDI L4A  <br>
    - /src/utils/EE_GEDI_L4A_query.js <br>

- Reference GEDI L2A  <br>
    - /src/utils/EE_GEDI_L2A_query.js <br>


Train/Val/Test dataset preparation
==============================
![sampling](/docs/images/sampling.png)<br>
- To use sentinel 1 , 2 and GEDI to train model. Perform uniform random sampling over region and convert to numpy .npy format (see picture as guideline)<br>
- Split the data into train/val/test subset with the same name for each instance input/groundtruth.<br>
- Dataset directory should look like <br>
#### Sentinel 1, 2 Input
```console
    /[Input Dir Name]
        /train
            - abc.npy 
            ...
        /test
            - def.npy 
            ...
        /val
            - ijk.npy 
            ...
```
#### GEDI ground truth
```console
    /[GT Dir Name]
        /train
            - abc.npy 
            ...
        /test
            - def.npy 
            ...
        /val
            - ijk.npy 
            ...
```
        
Training
==============================
#### Set up wandb API key to record model evaluation metric

The training script use both tensorboard to store model evaluation result locally as well as WANDB <br>

tensorboard result is locate at /log dir at output directory <br> 

```console
> export WANDB_API_KEY=[YOUR_WADB_API_KEY]
```
### Traing from scratch
```console
> usage:  train.py [--train_input_data_dir TRAIN_DIR] [--train_label_data_dir GT_DIR] [--all_checkpoint_dir CKP_DIR]

Train the UNet on Sentinel 2 + 1 stack image and GEDI masks

optional arguments:
  -h, --help                        show this help message and exit
  --out_dir                         output directory for the experiment , default='./tmp/'
  --nb_epoch E                      Number of epochs , default=100
  --batch_size B                    Batch size, default=32
  --base_learning_rate BLR          Initial learning rate, default=1e-7            
  --l2_lambda  L2                   L2 regularizer on weights hyperparameter, default=1e-8          
  --optimizer ['ADAM', 'SGD']       optimizer , default= 'ADAM'
  --momentum MT                     momentum for SGD, default=0.9
  --gradient_clipping G             gradient_clipping, default=1.0
  --amp [True , False]              Use mixed precision , default = False
  --bilinear [True , False]         Use bilinear upsampling, default = True
  --num_ch CH                       Number of input_channels, default=14
  --save_checkpoint [True , False]  Save weight every epoch? , default = True                 
```
### Traing from pretrained weight

```console

> usage:  train.py [--train_input_data_dir TRAIN_DIR] [--train_label_data_dir GT_DIR] [--all_checkpoint_dir CKP_DIR] [--model_weights_path MWP] [--train_mode 'resume']

Train the UNet on Sentinel 2 + 1 stack image and GEDI masks from previous checkpoins

Note : --model_weights_path MWP     path of model checkpoint include checkpoint itself eg; /path/to/checkpoint.pth

optional arguments are the same.              
```
Evaluation
==============================

### Evaluate one model

Each model after finished training , the code will run validation with test set automatically<br>

The output prediction , variance , evaluation result will locate at the same folder that store best_weights.pt (specify by argument --out_dir) <br>

If you want to run the test manually use command  <br>
```console

> usage:  train.py [--train_input_data_dir TRAIN_DIR] [--train_label_data_dir GT_DIR] [--all_checkpoint_dir CKP_DIR] [--skip_training True]  [--out_dir MODEL_WEIGHT_PATH]
            
```

### Evaluate ensemble model

To measure total variance of all candidate model and evaluate ensemble with test set <br>

Create a directory and place each model's directory contain evaluation result from step 'Evaluate one model' <br>

Rename each model directory to 'model_i' , subscript i with number of candiate (1 <= i <= n) <br>

For example : parent directory name is 'ensemble_best_weight' <br>
              candidate ensemble n = 5<br>

Directory hierarchy must be follow<br>

```console

/ensemble_best_weight
    /model_1
        /log
        best_weights.pt
        confusion.png
        predictions.npy
        results.txt
        targets.npy
        test_results.json
        variances.npy
    /model_2
        ...
    /model_3
        ....
    /model_4
        ....
    /model_5
        ....

```

To collect ensemble result, use command <br>

```console

> collect_ensembles.py [WEIGHTS_DIRS]  [N]

Example usage : python3 collect_ensembles.py path/to/ensembleweights  5

arguments:
WEIGHTS_DIRS        absolute path of model ensemble weight's collection parent directory
N                   total number of candidate ensemble

```
 
The script will creates:<br>
    1) a new subdirectory in e.g. "/experiment_dir/ensemble" # for futher implementation of K-fold cross validation,<br>
    2) a new subdirectory in the experiment base directory e.g. "/experiment_dir/ensemble_collected", containing the collected ensemble predictions and results <br>


Prediction
==============================

### Prediction on one model

Input : <br>    
- Sentinel 2 + 1 stack rasterimage <br>
- 14 channel .tif or tiff<br> 
- same band arrangement as training image

Output :
- 1 channel output prediction, raster image (.tif) <br> 
- 1 channel variance of output prediction, raster image (.tif) <br> 

```console

> usage:  predict_model.py  [--model_path  MODEL_DIRECTORY] [--input INPUT_FILE_PATH] [--output OUTPUT_FILE_PATH]

Note : MODEL_DIRECTORY , Specify the path in which the candidate ensemble model best_weights.pt is stored

optinal arguments:
  --channels num_ch         Number of input_channels , default=14
  --bilinear BL             Use bilinear upsampling,  default=True 
  --window_size WINDOW      Define size of window use in sliding window technique , default = 2096
  --stride STIRDE           Step size for window sliding  , default = 2000

```

### Merge ensemsble prediction

![inference](/docs/images/inference_concept.jpg)

To product final result from all ensemble candidate, the procedures are follows <br> 

1. prepare all of ensemble prediction output , variances.<br> 
The approach depends on computational resources either compute by subset or whole image.<br>


![mean](/docs/images/mean.png)<br>
1. avarage per-pixel output predictions  <br> 
    - Stacking all prediction output along the same axis and calculate pixel-wise avarage.<br>
    - Avarage of output can be implement be numpy.mean<br>
```console

    > pred_ensemble = np.mean(predicts_img, axis=0)

    Input : 
    predicts_img     (n,1,width,height) array of n prediction output images

    Output : 
    pred_ensemble   avarage prediction of all ensemble candidate (1,width,height)

```
![variance](/docs/images/variance.png)<br>
2. Calculate Total Variance of the ensemble model<br>
- Stacking all prediction output , variance along the same axis <br>
- Calculte epistemic uncertainty (model uncertainty)<br>
    - Can be done by calculate per pixel varince of all output predictions<br>

```console

    > epistemic_var = np.var(predicts_img, axis=0)

    Input : 
    predicts_img            (n,1,width,height) array of n prediction output images
    
    Output : epistemic_var  model uncertainty (1,width,height)
 
```
- Calculte aleatoric (data uncertainty)<br>
    - Can be done by calculate per-pixel avarage on all variances of ensemble<br>

```console

    > aleatoric_var = np.mean(variances_img, axis=0)

    Input : 
    variances_img     (n,1,width,height) array of n variance output of the models
    
    Output : 
    aleatoric_var     data uncertainty (1,width,height)
 
```

- Combine epistemic and aleatoric uncertainty<br>
    - Sum epistemic and aleatoric by total variance formula<br>

```console

    > predictive_var = epistemic_var + aleatoric_var

    Input : 
    epistemic_var       model uncertainty (1,width,height)
    aleatoric_var       data uncertainty (1,width,height)
    
    Output : 
    predictive_var      total uncertainty (1,width,height)
 
```

3. output pred_ensemble and predictive_var then can be write into raster image by embedding geospatial data.<br>
