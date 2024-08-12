# MHIST Classification Project

This repository contains the code and configurations for training classification models on the MHIST dataset. The models include ResNet variants and Vision Transformers (ViT).

## --- Install Conda Virtual Environment ---
Install the Conda virtual environment using the given .yml file as shown below: \
conda env create -f pytorch-env-mhist.yml

## --- Download the MHIST data and save the images ---
After downloading the data, save all the images inside the 'data/images' folder.


## --- MHIST Data Distribution Visualization ---

Please run the code `python data_dist_visual.py` or run the following Jupyter Notebook files:  
- **`data_dist_plt_bar_visual.ipynb`**: For displaying the data distribution using `plt.bar` (from `matplotlib`)  
- **`data_dist_sns_visual.ipynb`**: For displaying the data distribution using `sns` (`seaborn`)

The distribution plots will be saved in the `dataset_plots` folder.



## --- MHIST Classification Using ResNet Model ---

Before running the training code, please refer to the `config.yaml` file inside the `configs` directory to adjust the hyperparameters.

To run different ResNet variants, such as ResNet18 or ResNet34, change the `architecture` parameter to `"resnet18"` or `"resnet34"` (Options: `resnet18`, `resnet34`, `vit`, etc.). You can also choose whether to use a pretrained model or fine-tune the model with a specified number of layers.

For example, to use ResNet18 or ResNet34 with a pretrained model and fine-tune one layer, set the following in your `config.yaml`:

```yaml
architecture: "resnet18"  # or "resnet34"
pretrained: True
fine_tune_layers: 1
```

## --- MHIST Classification Using ViT Model ---

Similarly, before running the training code for the ViT model, refer to the `config.yaml` file inside the `configs` directory to adjust the hyperparameters. To run different ViT variants, such as variant: "base"  # Options: 'base', 'large', 'huge','custom-vit', change the `architecture` parameter to `"vit"` and then the variant parameter to your desired variant. (variants: 'base', 'large', 'huge','custom-vit', etc.). You can also choose whether to use a pretrained model or fine-tune the model with a specified number of layers. For example, to use ViT with a pretrained model and fine-tune one layer, set the following in your `config.yaml`:

```yaml
architecture: "vit"
variant: 'base'  # or 'large', 'huge','custom-vit'
pretrained: True
fine_tune_layers: 1
```
### Running the Training

To train the ViT model, use the following command:

```bash
python main.py
```

### Resuming Training

To resume training from a previous checkpoint, set resume: True in the config.yaml file:

```bash
resume: True
```


### Running the Testing

To evaluate the performance of the trained model on the test dataset, use the following command:

```bash
python test.py
```

### Running GradCam to Generate Heatmaps
To generate GradCam heatmaps, which visually explain the model's predictions by highlighting important regions in the input images, use the following command:

```bash
python test_gradcam.py
```
