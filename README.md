# BMLProject
Monte Carlo Diffusion Models

# Dependencies
Check and install all dependencies
```
pip install requirements.txt 
```
accelerate==0.14.0, torch==1.8.1, diffusers==0.7.2

# Run Examples
## Train
Changing hyperparameters in "config_dict" dictionary of corresponding files and with Accelerate from https://huggingface.co/docs/accelerate/index installed 

Run DDIM train:
```
accelerate launch ddim_train.py
```
Run DDPM train:
```
accelerate launch ddpm_train.py
```
## Evaluation
Run DDIM evlauation:
```
python ./eval/eval_ddim.py --model <model_path> --start_range <ensemble start step> --end_range <ensemble end step> --samples <number of ensemble samples> --batch_size <batch size> --num_images <number generated images> --name <dictionary name>
```
Run DDPM evlauation:
```
python ./eval/eval_ddpm.py --model <model_path> --start_range <ensemble start step> --end_range <ensemble end step> --samples <number of ensemble samples> --batch_size <batch size> --num_images <number generated images> --name <dictionary name>
```
