import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import diffusers
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
import sys
import utils
import argparse
from itertools import chain
from dataclasses import dataclass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM_CIFAR10_TEST")
    parser.add_argument("--model_name", choices=["DDPM", "LDM"], type=str, default="DDPM")
    parser.add_argument("--vae_model", choices=["CompVis/stable-diffusion-v1-4"], type=str, default="CompVis/stable-diffusion-v1-4")    
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_timesteps", type=int, default=1000)
    parser.add_argument("--dataset", choices=["CIFAR10", "FFHQ"], type=str, default="CIFAR10")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--save_model_epochs", type=int, default=1)
    parser.add_argument("--save_image_epochs", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--output_dir", type=str, default="cifar10_saved")
    parser.add_argument("--overwrite_output_dir", default=False, action="store_true")
    parser.add_argument("--user_token", type=str, default="")
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    return args  


def get_dataloader(configs, image_transforms):
    if config.dataset == "FFHQ":
        # FFHQ dataset
        #splits = list(range(0, 100))
        dataset = utils.ffhq_Dataset("dataset/ffhq/thumbnails128x128/", image_transforms)
        #dataset = torch.utils.data.Subset(dataset, splits)
        # set original image size for DDPM and latent feature size for LDM
        if config.model_name == "DDPM":
            config.image_size = 128
        elif config.model_name == "LDM":
            config.image_size = 16
        else:
            raise NotImplementedError("Model Name not Implemented!")
    elif config.dataset == "CIFAR10":
        #cifar10 dataset
        dataset = torchvision.datasets.CIFAR10(root= "../dataset/", download=True, transform=image_transforms)
        config.image_size = 32
    elif config.dataset == "CIFAR100":
        #cifar100 dataset
        dataset = torchvision.datasets.CIFAR100(root= "../dataset/", download=True, transform=image_transforms)
        config.image_size = 32
    else:
        raise ValueError("Invalid Dataset supplied")

    train_loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


if __name__ == "__main__":
    config = get_args()
    #transforms
    image_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # get trainloader
    train_loader =  get_dataloader(config, image_transforms)

    models = dict()
    # get models, optimizer and noise scheduler
    if config.model_name == "DDPM":
        backbone_model = utils.get_default_unet(config)
        models["backbone"] = backbone_model
        model_params = backbone_model.parameters()
        optimizer = torch.optim.AdamW(model_params, lr=config.learning_rate)
        noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps) 
    elif config.model_name == "LDM":
        backbone_model = utils.get_unconditional_unet(config)
        #vae_model = AutoencoderKL.from_pretrained(config.vae_model, subfolder="vae", use_auth_token=config.user_token) 
        vae_model = utils.get_custom_vae(config)
        models["backbone"] = backbone_model
        models["vae"] = vae_model
        model_params = chain(backbone_model.parameters(), vae_model.parameters())
        optimizer = torch.optim.AdamW(model_params, lr=config.learning_rate)
        noise_scheduler = DDIMScheduler(num_train_timesteps=config.num_train_timesteps)
    else:
        raise NotImplementedError("Model Name not Implemented!")

    # get learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_loader) * config.num_epochs),
    )

    # train model
    utils.train_loop(config, models, noise_scheduler, optimizer, train_loader, lr_scheduler)


