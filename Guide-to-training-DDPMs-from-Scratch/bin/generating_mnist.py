import gc
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda import amp
from torchvision.utils import make_grid

from ddpm.config import BaseConfig, TrainingConfig, ModelConfig
from ddpm.data_utils import get_dataloader
from ddpm.data_utils import inverse_transform
from ddpm.utils import setup_log_directory
from ddpm.models import UNet
from ddpm.diffusion import SimpleDiffusion, forward_diffusion, reverse_diffusion
from ddpm.training import train_one_epoch

def main():
    # Visualize Dataset

    loader = get_dataloader(
        dataset_name=BaseConfig.DATASET,
        batch_size=128,
        device='cpu',
    )

    plt.figure(figsize=(12, 6), facecolor='white')

    for b_image, _ in loader:
        b_image = inverse_transform(b_image).cpu()
        grid_img = make_grid(b_image / 255.0, nrow=16, padding=True, pad_value=1, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis("off")
        break

    ## Sample Forward Diffusion Process

    sd = SimpleDiffusion(num_diffusion_timesteps=TrainingConfig.TIMESTEPS, device="cpu")

    loader = iter(  # converting dataloader into an iterator for now.
        get_dataloader(
            dataset_name=BaseConfig.DATASET,
            batch_size=6,
            device="cpu",
        )
    )

    x0s, _ = next(loader)

    noisy_images = []
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = forward_diffusion(sd, x0s, timestep)
        xts = inverse_transform(xts) / 255.0
        xts = make_grid(xts, nrow=1, padding=1)

        noisy_images.append(xts)

    # Plot and see samples at different timesteps

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor='white')

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    plt.show()


    # Training

    # Algorithm 1: Training



    # Algorithm 2: Sampling


    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    model.to(BaseConfig.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

    dataloader = get_dataloader(
        dataset_name  = BaseConfig.DATASET,
        batch_size    = TrainingConfig.BATCH_SIZE,
        device        = BaseConfig.DEVICE,
        pin_memory    = True,
        num_workers   = TrainingConfig.NUM_WORKERS,
    )

    loss_fn = nn.MSELoss()

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    scaler = amp.GradScaler()

    total_epochs = TrainingConfig.NUM_EPOCHS + 1
    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

    generate_video = False
    ext = ".mp4" if generate_video else ".png"

    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        # Algorithm 1: Training
        train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)

        if epoch % 1 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")

            # Algorithm 2: Sampling
            reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=32, generate_video=generate_video,
                              save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
                              )

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict


    # Inference

    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    # checkpoint_dir = "/kaggle/working/Logs_Checkpoints/checkpoints/version_0"


    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])

    model.to(BaseConfig.DEVICE)

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    log_dir = "inference_results"
    os.makedirs(log_dir, exist_ok=True)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        num_images=64,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=8,
    )
    print(save_path)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        num_images=256,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=32,
    )
    print(save_path)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        num_images=256,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=16,
    )
    print(save_path)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        num_images=256,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=16,
    )
    print(save_path)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        num_images=512,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=32,
    )
    print(save_path)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        num_images=16,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=4,
    )
    print(save_path)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)
    print(save_path)

    reverse_diffusion(
        model,
        sd,
        num_images=128,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=8,
    )
    print(save_path)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)
    print(save_path)

    reverse_diffusion(
        model,
        sd,
        num_images=256,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=16,
    )
    print(save_path)

    generate_video = False

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)


    reverse_diffusion(
        model,
        sd,
        num_images=128,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=16,
    )
    print(save_path)


if __name__ == '__main__':
    main()
