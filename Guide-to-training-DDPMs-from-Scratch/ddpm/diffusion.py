from tqdm import tqdm
import torch
import torchvision.transforms as TF
import torchvision
from PIL import Image

from utils import get
from data_utils import inverse_transform
from utils import frames2vid, display


# Diffusion Process


class SimpleDiffusion:
    def __init__(
            self,
            num_diffusion_timesteps=1000,
            img_shape=(3, 64, 64),
            device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

        self_sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )


def forward_diffusion(sd: SimpleDiffusion, x0: torch.Tensor, timesteps: torch.Tensor):
    eps = torch.randn_like(x0)  # Noise
    mean = get(sd.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
    std_dev = get(sd.sqrt_one_minus_alpha_cumulative, t=timesteps)  # Noise scaled
    sample = mean + std_dev * eps  # scaled inputs * scaled noise

    return sample, eps  # return ... , gt noise --> model predicts this)


@torch.inference_mode()
def reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64),
                      num_images=5, nrow=8, device="cpu", **kwargs):
    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)),
                          total=timesteps - 1, dynamic_ncols=False,
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t = get(sd.beta, ts)
        one_by_sqrt_alpha_t = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts)

        x = (
                one_by_sqrt_alpha_t
                * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
                + torch.sqrt(beta_t) * z
        )

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = torchvision.utils.make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)

    if kwargs.get("generate_video", False):  # Generate and save video of the entire reverse process.
        frames2vid(outs, kwargs['save_path'])
        display(
            Image.fromarray(outs[-1][:, :, ::-1]))  # Display the image at the final timestep of the reverse process.
        return None

    else:  # Display and save the image at the final timestep of the reverse process.
        x = inverse_transform(x).type(torch.uint8)
        grid = torchvision.utils.make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = TF.functional.to_pil_image(grid)
        pil_image.save(kwargs['save_path'], format=kwargs['save_path'][-3:].upper())
        display(pil_image)
        return None
