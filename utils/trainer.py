from nn import (
  Generator,
  Discriminator
)

import argparse
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def get_argparser():
  parser = argparse.ArgumentParser(prog="trainer",
                      description="training loop for GAN")
  parser.add_argument("--root", type=str, default=os.path.join(".", "data", "datasets"))
  parser.add_argument("--dataset", type=str, default="MNIST",
                      choices=["MNIST", "FashionMNIST"])
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--pin_memory", action="store_true", default=True)
  parser.add_argument("--num_epochs", type=int, default=100)
  parser.add_argument("--image_height", type=int, default=32)
  parser.add_argument("--image_width", type=int, default=32)
  parser.add_argument("--generator_lr", type=float, default=0.0001)
  parser.add_argument("--discriminator_lr", type=float, default=0.0001)
  parser.add_argument("--optimizer", type=str, default="Adam",
                      choices=["Adam"])
  parser.add_argument("--beta1", type=float, default=0.5)
  parser.add_argument("--beta2", type=float, default=0.999)
  parser.add_argument("--latent_dim", type=int, default=100)
  parser.add_argument("--generator_in_features", type=int, nargs="+", default=None)
  parser.add_argument("--discriminator_in_features", type=int, nargs="+", default=None)
  parser.add_argument("--use_spectral_norm", action="store_true")
  parser.add_argument("--compile", action="store_true", default=False)
  parser.add_argument("--ckpt_dir", type=str, default=os.path.join(".", "checkpoints"))
  parser.add_argument("--save_ckpt_interval", type=int, default=200)
  parser.add_argument("--results_dir", type=str, default=os.path.join(".", "results"))
  parser.add_argument("--save_results_interval", type=int, default=10)
  parser.add_argument("--nrow_for_saved_samples", type=int, default=9)
  parser.add_argument("--random_seed", type=int, default=0)
  return parser

def main():
  args = get_argparser().parse_args()
  pad_length = len(str(args.num_epochs))

  torch.manual_seed(args.random_seed)
  np.random.seed(args.random_seed)
  random.seed(args.random_seed)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  print(f"Training on {args.dataset}")
  if args.dataset == "MNIST":
    dataset = torchvision.datasets.MNIST(
      root=args.root,
      train=True,
      transform=transforms.Compose([
        transforms.Resize(args.image_height),
        transforms.CenterCrop(args.image_height),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
      ])
    )
    dataloader = data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory
    )
    img_size = (1, args.image_height, args.image_width)
  elif args.dataset == "FashionMNIST":
    dataset = torchvision.datasets.FashionMNIST(
      root=args.root,
      train=True,
      transform=transforms.Compose([
        transforms.Resize(args.image_height),
        transforms.CenterCrop(args.image_height),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
      ])
    )
    dataloader = data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory
    )
    img_size = (1, args.image_height, args.image_width)
  else:
    raise RuntimeError("something went wrong: dataset was not set!")

  G = Generator(
    img_size=img_size,
    in_features=args.generator_in_features,
    latent_dim=args.latent_dim,
  )
  D = Discriminator(
    img_size=img_size,
    in_features=args.discriminator_in_features,
    use_spectral_norm=args.use_spectral_norm,
  )
  G = G.to(device)
  D = D.to(device)

  if args.compile:
    print("Compiling with Just-In-Time compilation")
    G = torch.compile(G)
    D = torch.compile(D)
  if args.optimizer == "Adam":
    G_optimizer = optim.Adam(G.parameters(), lr=args.generator_lr, betas=(args.beta1, args.beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=args.discriminator_lr, betas=(args.beta1, args.beta2))
  else:
    raise RuntimeError("something went wrong: optimizer was not set!")
  criterion = torch.nn.BCEWithLogitsLoss()

  if not (os.path.exists(args.ckpt_dir) and os.path.isdir(args.ckpt_dir)):
    os.makedirs(args.ckpt_dir, exist_ok=True)

  results_dir = os.path.join(args.results_dir, args.dataset, "epoch")
  if not (os.path.exists(results_dir) and os.path.isdir(results_dir)):
    os.makedirs(results_dir, exist_ok=True)

  G.train()
  D.train()
  test_noise = torch.randn(args.nrow_for_saved_samples ** 2, args.latent_dim, device=device)

  real_label_value = 1.0
  fake_label_value = 0.0
  for epoch in range(args.num_epochs):
    with tqdm(dataloader, desc=f"Training", unit="batch") as pbar:
      running_G_loss = 0.0
      running_D_loss = 0.0
      for images, _ in pbar:
        images = images.to(device)
        batch_size = images.size(0)

        real_labels = torch.full((batch_size, 1), real_label_value, device=device)
        fake_labels = torch.full((batch_size, 1), fake_label_value, device=device)

        # === Train Discriminator ===
        D_optimizer.zero_grad()
        real_loss = criterion(D(images), real_labels)
        noise = torch.randn(batch_size, args.latent_dim, device=device)
        fake_images = G(noise)
        fake_loss = criterion(D(fake_images.detach()), fake_labels)
        D_loss = real_loss + fake_loss
        D_loss.backward()
        D_optimizer.step()

        # === Train Generator ===
        G_optimizer.zero_grad()
        noise = torch.randn(batch_size, args.latent_dim, device=device)
        fake_images = G(noise)
        G_loss = criterion(D(fake_images), real_labels)
        G_loss.backward()
        G_optimizer.step()

        running_G_loss += G_loss.item()
        running_D_loss += D_loss.item()

        pbar.update(1)
        pbar.set_postfix({
          "G_loss": G_loss.item(),
          "D_loss": D_loss.item()
        })
    average_G_loss = running_G_loss / len(dataloader)
    average_D_loss = running_D_loss / len(dataloader)
    print(f"Epoch: {epoch + 1}/{args.num_epochs}")
    print(f"G Loss: {average_G_loss:.5f}, D Loss: {average_D_loss:.5f}")

    if (epoch + 1) % args.save_results_interval == 0:
      print("Saving fake images")
      G.eval()
      with torch.no_grad():
        fake_images = G(test_noise)
        grid = torchvision.utils.make_grid(fake_images, nrow=args.nrow_for_saved_samples, normalize=True)
        torchvision.utils.save_image(
          grid,
          os.path.join(results_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.png")
        )
      G.train()

    if (epoch + 1) % args.save_ckpt_interval == 0:
      print("Saving model checkpoints")
      checkpoint = {
        "dataset": args.dataset,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "generator_in_features": args.generator_in_features,
        "discriminator_in_features": args.discriminator_in_features,
        "use_spectral_norm": args.use_spectral_norm,
        "latent_dim": args.latent_dim,
        "generator_lr": args.generator_lr,
        "discriminator_lr": args.discriminator_lr,
        "G_optimizer": G_optimizer.state_dict(),
        "D_optimizer": D_optimizer.state_dict(),
        "image_height": args.image_height,
        "image_width": args.image_width
      }
      torch.save(checkpoint, os.path.join(args.ckpt_dir, f"{args.dataset}_checkpoint_{epoch + 1:0{pad_length}d}.pth"))
      generator = {
        "dataset": args.dataset,
        "G": G.state_dict(),
        "generator_in_features": args.generator_in_features,
        "latent_dim": args.latent_dim,
        "image_height": args.image_height,
        "image_width": args.image_width
      }
      torch.save(generator, os.path.join(args.ckpt_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.pth"))

  gif_dir = os.path.join(args.results_dir, args.dataset, "gif")
  if not (os.path.exists(gif_dir) and os.path.isdir(gif_dir)):
    os.makedirs(gif_dir, exist_ok=True)

  print("Saving gif")
  output_gif = os.path.join(gif_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.gif")
  png_files = sorted([f for f in os.listdir(results_dir) if f.endswith(".png")])
  images = [Image.open(os.path.join(results_dir, f)) for f in png_files]
  images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=150,
    loop=0
  )

  print("Saving final fake images")
  G.eval()
  with torch.no_grad():
    fake_images = G(test_noise)
    grid = torchvision.utils.make_grid(fake_images, nrow=args.nrow_for_saved_samples, normalize=True)
    torchvision.utils.save_image(
      grid,
      os.path.join(results_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.png")
    )
  G.train()

  print("Saving final model checkpoints")
  checkpoint = {
    "dataset": args.dataset,
    "G": G.state_dict(),
    "D": D.state_dict(),
    "generator_in_features": args.generator_in_features,
    "discriminator_in_features": args.discriminator_in_features,
    "latent_dim": args.latent_dim,
    "generator_lr": args.generator_lr,
    "discriminator_lr": args.discriminator_lr,
    "G_optimizer": G_optimizer.state_dict(),
    "D_optimizer": D_optimizer.state_dict(),
    "image_height": args.image_height,
    "image_width": args.image_width
  }
  torch.save(checkpoint, os.path.join(args.ckpt_dir, f"{args.dataset}_checkpoint_{epoch + 1:0{pad_length}d}.pth"))
  generator = {
    "dataset": args.dataset,
    "G": G.state_dict(),
    "generator_in_features": args.generator_in_features,
    "latent_dim": args.latent_dim,
    "image_height": args.image_height,
    "image_width": args.image_width
  }
  torch.save(generator, os.path.join(args.ckpt_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.pth"))

if __name__ == "__main__":
  main()
