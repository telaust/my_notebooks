import torch
from dataset import HorseZebraDataset
import sys
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import config
from utils import *

import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

def train(d_H, d_Z, g_Z, g_H, loader, opt_D, opt_G, l1, mse, d_scaler, g_scaler):
    H_reals, H_fakes= 0, 0

    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        with torch.cuda.amp.autocast():

            # HORSE
            fake_horse = g_H(zebra)

            D_horse_real = d_H(horse)
            D_horse_fake = d_H(fake_horse.detach())

            H_reals += D_horse_real.mean().item()
            H_fakes += D_horse_fake.mean().item()

            D_real_horse_loss = mse(D_horse_real, torch.ones_like(D_horse_real))
            D_fake_horse_loss = mse(D_horse_fake, torch.ones_like(D_horse_fake))
            D_horse_loss = D_real_horse_loss + D_fake_horse_loss

            # ZEBRA

            fake_zebra = g_H(horse)
            D_zebra_real = d_H(zebra)
            D_zebra_fake = d_H(fake_zebra.detach())


            D_real_zebra_loss = mse(D_zebra_real, torch.ones_like(D_zebra_real))
            D_fake_zebra_loss = mse(D_zebra_fake, torch.ones_like(D_zebra_fake))
            D_zebra_loss = D_zebra_real + D_zebra_fake

            D_loss = (D_zebra_loss + D_horse_loss) / 2

        opt_D.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_D)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_horse_fake = d_H(fake_horse)
            D_zera_fake = d_H(fake_zebra)

            loss_G_horse = mse(D_horse_fake, torch.ones_like(D_horse_fake))
            loss_G_zebra = mse(D_zera_fake, torch.ones_like(D_zera_fake))

            # cycle loss

            cycle_zebra = g_Z(fake_horse)
            cycle_horse = g_H(fake_zebra)

            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            identity_zebra = g_Z(zebra)
            identity_horse = g_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            G_loss = (
                loss_G_zebra
                + loss_G_horse
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse * config.LAMBDA_IDENTITY
                + identity_zebra * config.LAMBDA_IDENTITY
            )

        opt_G.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_G)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():

    d_H = Discriminator(in_channels=3).to(config.DEVICE)
    d_Z = Discriminator(in_channels=3).to(config.DEVICE)

    g_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    g_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_D = optim.Adam(
        list(d_H.parameters()) + list(d_Z.parameters()),
        lr = config.LEARNING_RATE,
        betas= (0.5, 0.999)
    )

    opt_G = optim.Adam(
        list(g_H.parameters()) + list(g_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, g_H, opt_G, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, g_Z, opt_G, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, d_H, opt_D, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, d_Z, opt_D, config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "/horses", root_zebra=config.TRAIN_DIR + "/zebras",
        transform=config.transforms
    )

    val_dataset = HorseZebraDataset(
        root_horse="cyclegan_test/horse1", root_zebra="cyclegan_test/zebra1",
        transform=config.transforms
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader \
        = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for ep in range(config.NUM_EPOCHS):
        train(d_H, d_Z, g_Z, g_H, loader, opt_D, opt_G, l1=L1,
              mse=mse, d_scaler=d_scaler, g_scaler=g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(g_H, opt_G, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(g_Z, opt_G, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(d_H, opt_D, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(d_Z, opt_D, filename=config.CHECKPOINT_CRITIC_Z)



if __name__ == "__main__":
    main()



