# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import List, Tuple, Dict, Any, Optional
from dataset_utils.animal_utils.animals import get_animal_data_loader
import PIL
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from SimpleViTVQGan.layers import ViTEncoder as Encoder, ViTDecoder as Decoder
from SimpleViTVQGan.quantizers import VectorQuantizer, GumbelQuantizer
from SimpleViTVQGan.losses.vqperceptual import VQLPIPSWithDiscriminator
from omegaconf import OmegaConf
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class ViTVQ(pl.LightningModule):
    def __init__(self, image_size: int, patch_size: int, encoder: OmegaConf, decoder: OmegaConf,
                 quantizer: OmegaConf,
                 loss: OmegaConf, path: Optional[str] = None, ignore_keys: List[str] = list(),
                 scheduler: Optional[OmegaConf] = None) -> None:
        super().__init__()
        self.path = path  # checkpoint path
        self.ignore_keys = ignore_keys

        self.loss = VQLPIPSWithDiscriminator(**loss)
        self.encoder = Encoder(image_size=image_size, patch_size=patch_size, **encoder)
        self.decoder = Decoder(image_size=image_size, patch_size=patch_size, **decoder)
        self.quantizer = VectorQuantizer(**quantizer)
        self.pre_quant = nn.Linear(encoder.dim, quantizer.embed_dim)
        self.post_quant = nn.Linear(quantizer.embed_dim, decoder.dim)

        # 逆标准化参数 (基于 ImageNet 的常用均值和标准差，可根据数据集调整)
        self.mean = torch.tensor([0.485, 0.456, 0.406])  # 替换为你的数据集的均值
        self.std = torch.tensor([0.229, 0.224, 0.225])  # 替换为你的数据集的标准差

        if path is not None:
            self.init_from_ckpt(path, ignore_keys)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        quant, diff = self.encode(x)
        dec = self.decode(quant)

        return dec, diff

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, _ = self.quantizer(h)

        return quant, emb_loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant)

        return dec

    def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        h = self.encoder(x)
        h = self.pre_quant(h)
        _, _, codes = self.quantizer(h)

        return codes

    def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
        quant = self.quantizer.embedding(code)
        quant = self.quantizer.norm(quant)

        if self.quantizer.use_residual:
            quant = quant.sum(-2)

        dec = self.decode(quant)

        return dec


    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        x = batch[0]
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencoder
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.decoder.get_last_layer(), split="train")

            self.log("train/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            del log_dict_ae["train/total_loss"]

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.decoder.get_last_layer(), split="train")

            self.log("train/disc_loss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            del log_dict_disc["train/disc_loss"]

            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            return discloss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = batch[0]
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.decoder.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/rec_loss"]

        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        del log_dict_ae["val/total_loss"]

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        if hasattr(self.loss, 'discriminator'):
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                                last_layer=self.decoder.get_last_layer(), split="val")

            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        # 随机选取 4 张图片进行可视化
        if batch_idx == 0:
            self.log_reconstructed_images(x, xrec)

        return self.log_dict


    def configure_optimizers(self) -> Tuple[List, List]:
        lr = 1e-4
        optim_groups = list(self.encoder.parameters()) + \
                       list(self.decoder.parameters()) + \
                       list(self.pre_quant.parameters()) + \
                       list(self.post_quant.parameters()) + \
                       list(self.quantizer.parameters())

        optimizers = [torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)]

        if hasattr(self.loss, 'discriminator'):
            optimizers.append(
                torch.optim.AdamW(self.loss.discriminator.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4))


        return optimizers

    def denormalize(self, tensor):
        """将标准化的图像逆标准化为原始图像范围"""
        mean = self.mean.to(tensor.device).view(1, -1, 1, 1)
        std = self.std.to(tensor.device).view(1, -1, 1, 1)
        return tensor * std + mean

    def log_reconstructed_images(self, x, x_recon):
        # 随机选取 4 张图片
        indices = random.sample(range(x.size(0)), min(4, x.size(0)))
        original_images = x[indices]
        reconstructed_images = x_recon[indices]

        # 逆标准化
        original_images = self.denormalize(original_images)
        reconstructed_images = self.denormalize(reconstructed_images)

        # 拼接图片
        grid = torch.cat((original_images, reconstructed_images), dim=0)
        grid = make_grid(grid, nrow=4, normalize=True, scale_each=True)

        # 显示图片
        plt.figure(figsize=(10, 5))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Top: Original, Bottom: Reconstructed")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    config_path = r"/SimpleViTVQGan\configs\imagenet_vitvq_small.yaml"
    config = OmegaConf.load(config_path)
    encoder_config = config.model.params.encoder
    decoder_config = config.model.params.decoder
    quantizer_config = config.model.params.quantizer
    image_size = config.model.params.image_size
    patch_size = config.model.params.patch_size
    loss_config = config.model.params.loss.params

    vitvqgan = ViTVQ(image_size=image_size, patch_size=patch_size, encoder=encoder_config, decoder=decoder_config,
                        quantizer=quantizer_config, loss=loss_config)

    train_loader, val_loader, num_classes = get_animal_data_loader(
        root_dir=r"D:\pyproject\representation_learning_models\dataset_utils\animals",
        image_size=224, batch_size=8, use_data_augmentation=False)

    trainer = Trainer(max_epochs=10, gpus=1, precision=16)
    trainer.fit(vitvqgan, train_loader, val_loader)



