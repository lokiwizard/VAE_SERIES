import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from dataset_utils.animal_utils.animals import get_animal_data_loader
from Simple_VQGAN.Taming.modules.diffusionmodules.model import Encoder, Decoder
from Simple_VQGAN.Taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from Simple_VQGAN.Taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from pytorch_lightning import Trainer

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np



class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.learning_rate = 1e-4
        self.image_key = image_key
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):

        quant_b = self.quantize.get_codebook_entry(code_b.view(-1), (-1, code_b.size(1), code_b.size(2), self.embed_dim))
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch[0]
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        #x = self.get_input(batch, self.image_key)
        x = batch[0]
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def reconstruction_visualization(self, original_images, recon_images, device):

        original_images = original_images * torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        original_images = original_images + torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        recon_images = recon_images * torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        recon_images = recon_images + torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        original_images = original_images.clamp(0, 1)
        recon_images= recon_images.clamp(0, 1)

        # 可视化
        n = 4
        plt.figure(figsize=(16, 4))
        for i in range(n):
            # 原始图像
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(np.transpose(original_images[i].detach().cpu().numpy(), (1, 2, 0)))
            plt.title("Original")
            plt.axis('off')

            # 重建图像
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(np.transpose(recon_images[i].detach().cpu().numpy(), (1, 2, 0)))
            plt.title("Reconstructed")
            plt.axis('off')
        plt.show()

    def on_train_epoch_end(self, *args, **kwargs):
        # 确保有一个验证数据加载器或测试数据加载器
        # 从验证集或测试集中抽取一些图像进行重建和可视化

        # 禁用梯度计算以提高效率
        with torch.no_grad():
            # 从验证数据加载器中获取一个批次的图像
            # 这里假设你有一个验证数据加载器 val_dataloader
            # 如果是 PyTorch Lightning，可以通过 self.trainer.val_dataloaders 获取
            batch = next(iter(self.trainer.val_dataloaders[0]))

            # 从批次中获取图像
            x = batch[0].to(self.device)

            # 重建图像
            xrec, _ = self(x)

            # 使用已经存在的 reconstruction_visualization 方法进行可视化
            self.reconstruction_visualization(x, xrec, x.device)


"""
if __name__ == "__main__":
    config = OmegaConf.load(r"D:\pyproject\representation_learning_models\Simple_VQGAN\model.yaml")
    ddconfig = config.model.get("params").get("ddconfig")
    lossconfig = config.model.params.get("lossconfig").get("params")

    device = "cuda"

    model = VQModel(ddconfig=ddconfig,lossconfig=lossconfig, n_embed=1024, embed_dim=256)
    #model.init_from_ckpt(r"D:\pyproject\representation_learning_models\Simple_VQGAN\last.ckpt")
    model.to("cuda")
    train_loader, val_loader, num_classes = get_animal_data_loader(root_dir=r"D:\pyproject\representation_learning_models\dataset_utils\animals",
                                                                   image_size=128, batch_size=16, use_data_augmentation=False)


    trainer = Trainer(max_epochs=10, gpus=1, precision=16)
    trainer.fit(model, train_loader, val_loader)
"""



