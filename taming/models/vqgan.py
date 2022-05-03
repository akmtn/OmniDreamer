import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder, UNet, ICIP_UNet
from taming.modules.vqvae.quantize import VectorQuantizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class ICIP_NonVQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_config=None,
                 first_stage_key="image",
                 image_key="image",
                 concat_input=False,
                 colorize_nlabels=None,
                 monitor=None,
                 refine_mode=False
                 ):
        super().__init__()
        self.concat_input = concat_input
        #self.encoder = Encoder(**ddconfig) ## We use ICIP_UNet, so don't need Encoder and Decoder
        #self.decoder = Decoder(**ddconfig)
        self.unet = ICIP_UNet(**ddconfig) ## here!
        self.loss = instantiate_from_config(lossconfig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if monitor is not None:
            self.monitor = monitor
        self.refine_mode = refine_mode


    def forward(self, input, mask):

        input_masked = input * mask
        x = self.unet(torch.cat((input_masked, mask), axis=1))

        # post-processing (replacing by the mask)
        x = input_masked * mask + x * (1. - mask)

        return x

    @torch.no_grad()
    def sample(self, input, mask):

        input_masked = input * mask
        x = self.unet(torch.cat((input_masked, mask), axis=1))

        # post-processing (replacing by the mask)
        x = input_masked * mask + x * (1. - mask)

        return x

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.concat_input == False:
            x = self.get_input(batch, self.image_key)
            xrec = self(x)

        elif self.concat_input == True and self.refine_mode == True:
            concat_x = self.get_input(batch, "concat_input")  # concat_x is including a masked_image
            binary_mask = concat_x[:,6:7,:,:]
            x = self.get_input(batch, "image") #reconstruction target is non-masked input image.
            xrec = self(x, binary_mask)

        else:
            concat_x = self.get_input(batch, "concat_input")  # concat_x is including a masked_image
            concat_xrec = self(concat_x)
            x = self.get_input(batch, "image") #reconstruction target is non-masked input image.
            concat_x = torch.cat((x, concat_x[:,3:,:,:]), dim=1)
            #x = concat_x[:,:3,:,:]
            xrec = concat_xrec[:,:3,:,:]

        qloss = torch.zeros(1)

        if optimizer_idx == 0:
            # autoencode (generator)
            if self.concat_input == False:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            elif self.concat_input == True and self.refine_mode == True:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            else:
                aeloss, log_dict_ae = self.loss(qloss, concat_x, concat_xrec, optimizer_idx, self.global_step,
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
        # inference
        if self.concat_input == False:
            x = self.get_input(batch, self.image_key)
            xrec = self(x)

        elif self.concat_input == True and self.refine_mode == True:
            concat_x = self.get_input(batch, "concat_input")  # concat_x is including a masked_image
            binary_mask = concat_x[:,6:7,:,:]
            x = self.get_input(batch, "image") #reconstruction target is non-masked input image.
            xrec = self(x, binary_mask)

        else:
            concat_x = self.get_input(batch, "concat_input")  # concat_x 
            concat_xrec = self(concat_x)
            x = concat_x[:,:3,:,:]
            xrec = concat_xrec[:,:3,:,:]
        # loss
        qloss = torch.zeros(1)
        if self.concat_input == False:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        elif self.concat_input == True and self.refine_mode == True:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        else:
            aeloss, log_dict_ae = self.loss(qloss, concat_x, concat_xrec, 0, self.global_step,
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
        opt_ae = torch.optim.Adam(list(self.unet.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.unet.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        if self.concat_input == False:
            x = self.get_input(batch, self.image_key)
            x = x.to(self.device)
            xrec = self(x)
        elif self.concat_input == True and self.refine_mode == True:
            concat_x = self.get_input(batch, "concat_input")  # concat_x is including a masked_image
            binary_mask = concat_x[:,6:7,:,:].to(self.device)
            x = self.get_input(batch, "image") #reconstruction target is non-masked input image.
            x = x.to(self.device)
            xrec = self(x, binary_mask)
        else:
            concat_x = self.get_input(batch, "concat_input")  # concat_x 
            concat_x = concat_x.to(self.device)
            concat_xrec = self(concat_x)
            x = concat_x[:,:3,:,:]
            xrec = concat_xrec[:,:3,:,:]
            # if you wanna save outputs below, remove comment-out.
            #log["coord_rec"] = concat_xrec[:,3:6,:,:]
            #log["binary_mask_rec"] = concat_xrec[:,6:,:,:]
            #log["coord"] = self.get_input(batch, "coord")
            #log["binary_mask"] = self.get_input(batch, "binary_mask")
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class NonVQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_config=None,
                 first_stage_key="image",
                 image_key="image",
                 concat_input=False,
                 colorize_nlabels=None,
                 monitor=None,
                 refine_mode=False
                 ):
        super().__init__()
        self.init_first_stage_from_ckpt(first_stage_config)
        self.first_stage_key = first_stage_key
        self.concat_input = concat_input
        #self.encoder = Encoder(**ddconfig) ## We use UNet, so don't need Encoder and Decoder
        #self.decoder = Decoder(**ddconfig)
        self.unet = UNet(**ddconfig) ## here!
        self.loss = instantiate_from_config(lossconfig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        #self.init_first_stage_from_ckpt(first_stage_config)
        self.image_key = image_key
        if monitor is not None:
            self.monitor = monitor
        self.refine_mode = refine_mode

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    @torch.no_grad()
    def forward_first_stage_model(self, x):
        return self.first_stage_model(x)

    @torch.no_grad()
    def add_noise(self, x):
        # x: input images. [bn, ch=3, h=256, w=256]
        # random sample
        rgb_shift = (torch.rand(x.shape[0],3,1,1, device=x.device) - 0.5) * 0.1 #[-0.05, 0.05]
        # add shift
        x = x + rgb_shift
        # clip to [-1,1]
        return x.clamp(min=-1., max=1.)

    def forward(self, input, mask):
        # First: 2nd vqgan inference
        # downsizing
        x = F.interpolate(input, size=(128,128), mode='area')
        #x = F.interpolate(x, size=(256,256), mode='bilinear')
        # add noise
        x = self.add_noise(x)
        # inference (reconstruction)
        x, _ = self.forward_first_stage_model(x)
        # upscaling
        x = F.interpolate(x, size=(input.shape[2], input.shape[3]), mode='bilinear')

        # Second: Refine Network
        # pre-processing (making holes by the mask)
        input_masked = input * mask

        # refeine 
        x = self.unet(torch.cat((input_masked, x, mask), axis=1)) 

        # post-processing (replacing by the mask)
        x = input_masked * mask + x * (1. - mask)

        return x

    @torch.no_grad()
    def sample(self, input, first_stage_out, mask):

        # input has been aldready masked
        x = F.interpolate(first_stage_out, size=(input.shape[2], input.shape[3]), mode='bilinear')
        # Second: Refine Network
        # refeine 
        input_masked = input * mask
        x = self.unet(torch.cat((input_masked, x, mask), axis=1)) 

        # post-processing (replacing by the mask)
        x = input_masked * mask + x * (1. - mask)

        return x

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.concat_input == False:
            x = self.get_input(batch, self.image_key)
            xrec = self(x)

        elif self.concat_input == True and self.refine_mode == True:
            concat_x = self.get_input(batch, "concat_input")  # concat_x is including a masked_image
            binary_mask = concat_x[:,6:7,:,:]
            x = self.get_input(batch, "image") #reconstruction target is non-masked input image.
            xrec = self(x, binary_mask)

        else:
            concat_x = self.get_input(batch, "concat_input")  # concat_x is including a masked_image
            concat_xrec = self(concat_x)
            x = self.get_input(batch, "image") #reconstruction target is non-masked input image.
            concat_x = torch.cat((x, concat_x[:,3:,:,:]), dim=1)
            #x = concat_x[:,:3,:,:]
            xrec = concat_xrec[:,:3,:,:]

        qloss = torch.zeros(1)

        if optimizer_idx == 0:
            # autoencode (generator)
            if self.concat_input == False:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            elif self.concat_input == True and self.refine_mode == True:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            else:
                aeloss, log_dict_ae = self.loss(qloss, concat_x, concat_xrec, optimizer_idx, self.global_step,
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
        # inference
        if self.concat_input == False:
            x = self.get_input(batch, self.image_key)
            xrec = self(x)

        elif self.concat_input == True and self.refine_mode == True:
            concat_x = self.get_input(batch, "concat_input")  # concat_x is including a masked_image
            binary_mask = concat_x[:,6:7,:,:]
            x = self.get_input(batch, "image") #reconstruction target is non-masked input image.
            xrec = self(x, binary_mask)

        else:
            concat_x = self.get_input(batch, "concat_input")  # concat_x 
            concat_xrec = self(concat_x)
            x = concat_x[:,:3,:,:]
            xrec = concat_xrec[:,:3,:,:]
        # loss
        qloss = torch.zeros(1)
        if self.concat_input == False:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        elif self.concat_input == True and self.refine_mode == True:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        else:
            aeloss, log_dict_ae = self.loss(qloss, concat_x, concat_xrec, 0, self.global_step,
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
        opt_ae = torch.optim.Adam(list(self.unet.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.unet.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        if self.concat_input == False:
            x = self.get_input(batch, self.image_key)
            x = x.to(self.device)
            xrec = self(x)
        elif self.concat_input == True and self.refine_mode == True:
            concat_x = self.get_input(batch, "concat_input")  # concat_x is including a masked_image
            binary_mask = concat_x[:,6:7,:,:].to(self.device)
            x = self.get_input(batch, "image") #reconstruction target is non-masked input image.
            x = x.to(self.device)
            xrec = self(x, binary_mask)
        else:
            concat_x = self.get_input(batch, "concat_input")  # concat_x 
            concat_x = concat_x.to(self.device)
            concat_xrec = self(concat_x)
            x = concat_x[:,:3,:,:]
            xrec = concat_xrec[:,:3,:,:]
            #log["coord_rec"] = concat_xrec[:,3:6,:,:]
            #log["binary_mask_rec"] = concat_xrec[:,6:,:,:]
            #log["coord"] = self.get_input(batch, "coord")
            #log["binary_mask"] = self.get_input(batch, "binary_mask")
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 concat_input=False,
                 completion=False,
                 colorize_nlabels=None,
                 monitor=None
                 ):
        super().__init__()
        self.image_key = image_key
        self.concat_input = concat_input
        self.completion = completion
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
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
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.concat_input == False:
            x = self.get_input(batch, self.image_key)
            xrec, qloss = self(x)
        else:
            concat_x = self.get_input(batch, "concat_input")  # concat_x is including a masked_image
            concat_xrec, qloss = self(concat_x)
            if self.completion == True:
                x = self.get_input(batch, "image") #reconstruction target is non-masked input image.
                concat_x = torch.cat((x, concat_x[:,3:,:,:]), dim=1)
            else:
                x = concat_x[:,:3,:,:]
            xrec = concat_xrec[:,:3,:,:]

        if optimizer_idx == 0:
            # autoencode (generator)
            if self.concat_input == False:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            else:
                aeloss, log_dict_ae = self.loss(qloss, concat_x, concat_xrec, optimizer_idx, self.global_step,
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
        # inference
        if self.concat_input == False:
            x = self.get_input(batch, self.image_key)
            xrec, qloss = self(x)
        else:
            concat_x = self.get_input(batch, "concat_input")  # concat_x 
            concat_xrec, qloss = self(concat_x)
            x = concat_x[:,:3,:,:]
            xrec = concat_xrec[:,:3,:,:]
        # loss
        if self.concat_input == False:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        else:
            aeloss, log_dict_ae = self.loss(qloss, concat_x, concat_xrec, 0, self.global_step,
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

    def log_images(self, batch, **kwargs):
        log = dict()
        if self.concat_input == False:
            x = self.get_input(batch, self.image_key)
            x = x.to(self.device)
            xrec, _ = self(x)
        else:
            concat_x = self.get_input(batch, "concat_input")  # concat_x 
            concat_x = concat_x.to(self.device)
            concat_xrec, _ = self(concat_x)
            x = concat_x[:,:3,:,:]
            xrec = concat_xrec[:,:3,:,:]
            # if you wanna save outputs below, remove comment-out.
            #log["coord_rec"] = concat_xrec[:,3:6,:,:]
            #log["binary_mask_rec"] = concat_xrec[:,6:,:,:]
            #log["coord"] = self.get_input(batch, "coord")
            #log["binary_mask"] = self.get_input(batch, "binary_mask")
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
