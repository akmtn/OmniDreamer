import yaml
import os
import numpy as np
import cv2
import torch
from omegaconf import OmegaConf
from PIL import Image
import cv2
import argparse 
import glob
import tqdm


def show_image(s, save_path):
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    if s.shape[2] == 1:
        s = np.tile(s, (1,1,3))
    elif s.shape[2] > 3:
        #print("condition_img is rounded to 3 ch")
        s = s[:,:,:3]
    s = Image.fromarray(s)
    s.save(save_path)

def masking(im, mask_path=None):
    h, w, c = im.shape
    if mask_path is not None:
      binary_mask = cv2.imread(mask_path)
      binary_mask = cv2.resize(binary_mask, (w,h), interpolation=cv2.INTER_AREA)[:,:,0:1]
      binary_mask[binary_mask>0] = 1. 
    else:
      binary_mask = np.zeros(( h,w,1))
      degree_h = 90
      degree_w = 180
      #binary_mask[int(h*45/180):int(h*135/180), int(w*135/360):int(w*225/360), :] = 1.
      binary_mask[int(h*(90-degree_h/2)/180):int(h*(90+degree_h/2)/180), int(w*(180-degree_w/2)/360):int(w*(180+degree_w/2)/360), :] = 1.
    canvas = np.ones_like(im) * 127.5
    canvas = im * binary_mask + canvas * (1 - binary_mask)
    return canvas, binary_mask

def add_cylinderical(coord):
    h, w, _ = coord.shape
    sin_img = np.sin(np.radians(np.arange(w) / w * 360))
    sin_img[np.abs(sin_img) < 1e-6] = 0
    sin_img = np.tile(sin_img, (h,1))[:,:,np.newaxis]
    cos_img = np.cos(np.radians(np.arange(w) / w * 360))
    cos_img[np.abs(cos_img) < 1e-6] = 0
    cos_img = np.tile(cos_img, (h,1))[:,:,np.newaxis]
    return np.concatenate((coord, sin_img, cos_img), axis=2)

def rotation_augmentation( im, coord, masked_im, binary_mask, random_deg=True):
    if random_deg:
        plit_point = np.random.randint(0, im.shape[1]) # w
    else:
        split_point = im.shape[1] // 2
    im = np.concatenate( (im[:,split_point:,:], im[:,:split_point,:]), axis=1 )
    coord = np.concatenate( (coord[:,split_point:,:], coord[:,:split_point,:]), axis=1 )
    masked_im = np.concatenate( (masked_im[:,split_point:,:], masked_im[:,:split_point,:]), axis=1 )
    binary_mask = np.concatenate( (binary_mask[:,split_point:,:], binary_mask[:,:split_point,:]), axis=1 )
    return im, coord, masked_im, binary_mask



def inference(args):

    # load test folder
    testdir_path = glob.glob(os.path.join(args.testdir, "*"))
    testdir_path.sort()
    print(f"{len(testdir_path)} images for test...")

    # first stage (completion with CompNets: VQGAN_1's Encoder + Transformer + VQGAN_2's Decoder)
    config = OmegaConf.load(args.config_path)
    print(yaml.dump(OmegaConf.to_container(config)))

    from taming.models.cond_transformer import Net2NetTransformer
    model = Net2NetTransformer(**config.model.params)
    sd = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.cuda().eval()

    # second stage (AdjustmentNet)
    if True:
        config_2 = OmegaConf.load(args.config_path_2)
        print(yaml.dump(OmegaConf.to_container(config_2)))

        from taming.models.vqgan import NonVQModel
        model_2 = NonVQModel(**config_2.model.params)
        sd_2 = torch.load(args.ckpt_path_2, map_location="cpu")["state_dict"]
        missing_2, unexpected_2 = model_2.load_state_dict(sd_2, strict=False)
        model_2.cuda().eval()
    torch.set_grad_enabled(False)



    for num, condition_img_path in tqdm.tqdm(enumerate(testdir_path)):

        if args.idxrange is not None:
            if ((args.idxrange - 1) * 1000) <= num and num < (args.idxrange * 1000):
                pass 
            else:
                continue
        elif args.imageidx is not None:
            if num < args.imageidx:
                continue
            elif num > args.imageidx:
                break # for debug
        elif args.imageidx is None or num == args.imageidx:
            for kloop in range(10): # How many outputs?
                print(f"Running #{kloop} of {condition_img_path}")

                ## Loading & Pre-processing
                # condition_img
                input_image = Image.open(condition_img_path)
                image = np.array(input_image).astype(np.uint8)

                # masking
                condition_img, binary_mask = masking(image, mask_path=args.mask_path)
                condition_img = (condition_img / 127.5 - 1).astype(np.float32)

                # resize for CompNets' fixed size
                if image.shape[1] != 512 or image.shape[0] != 256:
                    image = cv2.resize(image, (512,256), interpolation=cv2.INTER_AREA)
                    condition_img = cv2.resize(condition_img, (512,256), interpolation=cv2.INTER_AREA)
                    binary_mask = cv2.resize(binary_mask, (512,256), interpolation=cv2.INTER_AREA)[:,:,np.newaxis]
                    binary_mask[binary_mask>0] = 1.

                # preprocessing
                h,w,_ = condition_img.shape
                coord = np.tile(np.arange(h).reshape(h,1,1), (1,w,1)) / (h-1) * 2 - 1 # -1~ 1
                coord = add_cylinderical(coord) # sin, cos

                #rotation
                image, coord, condition_img, binary_mask = rotation_augmentation(image, coord, condition_img, binary_mask, random_deg=False)

                # concat_input
                condition_img = np.concatenate((condition_img, coord, binary_mask), axis=2)

                # input to completion stage
                condition_img = torch.tensor(condition_img.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)
                #show_image(condition_img, "outputs/condition_img_%08d.png" % num) # if you wanna check the input image, remove comment-out.


                ## Inference
                c_code, c_indices = model.encode_to_c(condition_img)
                assert c_code.shape[2]*c_code.shape[3] == c_indices.shape[1]
                condition_rec = model.cond_stage_model.decode(c_code)
                #show_image(condition_rec, os.path.join(args.outdir, "condition_rec_%08d.png" % num)) # if you wanna check a reconstraction of the input image, remove comment-out.



                codebook_size = config.model.params.first_stage_config.params.embed_dim
                z_indices_shape = [1,int(h*w/256)] if condition_img_path is None else c_indices.shape 
                z_code_shape = [1,256,int(h/16),int(w/16)] if condition_img_path is None else c_code.shape 
                z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)
                x_sample = model.decode_to_img(z_indices, z_code_shape)
                #show_image(x_sample, "outputs/random_init_%08d.png" % num)

                print("Running the first stage (completion)")

                import time

                idx = z_indices
                idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3])

                cidx = c_indices
                cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])

                temperature = 1.0
                top_k = 100
                update_every = 50

                start_t = time.time()

                ## Please change for circular inference of transformer.
                circular_type = 1 # if 0, no circular inference, just raster scan
                px = 8 # Adjust according an input resion.  eg. 90deg:4 180deg:8
                ##


                if circular_type == 1:
                    idx = torch.cat((idx[:,:,-px:], idx, idx[:,:,:px]), 2)
                    z_code_shape = [idx.shape[0], 256, idx.shape[1], idx.shape[2]]
                    cidx = torch.cat((cidx[:,:,-px:], cidx, cidx[:,:,:px]), 2)
                    c_code_shape = [cidx.shape[0], 256, cidx.shape[1], cidx.shape[2]]

                for i in range(0, z_code_shape[2]-0):
                    if i <= 8:
                        local_i = i
                    elif z_code_shape[2]-i < 8:
                        local_i = 16-(z_code_shape[2]-i)
                    else:
                        local_i = 8
                    for j in range(0,z_code_shape[3]-0):
                        if j <= 16:
                            local_j = j
                        elif z_code_shape[3]-j < 16:
                            local_j = 32-(z_code_shape[3]-j)
                        else:
                            local_j = 16

                        i_start = i-local_i
                        i_end = i_start+16
                        j_start = j-local_j
                        j_end = j_start+32
                        
                        patch = idx[:,i_start:i_end,j_start:j_end]
                        patch = patch.reshape(patch.shape[0],-1)
                        cpatch = cidx[:, i_start:i_end, j_start:j_end]
                        cpatch = cpatch.reshape(cpatch.shape[0], -1)
                        patch = torch.cat((cpatch, patch), dim=1)
                        logits,_ = model.transformer(patch[:,:-1])
                        logits = logits[:, -512:, :]
                        logits = logits.reshape(z_code_shape[0],16,32,-1)
                        logits = logits[:,local_i,local_j,:]

                        logits = logits/temperature

                        if top_k is not None:
                            logits = model.top_k_logits(logits, top_k)

                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        idx[:,i,j] = torch.multinomial(probs, num_samples=1)

                        step = i*z_code_shape[3]+j
                        if step%update_every==0 or step==z_code_shape[2]*z_code_shape[3]-1:
                            pass
                
                # copy to other side
                if circular_type ==1:
                    idx[:,:,:px] = idx[:,:,-2*px:-px]
                    idx[:,:,-px:] = idx[:,:,px:2*px] 

                #idx = torch.cat((idx[:,:,16:], idx[:,:,:16]),2) #lr_replace

                x_sample = model.decode_to_img(idx, z_code_shape)
                #print(f"Time: {time.time() - start_t} seconds")
                #print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
                #show_image(x_sample, os.path.join(args.outdir, "sample_%08d.png" % num))

                # drop both edges
                if circular_type == 1:
                    idx = idx[:,:,px:-px]
                    z_code_shape = [z_code_shape[0], z_code_shape[1], z_code_shape[2], idx.shape[2]]
                    x_sample = x_sample[:,:,:,16*px:-16*px]
                x_sample = torch.cat((x_sample[:,:,:,x_sample.shape[3]//2:], x_sample[:,:,:,:x_sample.shape[3]//2]), 3)
                show_image(x_sample, os.path.join(args.outdir, "sample_%08d_%02d.png" % (num,kloop)))

                if False:
                    # if adjustment stage is not needed, true
                    continue


                print("Running the second stage (adjustment)")
                # condition_img
                image = np.array(input_image).astype(np.uint8)
                output_size = (args.output_size_w, args.output_size_h) # eg. (1024, 512)
                if image.shape[1] != output_size[0] or image.shape[0] != output_size[1]:
                    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
                
                # masking
                condition_img, binary_mask = masking(image, mask_path=args.mask_path)
                condition_img = (condition_img / 127.5 - 1).astype(np.float32)
                image = (image / 127.5 - 1).astype(np.float32)

                # preprocessing
                h,w,_ = condition_img.shape
                coord = np.tile(np.arange(h).reshape(h,1,1), (1,w,1)) / (h-1) * 2 - 1 # -1~ 1
                coord = add_cylinderical(coord) # sin, cos


                condition_img = torch.tensor(condition_img.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)
                binary_mask = torch.tensor(binary_mask.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)
                condition_rec = model_2.sample(condition_img, x_sample, binary_mask)
                show_image(condition_rec, os.path.join(args.outdir, "refined_%08d_%02d.png" % (num,kloop)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path')
    parser.add_argument('--ckpt_path')
    parser.add_argument('--config_path_2')
    parser.add_argument('--ckpt_path_2')
    parser.add_argument('--output_size_w', '-ow', type=int, default=1024)
    parser.add_argument('--output_size_h', '-oh', type=int, default=512)
    parser.add_argument('--testdir', default="assets/test/")
    parser.add_argument('--outdir')
    parser.add_argument('--mask_path', default=None)
    parser.add_argument('--idxrange', '-ir', type=int, default=None)
    parser.add_argument('--imageidx', type=int, default=None)
    

    args = parser.parse_args()

    assert os.path.exists(args.outdir), "Create outdir, before running"

    inference(args)

