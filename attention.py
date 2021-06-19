import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import transforms
import cv2

activation = {}


def get_attn_softmax(name):
    def hook(model, input, output):
        with torch.no_grad():
            input = input[0]
            B, N, C = input.shape
            qkv = (
                model.qkv(input)
                .detach()
                .reshape(B, N, 3, model.num_heads, C // model.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * model.scale
            attn = attn.softmax(dim=-1)
            activation[name] = attn

    return hook


# expects timm vis transformer model
def add_attn_vis_hook(model):
    for idx, module in enumerate(list(model.blocks.children())):
        module.attn.register_forward_hook(get_attn_softmax(f"attn{idx}"))



def get_mask(im,att_mat):
    # Average the attention weights across all heads.
    # att_mat,_ = torch.max(att_mat, dim=1)
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")
    return result, joint_attentions, grid_size

def show_attention_map(model, img_path, shape):
    add_attn_vis_hook(model)
    im = Image.open(os.path.expandvars(img_path))
    im = im.resize((shape, shape))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    logits = model(transform(im).unsqueeze(0))

    attn_weights_list = list(activation.values())

    result, joint_attentions, grid_size = get_mask(im,torch.cat(attn_weights_list))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)

    probs = torch.nn.Softmax(dim=-1)(logits)
    top5 = torch.argsort(probs, dim=-1, descending=True)
    print("Prediction Label and Attention Map!\n")
    for idx in top5[0, :5]:
        print(f'{probs[0, idx.item()]:.5f} : {idx.item()}', end='')
    

    for i, v in enumerate(joint_attentions):
        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
        result = (mask * im).astype("uint8")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map_%d Layer' % (i+1))
        _ = ax1.imshow(im)
        _ = ax2.imshow(result)
    
    plt.show()

if __name__ == "__main__":
    import os
    import sys
    import timm

    model_names = timm.list_models("vit*")
    for model_name in model_names:
        print(f"\n{model_name}\n")
        m = timm.create_model(model_name, pretrained=True)

        shape = eval(model_name[-3:])

        show_attention_map(m, sys.argv[1], shape)