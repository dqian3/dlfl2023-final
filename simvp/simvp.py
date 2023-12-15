'''
Code taken from
https://www.kaggle.com/code/simuzilisen/simvp
'''

from .modules import *

class SimVP_Model(nn.Module):
    def __init__(self, in_shape, hid_S=64, hid_T=512, N_S=4, N_T=8, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec)

        # model_type = 'gsta' if model_type is None else model_type.lower()
        # if model_type == 'incepu':
        #     self.hid = MidIncepNet(T*hid_S, hid_T, N_T)
        # else:
        self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
            input_resolution=(H, W), model_type=model_type,
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    
        self.hid_S = hid_S
        self.in_shape = in_shape
        self.out_shape = in_shape
        
    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)

        T_out = self.out_shape[0]
        # print(hid.shape)
        # print(skip.shape)

        hid = hid.reshape(B*T_out, -1, H_, W_)
        skip = skip.reshape(B*T_out, -1, H, W)

        # print(hid.shape)
        # print(skip.shape)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, *self.out_shape)

        return Y
