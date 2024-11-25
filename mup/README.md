# Mu-Transfer for Mamba


## Messy Notes

### Config

`mamba-ssm` config options:
* Providing `d_intermediate != 0` inserts an MLP layer, otherwise a trivial `nn.Identity` is used
* Specifying `attn_layer_idx` puts a traditional transformer block at that layer.

