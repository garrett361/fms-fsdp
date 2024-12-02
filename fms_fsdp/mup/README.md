# Mu-Transfer for Mamba

* `mup/mup_mamba.py`:  mup implementation for `mamba-ssm`
* `mup/coord_check.py`:  utils for the mup ["coord check"](https://github.com/microsoft/mup?tab=readme-ov-file#coord-check), for sanity checking mup impl correctness
* `mup/transformer_only_coord_check.py`:  script for performing the coord check on a transformer-only model


Other files are scratchwork.


## Messy Notes

### Config

`mamba-ssm` config options:
* Providing `d_intermediate != 0` inserts an MLP layer, otherwise a trivial `nn.Identity` is used
* Specifying `attn_layer_idx` puts a traditional transformer block at that layer, rather than a mamba block.
* A vanilla transformer then comes from setting `d_intermediate!=0` and specifying all layerss in `attn_layer_idx`.



### Mup Implementation


*  Impl detail: for hidden `Linear` layers, should the mup-adjusted learning rates be literally `~ lr /in_features`, different and customized for each `Linear`,  or do we perform a more global rescaling `~lr / width`?
