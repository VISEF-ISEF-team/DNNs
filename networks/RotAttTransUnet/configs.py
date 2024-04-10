import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.width=128
    config.height=128
    config.patch_size = 16
    config.d_f = 768
    config.d_k = 768
    config.d_v = 768
    
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 8
    config.activation = 'softmax'
    return config