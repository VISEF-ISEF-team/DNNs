import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Input
    config.num_classes = 8
    config.in_channels = 1
    config.width = 256
    config.height = 256
    config.d_f = 768
    config.d_k = 768
    config.d_v = 768
    config.d_q = 768
    
    # Transformer
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 6
    config.transformer.att_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    
    # Resnet hybid model
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    
    # Decoder
    config.classifier = 'seg'
    config.n_skip = 3
    config.skip_channels = [512, 256, 64, 16]
    config.decoder_channels = (256, 128, 64, 16)
    
    # Others
    config.activation = 'softmax'
    config.vis = True
    
    return config