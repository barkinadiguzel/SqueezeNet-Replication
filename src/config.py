# SqueezeNet configuration

NUM_CLASSES = 1000

# Fire module hyperparameters
S1X1 = [16, 16, 32, 32, 48, 48, 64, 64]  # squeeze layer filter counts
E1X1 = [64, 64, 128, 128, 192, 192, 256, 256]  # expand 1x1 filter counts
E3X3 = [64, 64, 128, 128, 192, 192, 256, 256]  # expand 3x3 filter counts

# Pooling layers
POOL_KERNELS = {
    'maxpool1': 3,
    'maxpool4': 3,
    'maxpool8': 3,
    'avgpool': 13
}

# Convolution layer settings
CONV1_PARAMS = {'in_channels': 3, 'out_channels': 96, 'kernel_size': 7, 'stride': 2, 'padding': 3}
CONV10_PARAMS = {'in_channels': 512, 'out_channels': NUM_CLASSES, 'kernel_size': 1}
