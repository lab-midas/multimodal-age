"""File for image model"""
from models.ResNet3D import generate_ResNet
from models.unimodal import Unimodal

class Image(Unimodal):
    """
    Image unimodal model.
    """
    def __init__(self, config, device, image_input_channels):
        super(Image, self).__init__(config, device)

        if config.image_encoder_model_depth in [50, 101, 152, 200]:
            embedding_dim = 2048
        elif config.image_encoder_model_depth in [10, 18, 34]:
            embedding_dim = 512
        else:
            raise ValueError("Unsupported ResNet depth")

        self.encoder = generate_ResNet(model_depth=config.image_encoder_model_depth, n_input_channels = image_input_channels)
        self.init_heads(config, embedding_dim, config.hidden_dim_image_ph)

    def get_data(self, batch):
        return batch["image"]
