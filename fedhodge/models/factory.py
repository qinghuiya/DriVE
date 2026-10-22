from torch import nn


def create_model(name, *, num_classes, image_size=32, vocab_size=None):
    if name == 'resnet18':
        from .resnet import ResNet18
        return ResNet18(num_classes)
    if name == 'vit_tiny':
        from .vit_tiny import ViTTiny
        return ViTTiny(num_classes=num_classes, in_channels=3, image_size=image_size)
    if name == 'tinycnn':
        # Small network for checking the public interface without a GPU.
        return nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, num_classes))
    if name == 'audio_cnn':
        from .audio_cnn import AudioCNN
        return AudioCNN(num_classes=num_classes)
    if name == 'char_lstm':
        from .char_lstm import CharLSTM
        return CharLSTM(vocab_size=vocab_size or num_classes)
    raise ValueError(f'Unknown model: {name}')
