"""One data interface for all supported modalities."""


def build_data(config):
    name = config['dataset']['name']
    if name == 'urbansound8k':
        from .audio import build_audio_data
        return build_audio_data(config)
    if name == 'shakespeare':
        from .text import build_text_data
        return build_text_data(config)
    from .vision import build_vision_data
    return build_vision_data(config)
