from lib.media_auto.models.vision.model_registry import ModelRegistry
from lib.media_auto.models.interfaces.ai_model import ModelConfig

class ModelSwitcher:
    """模型切換器，用於動態更換模型"""
    def __init__(self, vision_manager: 'VisionContentManager'):
        self.manager = vision_manager
    
    def switch_vision_model(self, model_type: str, **config):
        """切換視覺模型"""
        model_class = ModelRegistry.get_model(model_type)
        self.manager.vision_model = model_class(ModelConfig(**config))
    
    def switch_text_model(self, model_type: str, **config):
        """切換文本模型"""
        model_class = ModelRegistry.get_model(model_type)
        self.manager.text_model = model_class(ModelConfig(**config)) 