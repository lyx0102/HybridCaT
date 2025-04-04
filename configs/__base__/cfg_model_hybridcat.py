from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_model_HybridCaT(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model_backbone = Namespace()
		self.model_backbone.name = '_vit_small_patch16_224_dino'
		self.model_backbone.kwargs = dict(pretrained=True, checkpoint_path='', strict=True,
								   img_size=256, teachers=[3, 6, 9], neck=[3, 6, 9, 12])
		self.model_fusion = Namespace()
		self.model_fusion.name = 'fusion_myad'
		self.model_fusion.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=384, mul=1)

		self.model_decoder = Namespace()
		self.model_decoder.name = '_create_decoder'
		self.model_decoder.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=384, num_heads=6, img_size=256)

		self.model = Namespace()
		self.model.name = 'HybridCaT'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_backbone=self.model_backbone,
								 model_fusion=self.model_fusion, model_decoder=self.model_decoder)