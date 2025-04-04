from configs.__base__ import *
from configs.__base__.cfg_model_hybridcat import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_myad):

    def __init__(self):
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)
        cfg_model_myad.__init__(self)

        self.seed = 42
        self.size = 256
        self.epoch_full = 500
        self.warmup_epochs = 0
        self.test_start_epoch = self.epoch_full
        self.test_per_epoch = self.epoch_full // 2
        self.batch_train = 8
        self.batch_test_per = 8
        self.lr = 1e-4 * self.batch_train / 8
        self.weight_decay = 0.0001
        self.metrics = [
            'mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max',
            'mAUPRO_px',
            'mAUROC_px', 'mAP_px', 'mF1_max_px',
            'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
            'mIoU_max_px',
        ]

        # ==> data.
        self.data.type = 'RealIAD'
        self.data.root = 'data/realiad'
        self.data.use_sample = False
        self.data.views = []  # ['C1', 'C2', 'C3', 'C4', 'C5']
        self.data.cls_names = ['audiojack', 'pcb', 'phone_battery', 'sim_card_set', 'switch', 'terminalblock',
                               'toothbrush', 'toy', 'transistor1', 'usb', 'usb_adaptor', 'zipper', 'bottle_cap',
                               'end_cap', 'fire_hood', 'mounts', 'plastic_nut', 'plastic_plug', 'regulator',
                               'rolled_strip_base', 'toy_brick', 'u_block', 'vcpill', 'wooden_beads', 'woodstick',
                               'tape', 'porcelain_doll', 'mint', 'eraser', 'button_battery']

        self.data.train_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.data.test_transforms = self.data.train_transforms
        self.data.target_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
        ]

        self.model_backbone = Namespace()
        self.model_backbone.name = '_vit_small_patch16_224_dino'
        self.model_backbone.kwargs = dict(pretrained=True, checkpoint_path='', strict=True,
                                          img_size=self.size, teachers=[3, 6, 9], neck=[3, 6, 9, 12])
        self.model_fusion = Namespace()
        self.model_fusion.name = 'fusion_myad'
        self.model_fusion.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=384, mul=1)

        self.model_decoder = Namespace()
        self.model_decoder.name = '_create_decoder'
        self.model_decoder.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=384, num_heads=6,
                                         img_size=self.size)

        self.model = Namespace()
        self.model.name = 'HybridCaT'
        self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_backbone=self.model_backbone,
                                 model_fusion=self.model_fusion, model_decoder=self.model_decoder)

        # ==> evaluator
        self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=[16, 16], max_step_aupro=100)

        # ==> optimizer
        self.optim.lr = self.lr
        self.optim.kwargs = dict(name='adamw', betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay,
                                 amsgrad=False)

        # ==> trainer
        self.trainer.name = 'MyADTrainer'
        self.trainer.logdir_sub = ''
        self.trainer.resume_dir = ''
        self.trainer.epoch_full = self.epoch_full
        self.trainer.scheduler_kwargs = dict(
            name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
            warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs,
            cooldown_epochs=0, use_iters=True,
            patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8),
            cycle_decay=0.1, decay_rate=0.1)
        self.trainer.mixup_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0,
                                         switch_prob=0.5, mode='batch', correct_lam=True, label_smoothing=0.1)
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch

        self.trainer.data.batch_size = self.batch_train
        self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

        # ==> loss
        self.loss.loss_terms = [
            dict(type='CosLoss', name='cos', avg=False, lam=1.0),
        ]

        # ==> logging
        self.logging.log_terms_train = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='data_t', fmt=':>5.3f'),
            dict(name='optim_t', fmt=':>5.3f'),
            dict(name='lr', fmt=':>7.6f'),
            dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
        self.logging.log_terms_test = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
