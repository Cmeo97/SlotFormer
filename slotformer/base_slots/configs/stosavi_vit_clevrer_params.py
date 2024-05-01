from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 2 # 2 GPUs should also be good
    max_epochs = 12  # 230k steps
    save_interval = 0.2  # save every 0.2 epoch
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 5  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4  # a small learning rate is very important for SAVi training
    clip_grad = 0.05  # following the paper
    warmup_steps_pct = 0.025  # warmup in the first 2.5% of total steps

    # data settings
    dataset = 'clevrer'
    data_root = './data/CLEVRER'
    n_sample_frames = 6  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    filter_enter = False  # no need to filter videos when training SAVi
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = gpus * 2

    # model configs
    model = 'ViTStoSAVi'  # stochastic version of SAVi
    resolution = (224, 224)
    input_frames = n_sample_frames

    # Slot Attention
    slot_dict = dict(
        num_slots=7,  # at most 6 objects per scene
        slot_size=128,
        slot_mlp_size=256,
        num_iterations=2,
        kernel_mlp=False,
    )

    # CNN Encoder
    enc_dict = dict(
        vit_model_name="samvit_base_patch16",
        # vit_model_name="vit_small_patch16_224_dino",÷
        # vit_model_name="vit_small_patch14_dinov2.lvd142m",
        vit_use_pretrained=True,
        vit_freeze=True,
        vit_feature_level=12,
        # vit_num_patches=16, # res 64
        vit_num_patches=196, # res 224
        vit_out_dim=768,
        enc_out_channels=128,
    )

    # CNN Decoder
    dec_dict = dict(
        dec_hidden_layers=(1024, 1024, 1024),
        img_dec_channels=(128, 64, 64, 64),
        img_dec_resolution=(8, 8),
        img_dec_ks=5,
        img_dec_norm='',
    )

    # Predictor
    pred_dict = dict(
        pred_type='mlp',  # less information fusion to avoid slots sharing objs
        pred_rnn=False,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ffn_dim=slot_dict['slot_size'] * 4,
        pred_sg_every=None,
        # pred_from='last', # [initial, last]
        # const_type='attn', # [repr, attn]
    )

    # loss configs
    loss_dict = dict(
        use_post_recon_loss=True,
        use_consistency_loss=False,
        use_gate_loss=False,
        kld_method='var-0.01',  # prior Gaussian variance is 0.01
        # kld_method='none'
    )

    post_recon_feat_loss_w = 1.
    post_recon_loss_w = 1.  # posterior slots image recon
    kld_loss_w = 1e-4  # kld on kernels distribution
    consistency_loss_w = 0.1  # consistency loss