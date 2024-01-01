from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 1  # 2 GPUs should also be good
    max_epochs = 100  # 230k steps
    save_interval = 0.2  # save every 0.2 epoch
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 5  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4  # a small learning rate is very important for SAVi training
    dec_lr = 3e-4  # 3e-4 for the Transformer decoder
    clip_grad = 0.05  # following the paper
    # warmup_steps_pct = 0.025  # warmup in the first 2.5% of total steps
    warmup_steps_pct = 0.0125  # warmup in the first 2.5% of total steps

    # data settings
    dataset = 'atari'
    data_root = './data/atari'
    n_sample_frames = 6  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    filter_enter = False  # no need to filter videos when training SAVi
    train_batch_size = 16 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'STEVE'
    resolution = (64, 64)
    input_frames = n_sample_frames

    # Slot Attention
    slot_size = 128
    slot_dict = dict(
        num_slots=7,  # at most 6 objects per scene
        slot_size=slot_size,
        # slot_mlp_size=256,
        slot_mlp_size=slot_size*4,
        num_iterations=2,
        kernel_mlp=False,
    )

    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=4096,
        dvae_ckp_path='pretrained/dvae_physion_params/model_20.pth',
    )

    # CNN Encoder
    enc_dict = dict(
        enc_channels=(3, 64, 64, 64, 64),
        enc_ks=5,
        enc_out_channels=slot_size,
        enc_norm='',
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=slot_size,
    )

    # Predictor
    pred_dict = dict(
        pred_type='transformer',
        pred_rnn=True,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ffn_dim=slot_size * 4,
        pred_sg_every=None,
    )

    # loss settings
    loss_dict = dict(
        use_img_recon_loss=True,  # additional img recon loss via dVAE decoder
        train_dvae=False,  # train dVAE
    )

    token_recon_loss_w = 1.
    img_recon_loss_w = 1.