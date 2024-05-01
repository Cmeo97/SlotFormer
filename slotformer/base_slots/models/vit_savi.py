import copy
import enum
import math

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Resize
import timm

from nerv.training import BaseModel
from nerv.models import deconv_out_shape, conv_norm_act, deconv_norm_act

from .savi import SlotAttention
from .utils import assert_shape, SoftPositionEmbed, torch_cat
from .predictor import ResidualMLPPredictor, TransformerPredictor, \
    RNNPredictorWrapper


def resize_patches_to_image(patches, size=None, scale_factor=None, resize_mode="bilinear"):
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns:
        Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = math.sqrt(n_patches)
    patch_size = int(math.sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = F.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])

class _VitFeatureType(enum.Enum):
    BLOCK = 1
    KEY = 2
    VALUE = 3
    QUERY = 4
    CLS = 5


class _VitFeatureHook:
    """Auxilliary class used to extract features from timm ViT models."""

    def __init__(self, feature_type: _VitFeatureType, block: int, drop_cls_token: bool = True):
        """Initialize VitFeatureHook.

        Args:
            feature_type: Type of feature to extract.
            block: Number of block to extract features from. Note that this is not zero-indexed.
            drop_cls_token: Drop the cls token from the features. This assumes the cls token to
                be the first token of the sequence.
        """
        assert isinstance(feature_type, _VitFeatureType)
        self.feature_type = feature_type
        self.block = block
        self.drop_cls_token = drop_cls_token
        self.name = f"{feature_type.name.lower()}{block}"
        self.remove_handle = None  # Can be used to remove this hook from the model again

        self._features = None

    @staticmethod
    def create_hook_from_feature_level(feature_level):
        feature_level = str(feature_level)
        prefixes = ("key", "query", "value", "block", "cls")
        for prefix in prefixes:
            if feature_level.startswith(prefix):
                _, _, block = feature_level.partition(prefix)
                feature_type = _VitFeatureType[prefix.upper()]
                block = int(block)
                break
        else:
            feature_type = _VitFeatureType.BLOCK
            try:
                block = int(feature_level)
            except ValueError:
                raise ValueError(f"Can not interpret feature_level '{feature_level}'.")

        return _VitFeatureHook(feature_type, block)

    def register_with(self, model):
        supported_models = (
            timm.models.vision_transformer.VisionTransformer,
            timm.models.beit.Beit,
            timm.models.vision_transformer_sam.VisionTransformerSAM,
        )
        model_names = ["vit", "beit", "samvit"]

        if not isinstance(model, supported_models):
            raise ValueError(
                f"This hook only supports classes {', '.join(str(cl) for cl in supported_models)}."
            )

        if self.block > len(model.blocks):
            raise ValueError(
                f"Trying to extract features of block {self.block}, but model only has "
                f"{len(model.blocks)} blocks"
            )

        block = model.blocks[self.block - 1]
        if self.feature_type == _VitFeatureType.BLOCK:
            self.remove_handle = block.register_forward_hook(self)
        else:
            if isinstance(block, timm.models.vision_transformer.ParallelBlock):
                raise ValueError(
                    f"ViT with `ParallelBlock` not supported for {self.feature_type} extraction."
                )
            elif isinstance(model, timm.models.beit.Beit):
                raise ValueError(f"BEIT not supported for {self.feature_type} extraction.")
            self.remove_handle = block.attn.qkv.register_forward_hook(self)

        model_name_map = dict(zip(supported_models, model_names))
        self.model_name = model_name_map.get(type(model), None)

        return self

    def pop(self) -> torch.Tensor:
        """Remove and return extracted feature from this hook.

        We only allow access to the features this way to not have any lingering references to them.
        """
        assert self._features is not None, "Feature extractor was not called yet!"
        features = self._features
        self._features = None
        return features

    def __call__(self, module, inp, outp):
        if self.feature_type == _VitFeatureType.BLOCK:
            features = outp
            if self.drop_cls_token:
                # First token is CLS token.
                if self.model_name == "samvit":
                    # reshape outp (B,H,W,C) -> (B,H*W,C)
                    features = outp.flatten(1,2)
                else:
                    features = features[:, 1:]
        elif self.feature_type in {
            _VitFeatureType.KEY,
            _VitFeatureType.QUERY,
            _VitFeatureType.VALUE,
        }:
            # This part is adapted from the timm implementation. Unfortunately, there is no more
            # elegant way to access keys, values, or queries.
            B, N, C = inp[0].shape
            qkv = outp.reshape(B, N, 3, C)  # outp has shape B, N, 3 * H * (C // H)
            q, k, v = qkv.unbind(2)

            if self.feature_type == _VitFeatureType.QUERY:
                features = q
            elif self.feature_type == _VitFeatureType.KEY:
                features = k
            else:
                features = v
            if self.drop_cls_token:
                # First token is CLS token.
                features = features[:, 1:]
        elif self.feature_type == _VitFeatureType.CLS:
            # We ignore self.drop_cls_token in this case as it doesn't make any sense.
            features = outp[:, 0]  # Only get class token.
        else:
            raise ValueError("Invalid VitFeatureType provided.")

        self._features = features


class ViTStoSAVi(BaseModel):
    """SA model with stochastic kernel and additional prior_slots head.
    Encoder is replaced by a SAM ViT.
    If loss_dict['kld_method'] = 'none', it becomes a standard SAVi model.
    """

    def __init__(
        self,
        resolution,
        clip_len,
        slot_dict=dict(
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=2,
            kernel_mlp=True,
        ),
        enc_dict=dict(
            enc_channels=(3, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=128,
            enc_norm='',
        ),
        dec_dict=dict(
            dec_channels=(128, 64, 64, 64, 64),
            dec_resolution=(8, 8),
            dec_ks=5,
            dec_norm='',
        ),
        pred_dict=dict(
            pred_type='transformer',
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=512,
            pred_sg_every=None,
        ),
        loss_dict=dict(
            use_post_recon_loss=True,
            kld_method='var-0.01',  # 'none' to make it deterministic
        ),
        eps=1e-6,
    ):
        super().__init__()

        self.resolution = resolution
        self.clip_len = clip_len
        self.eps = eps

        self.slot_dict = slot_dict
        self.enc_dict = enc_dict
        self.dec_dict = dec_dict
        self.pred_dict = pred_dict
        self.loss_dict = loss_dict

        self._build_slot_attention()
        self._build_encoder()
        self._build_decoder()
        self._build_img_decoder()
        self._build_predictor()
        self._build_loss()

        # a hack for only extracting slots
        self.testing = False

    def _build_slot_attention(self):
        # Build SlotAttention module
        # kernels x img_feats --> posterior_slots
        self.enc_out_channels = self.enc_dict['enc_out_channels']
        self.num_slots = self.slot_dict['num_slots']
        self.slot_size = self.slot_dict['slot_size']
        self.slot_mlp_size = self.slot_dict['slot_mlp_size']
        self.num_iterations = self.slot_dict['num_iterations']

        # directly use learnable embeddings for each slot
        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)))

        # predict the (\mu, \sigma) to sample the `kernels` input to SA
        if self.slot_dict.get('kernel_mlp', True):
            self.kernel_dist_layer = nn.Sequential(
                nn.Linear(self.slot_size, self.slot_size * 2),
                nn.LayerNorm(self.slot_size * 2),
                nn.ReLU(),
                nn.Linear(self.slot_size * 2, self.slot_size * 2),
            )
        else:
            self.kernel_dist_layer = nn.Sequential(
                nn.Linear(self.slot_size, self.slot_size * 2), )

        # predict the `prior_slots`
        # useless, just for compatibility to load pre-trained weights
        self.prior_slot_layer = nn.Sequential(
            nn.Linear(self.slot_size, self.slot_size),
            nn.LayerNorm(self.slot_size),
            nn.ReLU(),
            nn.Linear(self.slot_size, self.slot_size),
        )

        self.slot_attention = SlotAttention(
            in_features=self.enc_out_channels,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
        )

    def _build_encoder(self):
        self.vit_model_name = self.enc_dict['vit_model_name']
        self.vit_use_pretrained = self.enc_dict['vit_use_pretrained']
        self.vit_freeze = self.enc_dict['vit_freeze']
        self.vit_feature_level = self.enc_dict['vit_feature_level']
        self.vit_num_patches = self.enc_dict['vit_num_patches']

        def feature_level_to_list(feature_level):
            if feature_level is None:
                return []
            elif isinstance(feature_level, (int, str)):
                return [feature_level]
            else:
                return list(feature_level)

        self.feature_levels = feature_level_to_list(self.vit_feature_level)

        model = timm.create_model(self.vit_model_name, pretrained=self.vit_use_pretrained, dynamic_img_size=True)
        # Delete unused parameters from classification head
        if hasattr(model, "head"):
            del model.head
        if hasattr(model, "fc_norm"):
            del model.fc_norm

        if len(self.feature_levels) > 0:
            self._feature_hooks = [
                _VitFeatureHook.create_hook_from_feature_level(level).register_with(model) for level in self.feature_levels
            ]
            feature_dim = model.num_features * len(self.feature_levels)

            # Remove modules not needed in computation of features
            max_block = max(hook.block for hook in self._feature_hooks)
            new_blocks = model.blocks[:max_block]  # Creates a copy
            del model.blocks
            model.blocks = new_blocks
            model.norm = nn.Identity()

        self.vit = model
        self._feature_dim = feature_dim

        if self.vit_freeze:
            self.vit.requires_grad_(False)
            # BatchNorm layers update their statistics in train mode. This is probably not desired
            # when the model is supposed to be frozen.
            contains_bn = any(
                isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                for m in self.vit.modules()
            )
            self.run_in_eval_mode = contains_bn
        else:
            self.run_in_eval_mode = False

        self._init_pos_embed(self.enc_dict["vit_out_dim"], self.enc_dict["enc_out_channels"])

    def _init_pos_embed(self, encoder_output_dim, token_dim):
        layers = []
        layers.append(nn.LayerNorm(encoder_output_dim))
        layers.append(nn.Linear(encoder_output_dim, encoder_output_dim))
        nn.init.zeros_(layers[-1].bias)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(encoder_output_dim, token_dim))
        nn.init.zeros_(layers[-1].bias)
        self.encoder_pos_embedding = nn.Sequential(*layers)

    def _build_decoder(self):
        dec_hidden_layers = self.dec_dict["dec_hidden_layers"]
        self.dec_output_dim = self.enc_dict["vit_out_dim"]
        dec_input_dim = self.enc_dict["enc_out_channels"]

        layers = []
        current_dim = self.slot_dict["slot_size"]
    
        for dec_hidden_dim in dec_hidden_layers:
            layers.append(nn.Linear(current_dim, dec_hidden_dim))
            nn.init.zeros_(layers[-1].bias)
            layers.append(nn.ReLU(inplace=True))
            current_dim = dec_hidden_dim

        layers.append(nn.Linear(current_dim, self.dec_output_dim + 1))
        nn.init.zeros_(layers[-1].bias)
        
        self.decoder = nn.Sequential(*layers)
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.vit_num_patches, dec_input_dim) * 0.02)

    def _build_img_decoder(self):
        # Build Decoder
        # Spatial broadcast --> PosEnc --> DeConv CNN
        self.dec_channels = self.dec_dict['img_dec_channels']  # CNN channels
        self.dec_resolution = self.dec_dict['img_dec_resolution']  # broadcast size
        self.dec_ks = self.dec_dict['img_dec_ks']  # kernel size
        self.dec_norm = self.dec_dict['img_dec_norm']  # norm in CNN
        assert self.dec_channels[0] == self.slot_size, \
            'wrong in_channels for Decoder'
        modules = []
        in_size = self.dec_resolution[0]
        out_size = in_size
        stride = 2
        for i in range(len(self.dec_channels) - 1):
            if out_size == self.resolution[0]:
                stride = 1
            modules.append(
                deconv_norm_act(
                    self.dec_channels[i],
                    self.dec_channels[i + 1],
                    kernel_size=self.dec_ks,
                    stride=stride,
                    norm=self.dec_norm,
                    act='relu'))
            out_size = deconv_out_shape(out_size, stride, self.dec_ks // 2,
                                        self.dec_ks, stride - 1)

        # assert_shape(
        #     self.resolution,
        #     (out_size, out_size),
        #     message="Output shape of decoder did not match input resolution. "
        #     "Try changing `decoder_resolution`.",
        # )
        self.out_size = out_size
        self.use_downsample = self.resolution[0] != out_size
        if self.resolution[0] != out_size:
            print(f"Downscaling from {self.resolution} to ({out_size}, {out_size})")
            self.downsample = Resize(size=(out_size, out_size))

        # out Conv for RGB and seg mask
        modules.append(
            nn.Conv2d(
                self.dec_channels[-1], 4, kernel_size=1, stride=1, padding=0))

        self.img_decoder = nn.Sequential(*modules)
        self.img_decoder_pos_embedding = SoftPositionEmbed(self.slot_size,
                                                       self.dec_resolution)

    def _build_predictor(self):
        """Predictor as in SAVi to transition slot from time t to t+1."""
        # Build Predictor
        pred_type = self.pred_dict.get('pred_type', 'transformer')
        # Transformer (object interaction) --> LSTM (scene dynamic)
        if pred_type == 'mlp':
            self.predictor = ResidualMLPPredictor(
                [self.slot_size, self.slot_size * 2, self.slot_size],
                norm_first=self.pred_dict['pred_norm_first'],
            )
        elif pred_type == 'gru':
            self.predictor = nn.GRU(self.slot_size, self.slot_size, batch_first=True)
        elif pred_type == 'grucell':
            self.predictor = nn.GRUCell(self.slot_size, self.slot_size)
        elif pred_type == 'lstm':
            self.predictor = nn.LSTMCell(self.slot_size, self.slot_size)
        else:
            self.predictor = TransformerPredictor(
                self.slot_size,
                self.pred_dict['pred_num_layers'],
                self.pred_dict['pred_num_heads'],
                self.pred_dict['pred_ffn_dim'],
                norm_first=self.pred_dict['pred_norm_first'],
            )
        # wrap LSTM
        if self.pred_dict['pred_rnn']:
            self.predictor = RNNPredictorWrapper(
                self.predictor,
                self.slot_size,
                self.slot_mlp_size,
                num_layers=1,
                rnn_cell='LSTM',
                sg_every=self.pred_dict['pred_sg_every'],
            )

    def _build_loss(self):
        """Loss calculation settings."""
        self.use_post_recon_loss = self.loss_dict['use_post_recon_loss']
        self.use_consistency_loss = self.loss_dict['use_consistency_loss']
        assert self.use_post_recon_loss
        # stochastic SAVi by sampling the kernels
        kld_method = self.loss_dict['kld_method']
        # a smaller sigma for the prior distribution
        if '-' in kld_method:
            kld_method, kld_var = kld_method.split('-')
            self.kld_log_var = math.log(float(kld_var))
        else:
            self.kld_log_var = math.log(1.)
        self.kld_method = kld_method
        assert self.kld_method in ['var', 'none']

    def _kld_loss(self, prior_dist, post_slots):
        """KLD between (mu, sigma) and (0 or mu, 1)."""
        if self.kld_method == 'none':
            return torch.tensor(0.).type_as(prior_dist)
        assert prior_dist.shape[-1] == self.slot_size * 2
        mu1 = prior_dist[..., :self.slot_size]
        log_var1 = prior_dist[..., self.slot_size:]
        mu2 = mu1.detach().clone()  # no penalty for mu
        log_var2 = torch.ones_like(log_var1).detach() * self.kld_log_var
        sigma1 = torch.exp(log_var1 * 0.5)
        sigma2 = torch.exp(log_var2 * 0.5)
        kld = torch.log(sigma2 / sigma1) + \
            (torch.exp(log_var1) + (mu1 - mu2)**2) / \
            (2. * torch.exp(log_var2)) - 0.5
        return kld.sum(-1).mean()

    def _sample_dist(self, dist):
        """Sample values from Gaussian distribution."""
        assert dist.shape[-1] == self.slot_size * 2
        mu = dist[..., :self.slot_size]
        # not doing any stochastic
        if self.kld_method == 'none':
            return mu
        log_var = dist[..., self.slot_size:]
        eps = torch.randn_like(mu).detach()
        sigma = torch.exp(log_var * 0.5)
        return mu + eps * sigma

    def _transformer_compute_positions(self, features):
        """Compute positions for Transformer features."""
        n_tokens = features.shape[1]
        image_size = math.sqrt(n_tokens)
        image_size_int = int(image_size)
        assert (
            image_size_int == image_size
        ), "Position computation for Transformers requires square image"

        spatial_dims = (image_size_int, image_size_int)
        positions = torch.cartesian_prod(
            *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
        )
        return positions
    
    def vit_encode(self, x):
        if self.run_in_eval_mode and self.training:
            self.eval()

        if self.vit_freeze:
            # Speed things up a bit by not requiring grad computation.
            with torch.no_grad():
                features = self.vit.forward_features(x)
        else:
            features = self.vit.forward_features(x)

        if self._feature_hooks is not None:
            hook_features = [hook.pop() for hook in self._feature_hooks]

        if len(self.feature_levels) == 0:
            # Remove class token when not using hooks.
            features = features[:, 1:]
            positions = self._transformer_compute_positions(features)
        else:
            features = hook_features[: len(self.feature_levels)]
            positions = self._transformer_compute_positions(features[0])
            features = torch.cat(features, dim=-1)

        return features
    
    def _get_encoder_out(self, img):
        features = self.vit_encode(img)
        encoder_out = self.encoder_pos_embedding(features)
        # `encoder_out` has shape: [B, H*W, enc_out_channels]
        return encoder_out, features

    def encode(self, img, prev_slots=None):
        """Encode from img to slots."""
        B, T, C, H, W = img.shape
        img = img.flatten(0, 1)

        encoder_out, features = self._get_encoder_out(img)
        encoder_out = encoder_out.unflatten(0, (B, T))
        features = features.unflatten(0, (B, T))
        # `encoder_out` has shape: [B, T, H*W, out_features]

        # init slots
        init_latents = self.init_latents.repeat(B, 1, 1)  # [B, N, C]

        # apply SlotAttn on video frames via reusing slots
        all_kernel_dist, all_post_slots, all_attns = [], [], []
        hidden = None
        for idx in range(T):
            # init
            if prev_slots is None:
                latents = init_latents  # [B, N, C]
            else:
                if self.pred_dict['pred_type'] == 'gru':
                    latents, hidden = self.predictor(prev_slots, hidden)  # [B, N, C]
                elif self.pred_dict['pred_type'] == 'grucell':
                    latents = self.predictor(prev_slots.view(-1, self.slot_size), hidden)
                    latents = latents.view(-1, self.num_slots, self.slot_size)
                else:
                    latents = self.predictor(prev_slots)

            # stochastic `kernels` as SA input
            kernel_dist = self.kernel_dist_layer(latents)
            kernels = self._sample_dist(kernel_dist)
            all_kernel_dist.append(kernel_dist)

            # perform SA to get `post_slots`
            post_slots, attns = self.slot_attention(encoder_out[:, idx], kernels)
            all_post_slots.append(post_slots)
            all_attns.append(attns) # [B, num_inputs, num_slots]

            # next timestep
            prev_slots = post_slots

        # (B, T, self.num_slots, self.slot_size)
        kernel_dist = torch.stack(all_kernel_dist, dim=1)
        post_slots = torch.stack(all_post_slots, dim=1)
        attn_maps = torch.stack(all_attns, dim=1)

        return kernel_dist, post_slots, attn_maps, encoder_out, features

    def _reset_rnn(self):
        self.predictor.reset()

    def forward(self, data_dict):
        """A wrapper for model forward.

        If the input video is too long in testing, we manually cut it.
        """
        img = data_dict['img']
        T = img.shape[1]
        if T <= self.clip_len or self.training:
            return self._forward(img, None)

        # try to find the max len to input each time
        clip_len = T
        while True:
            try:
                _ = self._forward(img[:, :clip_len], None)
                del _
                torch.cuda.empty_cache()
                break
            except RuntimeError:  # CUDA out of memory
                clip_len = clip_len // 2 + 1
        # update `clip_len`
        self.clip_len = max(self.clip_len, clip_len)
        # no need to split
        if clip_len == T:
            return self._forward(img, None)

        # split along temporal dim
        cat_dict = None
        prev_slots = None
        for clip_idx in range(0, T, clip_len):
            out_dict = self._forward(img[:, clip_idx:clip_idx + clip_len],
                                     prev_slots)
            # because this should be in test mode, we detach the outputs
            if cat_dict is None:
                cat_dict = {k: [v.detach()] for k, v in out_dict.items()}
            else:
                for k, v in out_dict.items():
                    cat_dict[k].append(v.detach())
            prev_slots = cat_dict['post_slots'][-1][:, -1].detach().clone()
            del out_dict
            torch.cuda.empty_cache()
        cat_dict = {k: torch_cat(v, dim=1) for k, v in cat_dict.items()}
        return cat_dict

    def _forward(self, img, prev_slots=None):
        """Forward function.

        Args:
            img: [B, T, C, H, W]
            prev_slots: [B, num_slots, slot_size] or None,
                the `post_slots` from last timestep.
        """
        # reset RNN states if this is the first frame
        # if prev_slots is None:
        #     self._reset_rnn()

        B, T = img.shape[:2]
        if self.use_downsample:
            img_downsampled = self.downsample(rearrange(img, 'b t c h w -> (b t) c h w'))
            img_downsampled = rearrange(img_downsampled, '(b t) c h w -> b t c h w', b=B)

        kernel_dist, post_slots, attn_maps, encoder_out, features = \
            self.encode(img, prev_slots=prev_slots)
        # `slots` has shape: [B, T, self.num_slots, self.slot_size]

        out_dict = {
            'feat': features, # [B, T, H*W, D]
            'attn_maps': attn_maps,  # [B, T, num_inputs, num_slots]
            'post_slots': post_slots,  # [B, T, num_slots, C]
            'kernel_dist': kernel_dist,  # [B, T, num_slots, 2C]
            'img': img,  # [B, T, 3, H', W']
            'img_original': img,  # [B, T, 3, H, W]
        }
        if self.use_downsample:
            out_dict['img'] = img_downsampled
        if self.testing:
            return out_dict

        if self.use_post_recon_loss:
            post_recon_feat = self.decode(post_slots.flatten(0, 1))
            post_recon_img, post_recons, post_masks, _ = \
                self.img_decode(post_slots.flatten(0, 1).detach())
            post_dict = {
                'post_recon_feat': post_recon_feat,  # [B*T, H*W, D]
                'post_recon_combined': post_recon_img,  # [B*T, 3, H, W]
                'post_recons': post_recons,  # [B*T, num_slots, 3, H, W]
                'post_masks': post_masks,  # [B*T, num_slots, 1, H, W]
            }
            out_dict.update(
                {k: v.unflatten(0, (B, T))
                 for k, v in post_dict.items()})

        return out_dict
    
    def decode(self, slots):
        # slots (bt, k, d)
        init_shape = slots.shape[:-1]
        slots = slots.flatten(0, -2)
        slots = slots.unsqueeze(1).expand(-1, self.vit_num_patches, -1)

        # Simple learned additive embedding as in ViT
        slots = slots + self.dec_pos_embedding
        out = self.decoder(slots)
        out = out.unflatten(0, init_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = out.split([self.dec_output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        # masks = alpha.squeeze(-1)
        # masks_as_image = resize_patches_to_image(masks, size=64, resize_mode="bilinear")

        return reconstruction

    def img_decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        # `slots` has shape: [B, self.num_slots, self.slot_size].
        bs, num_slots, slot_size = slots.shape
        height, width = self.resolution if not self.use_downsample else self.out_size, self.out_size
        num_channels = 3

        # spatial broadcast
        decoder_in = slots.view(bs * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.dec_resolution[0],
                                       self.dec_resolution[1])

        out = self.img_decoder_pos_embedding(decoder_in)
        out = self.img_decoder(out)
        # `out` has shape: [B*num_slots, 4, H, W].

        out = out.view(bs, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]  # [B, num_slots, 3, H, W]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
        recon_combined = torch.sum(recons * masks, dim=1)  # [B, 3, H, W]
        return recon_combined, recons, masks, slots

    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        kld_loss = self._kld_loss(out_dict['kernel_dist'],
                                  out_dict['post_slots'])
        loss_dict = {
            'kld_loss': kld_loss,
        }
        if self.use_post_recon_loss:
            loss_dict['post_recon_feat_loss'] = torch.pow(out_dict['post_recon_feat'] - out_dict['feat'], 2).mean()
            if self.use_downsample:
                loss_dict['post_recon_loss'] = F.mse_loss(out_dict['post_recon_combined'], out_dict['img'])
            else:
                loss_dict['post_recon_loss'] = F.mse_loss(out_dict['post_recon_combined'], out_dict['img_original'])
        if self.use_consistency_loss:
            if self.pred_dict['const_type'] == 'repr':
                B, T, num_slots, slot_dim = out_dict['post_slots'].shape
                cosine_loss = 0.
                for t in range(out_dict['post_slots'].shape[1]-1):
                    z_curr = out_dict['post_slots'][:, t]
                    z_next = out_dict['post_slots'][:, t+1]
                    z_curr = z_curr / z_curr.norm(dim=-1, keepdim=True)
                    z_next = z_next / z_next.norm(dim=-1, keepdim=True)
                    pairwise_sim = torch.bmm(z_curr, z_next.transpose(1, 2)) # cosine similarity 
                    loss = F.mse_loss(pairwise_sim, torch.eye(num_slots, device=pairwise_sim.device).expand(B, -1, -1)) # MSE loss
                    cosine_loss += loss
                cosine_loss /= (out_dict['post_slots'].shape[1]-1)
            elif self.pred_dict['const_type'] == 'attn':
                B, T, num_inputs, num_slots = out_dict['attn_maps'].shape
                cosine_loss = 0.
                for t in range(out_dict['attn_maps'].shape[1]-1):
                    attn_curr = out_dict['attn_maps'][:, t]
                    attn_next = out_dict['attn_maps'][:, t+1]
                    attn_curr = attn_curr / attn_curr.norm(dim=1, keepdim=True)
                    attn_next = attn_next / attn_next.norm(dim=1, keepdim=True)
                    pairwise_sim = torch.bmm(attn_curr.transpose(1, 2), attn_next) # cosine similarity 
                    loss = torch.pow(torch.diagonal(pairwise_sim, dim1=-2, dim2=-1) - torch.ones(num_slots, device=pairwise_sim.device).expand(B, -1), 2).mean() # MSE loss (diag elem only) sum
                    cosine_loss += loss
                cosine_loss /= (out_dict['attn_maps'].shape[1]-1)
            loss_dict['consistency_loss'] = cosine_loss
        return loss_dict

    @property
    def dtype(self):
        return self.slot_attention.dtype

    @property
    def device(self):
        return self.slot_attention.device