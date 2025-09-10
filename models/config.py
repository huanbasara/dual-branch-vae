from utils.tensor import SVGTensor


class _DefaultConfig:
    """
    Model config.
    """
    def __init__(self):
        self.args_dim = 256              # Coordinate numericalization, default: 256 (8-bit)
        self.n_args = 2                  # Tensor nb of arguments (from config: 2 for x,y coordinates)
        self.n_commands = len(SVGTensor.COMMANDS_SIMPLIFIED)  # m, l, c, a, EOS, SOS, z

        self.dropout = 0.0                # Dropout rate used in basic layers and Transformers (from config)

        self.model_type = "transformer"  # "transformer" ("lstm" implementation is work in progress)

        self.encode_stages = 1           # One-stage or two-stage: 1 | 2
        self.decode_stages = 1           # One-stage or two-stage: 1 | 2

        self.use_resnet = False          # Use extra fully-connected residual blocks after Encoder (Fixed: was True)
        self.use_vae = False             # Sample latent vector (with reparametrization trick) or use encodings directly (Fixed: was True)
        self.use_vqvae = False           # Use Vector Quantization VAE (Added missing attribute)
        self.use_model_fusion = True     # Use dual-branch model fusion (Added missing attribute)

        self.pred_mode = "one_shot"      # Feed-forward (one-shot) or autogressive: "one_shot" | "autoregressive"
        self.rel_targets = False         # Predict coordinates in relative or absolute format

        self.label_condition = False     # Make all blocks conditional on the label
        self.n_labels = 100              # Number of labels (when used)
        self.dim_label = 64              # Label embedding dimensionality

        self.self_match = False          # Use Hungarian (self-match) or Ordered assignment

        self.n_layers = 4                # Number of Encoder blocks
        self.n_layers_decode = 4         # Number of Decoder blocks
        self.n_heads = 8                 # Transformer config: number of heads
        self.dim_feedforward = 512       # Transformer config: FF dimensionality
        self.d_model = 80                # Transformer config: model dimensionality (from config)

        self.dim_z = 24                  # Latent vector dimensionality (from config)

        self.max_num_groups = 1          # Number of paths (N_P) (from config)
        self.max_seq_len = 30            # Number of commands (N_C)
        self.max_total_len = 40          # Concatenated sequence length (from config)

        self.num_groups_proposal = self.max_num_groups  # Number of predicted paths, default: N_P
        
        # Additional missing attributes for dual-branch VAE (from vae_config_cmd_10.yaml)
        self.img_size = 64               # Image size for model fusion
        self.max_pts_len_thresh = 30     # Max points length threshold
        self.max_enc_len = 40            # Max encoder length
        self.max_dec_len = 40            # Max decoder length
        
        # VQ-VAE related attributes
        self.vq_edim = 2                 # VQ embedding dimension (from config)
        self.codebook_size = 512         # VQ codebook size (from config)
        self.use_cosine_sim = False      # Use cosine similarity in VQ
        self.vq_comb_num = 32            # VQ combination number
        self.vqvae_loss_weight = 0.02    # VQ-VAE loss weight
        
        # Image fusion related attributes  
        self.d_img_model = 1024          # Image encoder model dimension (from config)
        self.img_latent_dim = 16         # Image latent dimension (from config)
        self.loss_w_l1 = 0.01            # L1 loss weight
        
        # Model architecture attributes
        self.ModifiedConstEmbedding = False  # Use modified constant embedding
        self.avg_path_zdim = True        # Average path z dimension
        self.args_decoder = False        # Use arguments decoder
        self.bin_targets = False         # Binary targets
        self.abs_targets = True          # Absolute targets
        self.connect_through = True      # Connect through layers
        self.use_sigmoid = True          # Use sigmoid activation
        
        # Training attributes
        self.kl_coe = 0.4               # KL coefficient
        self.diffvg_loss_weight = 0.01  # DiffVG loss weight
        self.loader_num_workers = 16    # DataLoader workers

    def get_model_args(self):
        model_args = []

        model_args += ["commands_grouped", "args_grouped"] if self.encode_stages <= 1 else ["commands", "args"]

        if self.rel_targets:
            model_args += ["commands_grouped", "args_rel_grouped"] if self.decode_stages == 1 else ["commands", "args_rel"]
        else:
            model_args += ["commands_grouped", "args_grouped"] if self.decode_stages == 1 else ["commands", "args"]

        if self.label_condition:
            model_args.append("label")

        return model_args


class SketchRNN(_DefaultConfig):
    # LSTM - Autoregressive - One-stage
    def __init__(self):
        super().__init__()

        self.model_type = "lstm"

        self.pred_mode = "autoregressive"
        self.rel_targets = True


class Sketchformer(_DefaultConfig):
    # Transformer - Autoregressive - One-stage
    def __init__(self):
        super().__init__()

        self.pred_mode = "autoregressive"
        self.rel_targets = True


class OneStageOneShot(_DefaultConfig):
    # Transformer - One-shot - One-stage
    def __init__(self):
        super().__init__()

        self.encode_stages = 1
        self.decode_stages = 1


class Hierarchical(_DefaultConfig):
    # Transformer - One-shot - Two-stage - Ordered
    def __init__(self):
        super().__init__()

        self.encode_stages = 2
        self.decode_stages = 2


class HierarchicalSelfMatching(_DefaultConfig):
    # Transformer - One-shot - Two-stage - Hungarian
    def __init__(self):
        super().__init__()
        self.encode_stages = 2
        self.decode_stages = 2
        self.self_match = True
