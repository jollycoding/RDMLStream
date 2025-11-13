from net.FCModel import FCModel


class HyperParameter(object):
    def __init__(self):
        self.task = 'GASD'
        self.data_dir = f'data/{self.task}.csv'
        self.metric_learning = True
        self.retrain = False
        self.seed = 1

        self.k = 3
        self.minRE = 0.70
        self.maxMC = 800
        self.maxUMC = int(0.2 * self.maxMC)
        self.init_k_per_class = 30
        self.unlabeled_ratio = 0.8

        self.lmbda = 1e-4

        self.logging_steps = 1000

        # === Core Label Propagation Parameters ===
        self.enable_density_propagate = True
        self.density_threshold_base = 0.7  # Base density threshold
        self.density_radius_factor = 1.5  # Density calculation radius factor

        # Class balance parameters
        self.class_balance_alpha = 0.9  # Class weight smoothing factor
        self.class_balance_beta = 1.0  # Non-linear adjustment sensitivity

        # Time window parameters
        self.sliding_window_size = 1000  # Sliding window size
        self.decay_rate = 0.001  # Decay rate

        # Propagation strategy
        self.propagate_confidence_threshold = 0.5  # Trigger density propagation below this confidence
        self.minority_boost_factor = 0.8  # Minority class threshold reduction factor

        # Model parameters
        self.model = FCModel
        self.metric_epoch = 200
        self.train_datasize = 1000
        self.train_eval_split = 1
        self.train_batch_size = 200
        self.eval_batch_size = 200
        self.learning_rate = 2e-5
        self.device = 'cpu'