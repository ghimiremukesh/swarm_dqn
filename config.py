class CONFIG():

    def __init__(self, MEMORY_CAPACITY=100000, BATCH_SIZE=128, GAMMA_DECAY=0.5, EPS_DECAY=0.25, EPS_RESET_PERIOD=100,
                 GAMMA_PERIOD=200, EPSILON=0.95, MAX_EP_STEPS=200, SEED=0, MAX_UPDATES=2000000, GAMMA=0.99,
                 GAMMA_END=1, EPS_END=0.05, EPS_PERIOD=1, LR=1e-3, LR_PERIOD=50000, LR_DECAY=0.5, LR_END=1e-5,
                 LAM=0, HARD_UPDATE=0, TAU=0.001, SOFT_UPDATE=True, MAX_MODEL=5, BETA=0.25, OMEGA=1.5):

        self.MAX_UPDATES = MAX_UPDATES
        self.MAX_EP_STEPS = MAX_EP_STEPS

        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE

        self.GAMMA = GAMMA
        self.GAMMA_END = GAMMA_END
        self.GAMMA_DECAY = GAMMA_DECAY
        self.GAMMA_PERIOD = GAMMA_PERIOD

        self.EPS_RESET_PERIOD = EPS_RESET_PERIOD
        self.EPS_PERIOD = EPS_PERIOD
        self.EPS_DECAY = EPS_DECAY
        self.EPS_END = EPS_END

        self.EPSILON = EPSILON
        self.SEED = SEED
        self.LR = LR
        self.LR_PERIOD = LR_PERIOD
        self.LR_DECAY = LR_DECAY
        self.LR_END = LR_END
        self.LAM = LAM

        self.HARD_UPDATE = HARD_UPDATE
        self.SOFT_UPDATE = SOFT_UPDATE
        self.TAU = TAU

        self.MAX_MODEL = MAX_MODEL

        self.BETA = BETA

        self.OMEGA = OMEGA




