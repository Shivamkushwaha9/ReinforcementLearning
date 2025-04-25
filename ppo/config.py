#Parameters
NUM_STEPS = 2048          # Number of steps per environment per update
BATCH_SIZE = 64           # Mini-batch size for updates
NUM_EPOCHS = 10           # Number of optimization epochs per update
GAMMA = 0.99              # Discount factor
GAE_LAMBDA = 0.95         # GAE parameter
CLIP_EPS = 0.2            # PPO clip parameter
LR = 3e-4                 # Learning rate
HIDDEN_DIM = 256          # Network hidden layer size
ENTROPY_COEF = 0.01       # Entropy coefficient
VALUE_COEF = 0.5          # Value loss coefficient
MAX_EPISODES = 1000       # Maximum training episodes