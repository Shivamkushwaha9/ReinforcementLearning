#Parameters
CLIP_EPS = 0.2              #PPO clip parameter
NUM_EPOCHS = 10             #number of times the ppo will be updated
HIDDEN_DIM = 256            #N/w Hidden layer sizeeee
MAX_EPISODES = 500          #Max Episode to run
MAX_TIMESTEPS = 1500        #Max No of timesteps in 1 Ep
LR = 3e-4                   #Learning rate for the N/W
GAE_LAMBDA = 0.95           #GAE Parameter
GAMMA = 0.99                #Discount Factoor
BATCH_SIZE = 64             #Batch size for dataloadear
ENTROPY_COEF = 0.01         # Entropy coefficient
VALUE_COEF = 0.5            # Value loss coefficient