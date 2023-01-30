import os
## Configure code settings here ##

# USER DEFINED BELOW:
audio_metadata = 'dummy.csv'
train_test_split = 'dummy.csv'
audio_lookup = 'dummy.csv'
participant_metadata = 'dummy.csv'
bucket_name = 'dummy'

# Get root of project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

feat_dir = os.path.join(ROOT_DIR, 'outputs', 'feats')  # Output directory to save features
results_dir = os.path.join(ROOT_DIR, 'outputs', 'results')  # Output for results

batch_size = 16  # Batch size used for training BNN ResNet-50
epochs = 20  # Number of epochs for training
max_overrun = 10  # Max number of epochs to train for without improvement
class_threshold = 0.5  # Class threshold on decision making
lr=0.0001  # Learning rate of optimiser (BCELoss() default criterion)

# Number of samples for posterior estimation of BNN, used in evaluate.py only
n_samples = 20


# Create directories if they do not exist:
for directory in [feat_dir, results_dir]:
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print("Created directory:", directory)