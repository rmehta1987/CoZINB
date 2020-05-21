import numpy as np
import torch
from utils import createTensors
from argument import parse_arguments
from prme import PRME

# parse arguments
args = parse_arguments()

#args.data_filename = 'patient_bow.npz'
args.vocab_filename = 'patient_vocab_all.npz'
args.vocab_size = 21332
#  Created tensors from numpy instead of loading into object for time saving
# createTensors('patient_bow_all.npz', 'patient_vocab_all.npz')

#data = np.load(args.data_filename)

# train model
# fix random seeds for reproducing results
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

model = PRME(args)
#model.fit(data)
model.fit([])