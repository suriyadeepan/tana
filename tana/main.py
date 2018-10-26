import argparse
import os

""" IMPORTS
import torch
import torch.nn.functional as F

from data import load_data
from model import <ModelName>
"""

# cmd line arguments
parser = argparse.ArgumentParser()
""" CMDLINE ARGUMENTS
parser.add_argument("--mode", default='train', help='run mode : trian/evaluate/predict')
parser.add_argument("--batch_size", default=32, help='batch size for trianing')
parser.add_argument("--input", 
    default='this is a sample sentence for prediction', 
    help='input sentence to run prediction on')
args, unknown = parser.parse_known_args()
"""

# settings
""" SETTINGS
MODEL_SAVE_PATH='.model'
MODEL_SAVE_FILE=os.path.join(MODEL_SAVE_PATH, '<model_name>.pt')
# create dir if necessary
if not os.path.isdir(MODEL_SAVE_PATH):
  os.makedirs(MODEL_SAVE_PATH)
"""


if __name__ == '__main__':
  # load data
  """ DATA
  _fields, _iters, emb, vocab_size = load_data(batch_size=args.batch_size)
  text_field, label_field = _fields
  train_iter, test_iter, valid_iter = _iters
  """

  # define a loss function
  """ LOSS FUNCTION
  loss_fn = F.cross_entropy
  """
  
  # set hyperparameters
  """ HYPERPARAMETERS
  hparams = { 
    'vocab_size'  : vocab_size, 
    'emb_dim'     : glove_emb.size()[-1],
    'hidden_dim'  : args.hidden_dim,
    'lr'          : 2e-5,
    'output_size' : 2,
    'loss_fn'     : loss_fn,
    'batch_size'  : args.batch_size
    }
  """

  if args.mode == 'predict':
    # load trained model from file
    model = torch.load(MODEL_SAVE_FILE)  
    """ PREDICTION
    predict(model, args.input, _fields)
    """

  elif args.mode == 'train':
    """ TRAINING
    # create LSTM model
    lstmClassifier = LstmClassifier( hparams,
        weights = { 'glove' : glove_emb }
        )
    # train model
    training(lstmClassifier, hparams, train_iter, valid_iter, epochs=10)
    """

  elif args.mode == 'evaluate':
    """ EVALUATION
    # load trained model
    model = torch.load(MODEL_SAVE_FILE)
    # run evaluation
    ev_loss, ev_accuracy = evaluate(model, valid_iter, hparams)
    """
