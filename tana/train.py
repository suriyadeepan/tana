import torch


def train_epoch(model, train_iter, hparams):
  # prepare model for training
  if torch.cuda.is_available():
    model.cuda()

  # put model in train mode
  model.train()

  # get loss function from hparams
  #  NOTE are we gerrymandering the term "hyperparameter" ?
  loss_fn = hparams['loss_fn']
  
  # define optimizer
  optim = torch.optim.Adam(
      [ p for p in model.parameters() if p.requires_grad ] # what we don't wanna train
      )

  # keep track of steps
  #  NOTE doesn't idx take care of this? yes, but what about the batches we have skipped?
  steps = 0

  # init epoch-wide loss, accuracy
  epoch_loss, epoch_accuracy = 0, 0

  for idx, batch in enumerate(train_iter):
    # (1) clear gradients
    optim.zero_grad()

    # (2) inputs and targets
    inputs, targets = batch.text[0], batch.label # NOTE why bath.text[0]?

    # if cuda : move data to GPU
    if torch.cuda.is_available():
      inputs = inputs.cuda()
      targets = targets.cuda()

    # skip bad batches; fuck 'em
    if inputs.size()[0] is not hparams['batch_size']:
      continue

    # (3) forward pass
    likelihood = model(inputs)

    # (4) loss calculation
    loss = loss_fn(likelihood, targets)

    # accumulate loss
    epoch_loss += loss.item()

    # (5) optimization
    loss.backward()
    # clip gradients (in-place)
    clip_gradient(model, 1e-1)
    # run optimiztion step on model params using clipped gradients
    optim.step()
    
    # increment step
    steps += 1
    epoch_loss += loss.item()

    # (6) accuracy calculation
    #  TODO look into this. what is happening here? i'm so confused.
    num_corrects = (torch.max(likelihood, 1)[1].view(targets.size()).data == targets.data).float().sum()
    acc = 100.0 * num_corrects/len(batch)

    # accumulate accuracy
    epoch_accuracy += acc.item()

    # (7) print results
    #  every once in a while; NOTE don't add this as a parameter to function and complicate things
    if idx and idx%100 == 0:
      print('({}) Iteration loss : {}'.format(idx, loss.item()))
      
  # Epoch-wide loss and accuracy
  print('Epoch loss : {}, Epoch accuracy : {}%'.format(epoch_loss/steps, epoch_accuracy/steps))

  return epoch_loss/steps, epoch_accuracy/steps

def evaluate(model, test_iter, hparams):
  epoch_loss, epoch_accuracy = 0., 0.
  loss_fn = hparams['loss_fn']

  # prepare model for evaluation
  model.eval()
  if torch.cuda.is_available():
    model.cuda()

  steps = 0
  with torch.no_grad():
    for idx, batch in enumerate(test_iter):

      # (1) get inputs and targets
      inputs, targets = batch.text[0], batch.label

      # if cuda : copy batch to GPU
      if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()

      # avoid bad batches; they suck!
      if inputs.size()[0] is not hparams['batch_size']::
        continue

      # (2) forward
      likelihood = model(inputs)

      # (3) loss calc
      loss = loss_fn(likelihood, targets)
      epoch_loss += loss.item()

      # (4) accuracy calc
      num_corrects = (torch.max(likelihood, 1)[1].view(targets.size()).data == targets.data).float().sum()
      acc = 100.0 * num_corrects/len(batch)
      epoch_accuracy += acc.item()

      steps += 1

    # Epoch-wide loss and accuracy
    print('::[evaluation] Loss : {}, Accuracy : {}'.format(
      epoch_loss/(steps), epoch_accuracy/(steps)))

    return epoch_loss/steps, epoch_accuracy/steps
    
def training(model, hparams, train_iter, valid_iter, 
    epochs=10, save_model_file='.model/model_name.pt'):

  # NOTE select best parameters based on accuracy on validation set
  ev_accuracies = []
  for epoch in range(epochs):
    print('[{}]'.format(epoch+1))
    tr_loss, tr_accuracy = train_epoch(model, train_iter, hparams)
    ev_loss, ev_accuracy = evaluate(model, valid_iter, hparams)
    
    # check for best parameters criterion
    if len(ev_accuracies) and ev_accuracy > max(ev_accuracies):
      # save model
      torch.save(model, model_save_file)

    # keep track of evaluation accuracy
    ev_accuracies.append(ev_loss)
