class WarmUpExponentialLR():
  def __init__(self, gamma, warmup_epochs, steps_per_epoch=1):
    self.gamma = gamma
    self.coef = 1.0
    self.steps_per_epoch = steps_per_epoch
    self.warmup_epochs = warmup_epochs
  
  def __call__(self, epoch):
    if (epoch + 1) <= self.warmup_epochs:
      return (epoch + 1) / self.warmup_epochs 
    else:
      self.coef = self.gamma ** ((epoch - self.warmup_epochs) / self.steps_per_epoch)
      return self.coef