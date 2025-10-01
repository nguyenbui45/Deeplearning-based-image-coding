from torch.optim.lr_scheduler import LambdaLR

def poly_decay_every_10(epoch,decay_every,total_epochs,power):
    # decay_step = how many times we've passed a decay interval
    decay_step = epoch // decay_every
    total_steps = total_epochs // decay_every
    return (1 - decay_step / total_steps) ** power
