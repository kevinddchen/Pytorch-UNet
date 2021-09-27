'''
Utility class that estimates time remaining.
'''

import time


class TimeEstimator:
  def __init__(self, num_iterations):
    self.start_t = time.time()
    self.t = time.time()
    self.N = num_iterations
    self.ave_delta_t = 0

  def update(self):
    '''Return time since last update and estimated time remaining (in seconds).'''
    self.N -= 1
    delta_t, self.t = time.time() - self.t, time.time()
    if self.ave_delta_t:
      self.ave_delta_t = .9*self.ave_delta_t + .1*delta_t   # exponential moving average
    else:
      self.ave_delta_t = delta_t
    remaining_t = self.N * self.ave_delta_t
    return delta_t, remaining_t

  def reset(self):
    '''Reset delta time.'''
    self.t = time.time()

  def total(self):
    '''Return total elapsed time since creation.'''
    return time.time() - self.start_t
    