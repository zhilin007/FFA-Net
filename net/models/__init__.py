import sys,os
dir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)
from FFA import FFA
from PerceptualLoss import LossNetwork as PerLoss
