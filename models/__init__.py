import sys,os
dir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)
from LMF import LMF
from PerceptualLoss import LossNetwork as PerLoss
