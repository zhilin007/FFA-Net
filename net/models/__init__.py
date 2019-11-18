import sys,os
dir=os.path.abspath(os.path.dirname(__file__))
# print(dir)
sys.path.append(dir)
from rcan import RCAN
from net_valia import Net
from net_fa import Net_FA
from RCAN821 import *
from RCAN822_2 import *
from RCAN823_1 import *

from RCAN824_1 import *
from RPAN824_2 import *
from RPCAN824_3 import *
from RCAN824_5 import *
from RCAN824_7 import *
from  RCAN824_fusion1_G import *
from  RCAN824_fusion2_G import *



from RCAN825_D1 import RCAN825_D1
from RCAN825_D2 import RCAN825_D2
from RPAN825_2 import RPAN825_2
from RPAN825_F2G import RPAN825_F2G
from RPAN825_D1 import RPAN825_D1
from RPAN825_D2 import RPAN825_D2
from RPAN825_D2_F2G import RPAN825_D2_F2G
from PerceptualLoss import LossNetwork as PerLoss


from RCAN826_D3 import RCAN826_D3

from RN826_D1_CA_PA import RN826_D1_CA_PA
from RN826_D1_CA_PA_F2G import RN826_D1_CA_PA_F2G
from RN826_D1_CA_F2G import RN826_D1_CA_F2G
from RN826_D1_CA_PA_F3G import RN826_D1_CA_PA_F3G
from RN826_D1_CA_PA_F4G import RN826_D1_CA_PA_F4G

from RN827_D3_F2G import RN827_D3_F2G
from RCAN827_1 import RCAN827_1

from RN827_D1_CA_F2G_2 import RN827_D1_CA_F2G_2
from RCAN827_DiConv import RCAN827_DiConv
from RN827_D1_CA_F2G import RN827_D1_CA_F2G
from RN_Final import RN_Final

from FFANet import FFANet
from RN829_1 import RN829_1
from RN829_2 import RN829_2
from FFA import FFA
