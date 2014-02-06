#!/usr/bin/env python2

from core2D import *
import numpy as np
from collections import OrderedDict
import random
import math

# Input Data
settings = {
'batches' : 500,
'inactive' : 100,
'particles' : 1000,
'run_cmfd' : 'false'
}
n_densities = 1 # number of unique densities from hzp to 0.66
n_temps = 1  # number of unique fuel temperature linear from 600 to 1200
n_water = 25 # number of water materials, cmfd regions
cmfd = {
'power' : 17.674e6,
'flowrate' : 88.5145,
'inlet_enthalpy' : 1301740.,
'n_assemblies': 1,
'boron' : 975,
'thinner' : 1,
'thouter' : 1,
'interval' : 1000,
'begin':500,
'active_flush':70,
'feedback':'false'
}

# Global data
hzp_density = 0.73986            # Highest density
low_density = 0.66               # Lowest density
pin_pitch = 1.25984              # Pin pitch
assy_pitch = 21.50364            # Assembly pitch
rpv_OR = 251.9

assy_dict.update({
'A___5' : Assembly(enr = '3.1', bp = None),
'A___6' : Assembly(enr = '3.1', bp = '6W'),
'A___7' : Assembly(enr = '3.1', bp = None),
'A___8' : Assembly(enr = '3.1', bp = '6W'),
'A___9' : Assembly(enr = '3.1', bp = None),
'A__10' : Assembly(enr = '3.1', bp = '6W'),
'A__11' : Assembly(enr = '3.1', bp = None),

'B___3' : Assembly(enr = '3.1', bp = None),
'B___4' : Assembly(enr = '3.1', bp = None),
'B___5' : Assembly(enr = '3.1', bp = '16'),
'B___6' : Assembly(enr = '1.6', bp = None),
'B___7' : Assembly(enr = '3.1', bp = '20'),
'B___8' : Assembly(enr = '1.6', bp = None),
'B___9' : Assembly(enr = '3.1', bp = '20'),
'B__10' : Assembly(enr = '1.6', bp = None),
'B__11' : Assembly(enr = '3.1', bp = '16'),
'B__12' : Assembly(enr = '3.1', bp = None),
'B__13' : Assembly(enr = '3.1', bp = None),

'C___2' : Assembly(enr = '3.1', bp = None),
'C___3' : Assembly(enr = '3.1', bp = '15SW'),
'C___4' : Assembly(enr = '2.4', bp = '16'),
'C___5' : Assembly(enr = '1.6', bp = None),
'C___6' : Assembly(enr = '2.4', bp = '16'),
'C___7' : Assembly(enr = '1.6', bp = None),
'C___8' : Assembly(enr = '2.4', bp = '16'),
'C___9' : Assembly(enr = '1.6', bp = None),
'C__10' : Assembly(enr = '2.4', bp = '16'),
'C__11' : Assembly(enr = '1.6', bp = None),
'C__12' : Assembly(enr = '2.4', bp = '16'),
'C__13' : Assembly(enr = '3.1', bp = '15NW'),
'C__14' : Assembly(enr = '3.1', bp = None),

'D___2' : Assembly(enr = '3.1', bp = None),
'D___3' : Assembly(enr = '2.4', bp = '16'),
'D___4' : Assembly(enr = '2.4', bp = None),
'D___5' : Assembly(enr = '2.4', bp = '16'),
'D___6' : Assembly(enr = '1.6', bp = None),
'D___7' : Assembly(enr = '2.4', bp = '12'),
'D___8' : Assembly(enr = '1.6', bp = None),
'D___9' : Assembly(enr = '2.4', bp = '12'),
'D__10' : Assembly(enr = '1.6', bp = None),
'D__11' : Assembly(enr = '2.4', bp = '16'),
'D__12' : Assembly(enr = '2.4', bp = None),
'D__13' : Assembly(enr = '2.4', bp = '16'),
'D__14' : Assembly(enr = '3.1', bp = None),

'E___1' : Assembly(enr = '3.1', bp = None),
'E___2' : Assembly(enr = '3.1', bp = '16'),
'E___3' : Assembly(enr = '1.6', bp = None),
'E___4' : Assembly(enr = '2.4', bp = '16'),
'E___5' : Assembly(enr = '1.6', bp = None),
'E___6' : Assembly(enr = '2.4', bp = '12'),
'E___7' : Assembly(enr = '1.6', bp = None),
'E___8' : Assembly(enr = '2.4', bp = '12'),
'E___9' : Assembly(enr = '1.6', bp = None),
'E__10' : Assembly(enr = '2.4', bp = '12'),
'E__11' : Assembly(enr = '1.6', bp = None),
'E__12' : Assembly(enr = '2.4', bp = '16'),
'E__13' : Assembly(enr = '1.6', bp = None),
'E__14' : Assembly(enr = '3.1', bp = '16'),
'E__15' : Assembly(enr = '3.1', bp = None),

'F___1' : Assembly(enr = '3.1', bp = '6S'),
'F___2' : Assembly(enr = '1.6', bp = None),
'F___3' : Assembly(enr = '2.4', bp = '16'),
'F___4' : Assembly(enr = '1.6', bp = None),
'F___5' : Assembly(enr = '2.4', bp = '12'),
'F___6' : Assembly(enr = '1.6', bp = None),
'F___7' : Assembly(enr = '2.4', bp = '12'),
'F___8' : Assembly(enr = '1.6', bp = None),
'F___9' : Assembly(enr = '2.4', bp = '12'),
'F__10' : Assembly(enr = '1.6', bp = None),
'F__11' : Assembly(enr = '2.4', bp = '12'),
'F__12' : Assembly(enr = '1.6', bp = None),
'F__13' : Assembly(enr = '2.4', bp = '16'),
'F__14' : Assembly(enr = '1.6', bp = None),
'F__15' : Assembly(enr = '3.1', bp = '6N'),

'G___1' : Assembly(enr = '3.1', bp = None),
'G___2' : Assembly(enr = '3.1', bp = '20'),
'G___3' : Assembly(enr = '1.6', bp = None),
'G___4' : Assembly(enr = '2.4', bp = '12'),
'G___5' : Assembly(enr = '1.6', bp = None),
'G___6' : Assembly(enr = '2.4', bp = '12'),
'G___7' : Assembly(enr = '1.6', bp = None),
'G___8' : Assembly(enr = '2.4', bp = '16'),
'G___9' : Assembly(enr = '1.6', bp = None),
'G__10' : Assembly(enr = '2.4', bp = '12'),
'G__11' : Assembly(enr = '1.6', bp = None),
'G__12' : Assembly(enr = '2.4', bp = '12'),
'G__13' : Assembly(enr = '1.6', bp = None),
'G__14' : Assembly(enr = '3.1', bp = '20'),
'G__15' : Assembly(enr = '3.1', bp = None),

'H___1' : Assembly(enr = '3.1', bp = '6S'),
'H___2' : Assembly(enr = '1.6', bp = None),
'H___3' : Assembly(enr = '2.4', bp = '16'),
'H___4' : Assembly(enr = '1.6', bp = None),
'H___5' : Assembly(enr = '2.4', bp = '12'),
'H___6' : Assembly(enr = '1.6', bp = None),
'H___7' : Assembly(enr = '2.4', bp = '16'),
'H___8' : Assembly(enr = '1.6', bp = None),
'H___9' : Assembly(enr = '2.4', bp = '16'),
'H__10' : Assembly(enr = '1.6', bp = None),
'H__11' : Assembly(enr = '2.4', bp = '12'),
'H__12' : Assembly(enr = '1.6', bp = None),
'H__13' : Assembly(enr = '2.4', bp = '16'),
'H__14' : Assembly(enr = '1.6', bp = None),
'H__15' : Assembly(enr = '3.1', bp = '6N'),

'J___1' : Assembly(enr = '3.1', bp = None),
'J___2' : Assembly(enr = '3.1', bp = '20'),
'J___3' : Assembly(enr = '1.6', bp = None),
'J___4' : Assembly(enr = '2.4', bp = '12'),
'J___5' : Assembly(enr = '1.6', bp = None),
'J___6' : Assembly(enr = '2.4', bp = '12'),
'J___7' : Assembly(enr = '1.6', bp = None),
'J___8' : Assembly(enr = '2.4', bp = '16'),
'J___9' : Assembly(enr = '1.6', bp = None),
'J__10' : Assembly(enr = '2.4', bp = '12'),
'J__11' : Assembly(enr = '1.6', bp = None),
'J__12' : Assembly(enr = '2.4', bp = '12'),
'J__13' : Assembly(enr = '1.6', bp = None),
'J__14' : Assembly(enr = '3.1', bp = '20'),
'J__15' : Assembly(enr = '3.1', bp = None),

'K___1' : Assembly(enr = '3.1', bp = '6S'),
'K___2' : Assembly(enr = '1.6', bp = None),
'K___3' : Assembly(enr = '2.4', bp = '16'),
'K___4' : Assembly(enr = '1.6', bp = None),
'K___5' : Assembly(enr = '2.4', bp = '12'),
'K___6' : Assembly(enr = '1.6', bp = None),
'K___7' : Assembly(enr = '2.4', bp = '12'),
'K___8' : Assembly(enr = '1.6', bp = None),
'K___9' : Assembly(enr = '2.4', bp = '12'),
'K__10' : Assembly(enr = '1.6', bp = None),
'K__11' : Assembly(enr = '2.4', bp = '12'),
'K__12' : Assembly(enr = '1.6', bp = None),
'K__13' : Assembly(enr = '2.4', bp = '16'),
'K__14' : Assembly(enr = '1.6', bp = None),
'K__15' : Assembly(enr = '3.1', bp = '6N'),

'L___1' : Assembly(enr = '3.1', bp = None),
'L___2' : Assembly(enr = '3.1', bp = '16'),
'L___3' : Assembly(enr = '1.6', bp = None),
'L___4' : Assembly(enr = '2.4', bp = '16'),
'L___5' : Assembly(enr = '1.6', bp = None),
'L___6' : Assembly(enr = '2.4', bp = '12'),
'L___7' : Assembly(enr = '1.6', bp = None),
'L___8' : Assembly(enr = '2.4', bp = '12'),
'L___9' : Assembly(enr = '1.6', bp = None),
'L__10' : Assembly(enr = '2.4', bp = '12'),
'L__11' : Assembly(enr = '1.6', bp = None),
'L__12' : Assembly(enr = '2.4', bp = '16'),
'L__13' : Assembly(enr = '1.6', bp = None),
'L__14' : Assembly(enr = '3.1', bp = '16'),
'L__15' : Assembly(enr = '3.1', bp = None),

'M___2' : Assembly(enr = '3.1', bp = None),
'M___3' : Assembly(enr = '2.4', bp = '16'),
'M___4' : Assembly(enr = '2.4', bp = None),
'M___5' : Assembly(enr = '2.4', bp = '16'),
'M___6' : Assembly(enr = '1.6', bp = None),
'M___7' : Assembly(enr = '2.4', bp = '12'),
'M___8' : Assembly(enr = '1.6', bp = None),
'M___9' : Assembly(enr = '2.4', bp = '12'),
'M__10' : Assembly(enr = '1.6', bp = None),
'M__11' : Assembly(enr = '2.4', bp = '16'),
'M__12' : Assembly(enr = '2.4', bp = None),
'M__13' : Assembly(enr = '2.4', bp = '16'),
'M__14' : Assembly(enr = '3.1', bp = None),

'N___2' : Assembly(enr = '3.1', bp = None),
'N___3' : Assembly(enr = '3.1', bp = '15SE'),
'N___4' : Assembly(enr = '2.4', bp = '16'),
'N___5' : Assembly(enr = '1.6', bp = None),
'N___6' : Assembly(enr = '2.4', bp = '16'),
'N___7' : Assembly(enr = '1.6', bp = None),
'N___8' : Assembly(enr = '2.4', bp = '16'),
'N___9' : Assembly(enr = '1.6', bp = None),
'N__10' : Assembly(enr = '2.4', bp = '16'),
'N__11' : Assembly(enr = '1.6', bp = None),
'N__12' : Assembly(enr = '2.4', bp = '16'),
'N__13' : Assembly(enr = '3.1', bp = '15NE'),
'N__14' : Assembly(enr = '3.1', bp = None),

'P___3' : Assembly(enr = '3.1', bp = None),
'P___4' : Assembly(enr = '3.1', bp = None),
'P___5' : Assembly(enr = '3.1', bp = '16'),
'P___6' : Assembly(enr = '1.6', bp = None),
'P___7' : Assembly(enr = '3.1', bp = '20'),
'P___8' : Assembly(enr = '1.6', bp = None),
'P___9' : Assembly(enr = '3.1', bp = '20'),
'P__10' : Assembly(enr = '1.6', bp = None),
'P__11' : Assembly(enr = '3.1', bp = '16'),
'P__12' : Assembly(enr = '3.1', bp = None),
'P__13' : Assembly(enr = '3.1', bp = None),

'R___5' : Assembly(enr = '3.1', bp = None),
'R___6' : Assembly(enr = '3.1', bp = '6E'),
'R___7' : Assembly(enr = '3.1', bp = None),
'R___8' : Assembly(enr = '3.1', bp = '6E'),
'R___9' : Assembly(enr = '3.1', bp = None),
'R__10' : Assembly(enr = '3.1', bp = '6E'),
'R__11' : Assembly(enr = '3.1', bp = None),
})

assembly_map = """
{MOD__.u:>4} {MOD__.u:>4} {MOD__.u:>4} {MOD__.u:>4} {GRNWc.u:>4} {GR__N.u:>4} {GR__N.u:>4} {GR__N.u:>4} {GR__N.u:>4} {GR__N.u:>4} {GR__N.u:>4} {GR__N.u:>4} {GRNEc.u:>4} {MOD__.u:>4} {MOD__.u:>4} {MOD__.u:>4} {MOD__.u:>4} 
{MOD__.u:>4} {MOD__.u:>4} {GRNWc.u:>4} {GR__N.u:>4} {GR_NW.u:>4} {L___1.u:>4} {K___1.u:>4} {J___1.u:>4} {H___1.u:>4} {G___1.u:>4} {F___1.u:>4} {E___1.u:>4} {GR_NE.u:>4} {GR__N.u:>4} {GRNEc.u:>4} {MOD__.u:>4} {MOD__.u:>4}
{MOD__.u:>4} {GRNWc.u:>4} {GR_NW.u:>4} {N___2.u:>4} {M___2.u:>4} {L___2.u:>4} {K___2.u:>4} {J___2.u:>4} {H___2.u:>4} {G___2.u:>4} {F___2.u:>4} {E___2.u:>4} {D___2.u:>4} {C___2.u:>4} {GR_NE.u:>4} {GRNEc.u:>4} {MOD__.u:>4}
{MOD__.u:>4} {GR__W.u:>4} {P___3.u:>4} {N___3.u:>4} {M___3.u:>4} {L___3.u:>4} {K___3.u:>4} {J___3.u:>4} {H___3.u:>4} {G___3.u:>4} {F___3.u:>4} {E___3.u:>4} {D___3.u:>4} {C___3.u:>4} {B___3.u:>4} {GR__E.u:>4} {MOD__.u:>4}
{GRNWc.u:>4} {GR_NW.u:>4} {P___4.u:>4} {N___4.u:>4} {M___4.u:>4} {L___4.u:>4} {K___4.u:>4} {J___4.u:>4} {H___4.u:>4} {G___4.u:>4} {F___4.u:>4} {E___4.u:>4} {D___4.u:>4} {C___4.u:>4} {B___4.u:>4} {GR_NE.u:>4} {GRNEc.u:>4}
{GR__W.u:>4} {R___5.u:>4} {P___5.u:>4} {N___5.u:>4} {M___5.u:>4} {L___5.u:>4} {K___5.u:>4} {J___5.u:>4} {H___5.u:>4} {G___5.u:>4} {F___5.u:>4} {E___5.u:>4} {D___5.u:>4} {C___5.u:>4} {B___5.u:>4} {A___5.u:>4} {GR__E.u:>4}
{GR__W.u:>4} {R___6.u:>4} {P___6.u:>4} {N___6.u:>4} {M___6.u:>4} {L___6.u:>4} {K___6.u:>4} {J___6.u:>4} {H___6.u:>4} {G___6.u:>4} {F___6.u:>4} {E___6.u:>4} {D___6.u:>4} {C___6.u:>4} {B___6.u:>4} {A___6.u:>4} {GR__E.u:>4}
{GR__W.u:>4} {R___7.u:>4} {P___7.u:>4} {N___7.u:>4} {M___7.u:>4} {L___7.u:>4} {K___7.u:>4} {J___7.u:>4} {H___7.u:>4} {G___7.u:>4} {F___7.u:>4} {E___7.u:>4} {D___7.u:>4} {C___7.u:>4} {B___7.u:>4} {A___7.u:>4} {GR__E.u:>4}
{GR__W.u:>4} {R___8.u:>4} {P___8.u:>4} {N___8.u:>4} {M___8.u:>4} {L___8.u:>4} {K___8.u:>4} {J___8.u:>4} {H___8.u:>4} {G___8.u:>4} {F___8.u:>4} {E___8.u:>4} {D___8.u:>4} {C___8.u:>4} {B___8.u:>4} {A___8.u:>4} {GR__E.u:>4}
{GR__W.u:>4} {R___9.u:>4} {P___9.u:>4} {N___9.u:>4} {M___9.u:>4} {L___9.u:>4} {K___9.u:>4} {J___9.u:>4} {H___9.u:>4} {G___9.u:>4} {F___9.u:>4} {E___9.u:>4} {D___9.u:>4} {C___9.u:>4} {B___9.u:>4} {A___9.u:>4} {GR__E.u:>4}
{GR__W.u:>4} {R__10.u:>4} {P__10.u:>4} {N__10.u:>4} {M__10.u:>4} {L__10.u:>4} {K__10.u:>4} {J__10.u:>4} {H__10.u:>4} {G__10.u:>4} {F__10.u:>4} {E__10.u:>4} {D__10.u:>4} {C__10.u:>4} {B__10.u:>4} {A__10.u:>4} {GR__E.u:>4}
{GR__W.u:>4} {R__11.u:>4} {P__11.u:>4} {N__11.u:>4} {M__11.u:>4} {L__11.u:>4} {K__11.u:>4} {J__11.u:>4} {H__11.u:>4} {G__11.u:>4} {F__11.u:>4} {E__11.u:>4} {D__11.u:>4} {C__11.u:>4} {B__11.u:>4} {A__11.u:>4} {GR__E.u:>4}
{GRSWc.u:>4} {GR_SW.u:>4} {P__12.u:>4} {N__12.u:>4} {M__12.u:>4} {L__12.u:>4} {K__12.u:>4} {J__12.u:>4} {H__12.u:>4} {G__12.u:>4} {F__12.u:>4} {E__12.u:>4} {D__12.u:>4} {C__12.u:>4} {B__12.u:>4} {GR_SE.u:>4} {GRSEc.u:>4}
{MOD__.u:>4} {GR__W.u:>4} {P__13.u:>4} {N__13.u:>4} {M__13.u:>4} {L__13.u:>4} {K__13.u:>4} {J__13.u:>4} {H__13.u:>4} {G__13.u:>4} {F__13.u:>4} {E__13.u:>4} {D__13.u:>4} {C__13.u:>4} {B__13.u:>4} {GR__E.u:>4} {MOD__.u:>4}
{MOD__.u:>4} {GRSWc.u:>4} {GR_SW.u:>4} {N__14.u:>4} {M__14.u:>4} {L__14.u:>4} {K__14.u:>4} {J__14.u:>4} {H__14.u:>4} {G__14.u:>4} {F__14.u:>4} {E__14.u:>4} {D__14.u:>4} {C__14.u:>4} {GR_SE.u:>4} {GRSEc.u:>4} {MOD__.u:>4}
{MOD__.u:>4} {MOD__.u:>4} {GRSWc.u:>4} {GR__S.u:>4} {GR_SW.u:>4} {L__15.u:>4} {K__15.u:>4} {J__15.u:>4} {H__15.u:>4} {G__15.u:>4} {F__15.u:>4} {E__15.u:>4} {GR_SE.u:>4} {GR__S.u:>4} {GRSEc.u:>4} {MOD__.u:>4} {MOD__.u:>4}
{MOD__.u:>4} {MOD__.u:>4} {MOD__.u:>4} {MOD__.u:>4} {GRSWc.u:>4} {GR__S.u:>4} {GR__S.u:>4} {GR__S.u:>4} {GR__S.u:>4} {GR__S.u:>4} {GR__S.u:>4} {GR__S.u:>4} {GRSEc.u:>4} {MOD__.u:>4} {MOD__.u:>4} {MOD__.u:>4} {MOD__.u:>4}
"""

pin_lattice ="""
{nw:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {no:>4} {ne:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {pa:>4} {fp:>4} {fp:>4} {pb:>4} {fp:>4} {fp:>4} {pc:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {pd:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {pe:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {pf:>4} {fp:>4} {fp:>4} {pg:>4} {fp:>4} {fp:>4} {ph:>4} {fp:>4} {fp:>4} {pi:>4} {fp:>4} {fp:>4} {pj:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {pk:>4} {fp:>4} {fp:>4} {pl:>4} {fp:>4} {fp:>4} {pm:>4} {fp:>4} {fp:>4} {pn:>4} {fp:>4} {fp:>4} {po:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {pp:>4} {fp:>4} {fp:>4} {pq:>4} {fp:>4} {fp:>4} {pr:>4} {fp:>4} {fp:>4} {ps:>4} {fp:>4} {fp:>4} {pt:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {pu:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {pv:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {pw:>4} {fp:>4} {fp:>4} {px:>4} {fp:>4} {fp:>4} {py:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{we:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {fp:>4} {ea:>4}
{sw:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {so:>4} {se:>4}
"""

random.seed(974357)

def main():

    # Create all static materials
    create_static_materials()

    # Create surfaces
    create_surfaces()

    # Create static pins
    create_fuelpin('fuel16')
    create_fuelpin('fuel24')
    create_fuelpin('fuel31')
    create_bppin()
    create_gtpin()

    # Create baffle
    create_baffle()

    # Make assemblies
    create_assemblies()

    # Create core
    create_core()

    # Create cmfd
#   create_cmfd()

    # Write OpenMC files
    write_openmc_input()

def create_static_materials():

    # HZP Water Material
    mat_hzph2o = Material('h2o_hzp', 'HZP Water @ 0.73986 g/cc')
    mat_hzph2o.add_nuclide('B-10', '71c', '8.0042e-06')
    mat_hzph2o.add_nuclide('B-11', '71c', '3.2218e-05')
    mat_hzph2o.add_nuclide('H-1', '71c', '4.9457e-02')
    mat_hzph2o.add_nuclide('H-2', '71c', '7.4196e-06')
    mat_hzph2o.add_nuclide('O-16', '71c', '2.4672e-02')
    mat_hzph2o.add_nuclide('O-17', '71c', '6.0099e-05')
    mat_hzph2o.add_sab('lwtr', '15t')
    mat_hzph2o.add_color('0 0 255')
    mat_hzph2o.finalize()

    # Helium Material
    mat_hel = Material('he', 'Helium for Gap')
    mat_hel.add_nuclide('He-4', '71c', '2.4044e-04')
    mat_hel.add_color('255 218 185')
    mat_hel.finalize()

    # Air Material
    mat_air = Material('air', 'Air for Instrumentation Tubes')
    mat_air.add_nuclide('C-Nat', '71c', '6.8296e-09')
    mat_air.add_nuclide('O-16', '71c', '5.2864e-06')
    mat_air.add_nuclide('O-17', '71c', '1.2877e-08')
    mat_air.add_nuclide('N-14', '71c', '1.9681e-05')
    mat_air.add_nuclide('N-15', '71c', '7.1900e-08')
    mat_air.add_nuclide('Ar-36', '71c', '7.9414e-10')
    mat_air.add_nuclide('Ar-38', '71c', '1.4915e-10')
    mat_air.add_nuclide('Ar-40', '71c', '2.3506e-07')
    mat_air.add_color('255 255 255')
    mat_air.finalize()

    # Inconel Material
    mat_in = Material('in', 'Inconel 718 for Grids')
    mat_in.add_nuclide('Si-28', '71c', '5.6753e-04')
    mat_in.add_nuclide('Si-29', '71c', '2.8831e-05')
    mat_in.add_nuclide('Si-30', '71c', '1.9028e-05')
    mat_in.add_nuclide('Cr-50', '71c', '7.8239e-04')
    mat_in.add_nuclide('Cr-52', '71c', '1.5088e-02')
    mat_in.add_nuclide('Cr-53', '71c', '1.7108e-03')
    mat_in.add_nuclide('Cr-54', '71c', '4.2586e-04')
    mat_in.add_nuclide('Mn-55', '71c', '7.8201e-04')
    mat_in.add_nuclide('Fe-54', '71c', '1.4797e-03')
    mat_in.add_nuclide('Fe-56', '71c', '2.3229e-02')
    mat_in.add_nuclide('Fe-57', '71c', '5.3645e-04')
    mat_in.add_nuclide('Fe-58', '71c', '7.1392e-05')
    mat_in.add_nuclide('Ni-58', '71c', '2.9320e-02')
    mat_in.add_nuclide('Ni-60', '71c', '1.1294e-02')
    mat_in.add_nuclide('Ni-61', '71c', '4.9094e-04')
    mat_in.add_nuclide('Ni-62', '71c', '1.5653e-03')
    mat_in.add_nuclide('Ni-64', '71c', '3.9864e-04')
    mat_in.add_color('101 101 101')
    mat_in.finalize()

    # Stainless Steel Material
    mat_ss = Material('ss', 'Stainless Steel 304')
    mat_ss.add_nuclide('Si-28', '71c', '9.5274e-04')
    mat_ss.add_nuclide('Si-29', '71c', '4.8400e-05')
    mat_ss.add_nuclide('Si-30', '71c', '3.1943e-05')
    mat_ss.add_nuclide('Cr-50', '71c', '7.6778e-04')
    mat_ss.add_nuclide('Cr-52', '71c', '1.4806e-02')
    mat_ss.add_nuclide('Cr-53', '71c', '1.6789e-03')
    mat_ss.add_nuclide('Cr-54', '71c', '4.1791e-04')
    mat_ss.add_nuclide('Mn-55', '71c', '1.7604e-03')
    mat_ss.add_nuclide('Fe-54', '71c', '3.4620e-03')
    mat_ss.add_nuclide('Fe-56', '71c', '5.4345e-02')
    mat_ss.add_nuclide('Fe-57', '71c', '1.2551e-03')
    mat_ss.add_nuclide('Fe-58', '71c', '1.6703e-04')
    mat_ss.add_nuclide('Ni-58', '71c', '5.6089e-03')
    mat_ss.add_nuclide('Ni-60', '71c', '2.1605e-03')
    mat_ss.add_nuclide('Ni-61', '71c', '9.3917e-05')
    mat_ss.add_nuclide('Ni-62', '71c', '2.9945e-04')
    mat_ss.add_nuclide('Ni-64', '71c', '7.6261e-05')
    mat_ss.add_color('0 0 0')
    mat_ss.finalize()

    # Zircaloy Material
    mat_zr = Material('zr', 'Zircaloy-4')
    mat_zr.add_nuclide('O-16', '71c', '3.0743e-04')
    mat_zr.add_nuclide('O-17', '71c', '7.4887e-07')
    mat_zr.add_nuclide('Cr-50', '71c', '3.2962e-06')
    mat_zr.add_nuclide('Cr-52', '71c', '6.3564e-05')
    mat_zr.add_nuclide('Cr-53', '71c', '7.2076e-06')
    mat_zr.add_nuclide('Cr-54', '71c', '1.7941e-06')
    mat_zr.add_nuclide('Fe-54', '71c', '8.6699e-06')
    mat_zr.add_nuclide('Fe-56', '71c', '1.3610e-04')
    mat_zr.add_nuclide('Fe-57', '71c', '3.1431e-06')
    mat_zr.add_nuclide('Fe-58', '71c', '4.1829e-07')
    mat_zr.add_nuclide('Zr-90', '71c', '2.1827e-02')
    mat_zr.add_nuclide('Zr-91', '71c', '4.7600e-03')
    mat_zr.add_nuclide('Zr-92', '71c', '7.2758e-03')
    mat_zr.add_nuclide('Zr-94', '71c', '7.3734e-03')
    mat_zr.add_nuclide('Zr-96', '71c', '1.1879e-03')
    mat_zr.add_nuclide('Sn-112', '71c', '4.6735e-06')
    mat_zr.add_nuclide('Sn-114', '71c', '3.1799e-06')
    mat_zr.add_nuclide('Sn-115', '71c', '1.6381e-06')
    mat_zr.add_nuclide('Sn-116', '71c', '7.0055e-05')
    mat_zr.add_nuclide('Sn-117', '71c', '3.7003e-05')
    mat_zr.add_nuclide('Sn-118', '71c', '1.1669e-04')
    mat_zr.add_nuclide('Sn-119', '71c', '4.1387e-05')
    mat_zr.add_nuclide('Sn-120', '71c', '1.5697e-04')
    mat_zr.add_nuclide('Sn-122', '71c', '2.2308e-05')
    mat_zr.add_nuclide('Sn-124', '71c', '2.7897e-05')
    mat_zr.add_color('201 201 201')
    mat_zr.finalize()

    # UO2 at 1.6% enrichment Material
    mat_fuel24 = Material('fuel16', 'UO2 Fuel 1.6 w/o')
    mat_fuel24.add_nuclide('U-234', '71c', '3.0131e-06')
    mat_fuel24.add_nuclide('U-235', '71c', '3.7503e-04')
    mat_fuel24.add_nuclide('U-238', '71c', '2.2626e-02')
    mat_fuel24.add_nuclide('O-16', '71c', '4.5896e-02')
    mat_fuel24.add_nuclide('O-17', '71c', '1.1180e-04')
    mat_fuel24.add_color('142 35 35')
    mat_fuel24.finalize()

    # UO2 at 2.4% enrichment Material
    mat_fuel24 = Material('fuel24', 'UO2 Fuel 2.4 w/o')
    mat_fuel24.add_nuclide('U-234', '71c', '4.4843e-06')
    mat_fuel24.add_nuclide('U-235', '71c', '5.5815e-04')
    mat_fuel24.add_nuclide('U-238', '71c', '2.2408e-02')
    mat_fuel24.add_nuclide('O-16', '71c', '4.5829e-02')
    mat_fuel24.add_nuclide('O-17', '71c', '1.1164e-04')
    mat_fuel24.add_color('255 215 0')
    mat_fuel24.finalize()

    # UO2 at 3.1% enrichment Material
    mat_fuel24 = Material('fuel31', 'UO2 Fuel 2.4 w/o')
    mat_fuel24.add_nuclide('U-234', '71c', '5.7988e-06')
    mat_fuel24.add_nuclide('U-235', '71c', '7.2176e-04')
    mat_fuel24.add_nuclide('U-238', '71c', '2.2254e-02')
    mat_fuel24.add_nuclide('O-16', '71c', '4.5851e-02')
    mat_fuel24.add_nuclide('O-17', '71c', '1.1169e-04')
    mat_fuel24.add_color('0 0 128')
    mat_fuel24.finalize()

    # Borosilicate Glass Material
    mat_bsg = Material('bsg', 'Borosilicate Glass in BP Rod')
    mat_bsg.add_nuclide('B-10', '71c', '9.6506e-04')
    mat_bsg.add_nuclide('B-11', '71c', '3.9189e-03')
    mat_bsg.add_nuclide('O-16', '71c', '4.6511e-02')
    mat_bsg.add_nuclide('O-17', '71c', '1.1330e-04')
    mat_bsg.add_nuclide('Al-27', '71c', '1.7352e-03')
    mat_bsg.add_nuclide('Si-28', '71c', '1.6924e-02')
    mat_bsg.add_nuclide('Si-29', '71c', '8.5977e-04')
    mat_bsg.add_nuclide('Si-30', '71c', '5.6743e-04')
    mat_bsg.add_color('0 255 0')
    mat_bsg.finalize()

def create_surfaces():

    # Create fuel pin surfaces
    add_surface('fuelOR', 'z-cylinder', '0.0 0.0 0.392180', comment = 'Fuel Outer Radius')
    add_surface('cladIR', 'z-cylinder', '0.0 0.0 0.400050', comment = 'Clad Inner Radius')
    add_surface('cladOR', 'z-cylinder', '0.0 0.0 0.457200', comment = 'Clad Outer Radius')
    add_surface('springOR', 'z-cylinder', '0.0 0.0 0.06459', comment = 'Spring radius')

    # Create Guide Tube surfaces
    add_surface('gtIR', 'z-cylinder', '0.0 0.0 0.561340', comment = 'Guide Tube Inner Radius above Dashpot')
    add_surface('gtOR', 'z-cylinder', '0.0 0.0 0.601980', comment = 'Guide Tube Outer Radius above Dashpot')
    add_surface('gtIRdp', 'z-cylinder', '0.0 0.0 0.504190', comment = 'Guide Tube Inner Radius at Dashpot')
    add_surface('gtORdp', 'z-cylinder', '0.0 0.0 0.546100', comment = 'Guide Tube Outer Radius at Dashpot')

    # Burnable Poison surfaces
    add_surface('bpIR1', 'z-cylinder', '0.0 0.0 0.214000', comment = 'Burnable Absorber Rod Inner Radius 1')
    add_surface('bpIR2', 'z-cylinder', '0.0 0.0 0.230510', comment = 'Burnable Absorber Rod Inner Radius 2')
    add_surface('bpIR3', 'z-cylinder', '0.0 0.0 0.241300', comment = 'Burnable Absorber Rod Inner Radius 3')
    add_surface('bpIR4', 'z-cylinder', '0.0 0.0 0.426720', comment = 'Burnable Absorber Rod Inner Radius 4')
    add_surface('bpIR5', 'z-cylinder', '0.0 0.0 0.436880', comment = 'Burnable Absorber Rod Inner Radius 5')
    add_surface('bpIR6', 'z-cylinder', '0.0 0.0 0.483870', comment = 'Burnable Absorber Rod Inner Radius 6')

    # Baffle 
    baffle_width = 2.22250 
    baffle_left = assy_pitch/2.0 - baffle_width
    baffle_right = -assy_pitch/2.0 + baffle_width
    baffle_back = assy_pitch/2.0 - baffle_width 
    baffle_front = -assy_pitch/2.0 + baffle_width
    add_surface('baffleleft', 'x-plane', '{0}'.format(baffle_left), comment = 'Baffle Left')
    add_surface('baffleright', 'x-plane', '{0}'.format(baffle_right), comment = 'Baffle Right')
    add_surface('baffleback', 'y-plane', '{0}'.format(baffle_back), comment = 'Baffle Back')
    add_surface('bafflefront', 'y-plane', '{0}'.format(baffle_front), comment = 'Baffle Front')

    # Core surfaces
    box = 17.0*assy_pitch/2.0
    add_surface('core_left', 'x-plane', '{0}'.format(-box), comment = 'Core left surface')
    add_surface('core_right', 'x-plane', '{0}'.format(box), comment = 'Core right surface')
    add_surface('core_back', 'y-plane', '{0}'.format(-box), comment = 'Core back surface')
    add_surface('core_front', 'y-plane', '{0}'.format(box), comment = 'Core front surface')

    # Peripheral structures
    add_surface('core_barrelIR', 'z-cylinder', '0.0 0.0 187.960', comment = 'Core barrel inner radius')
    add_surface('core_barrelOR', 'z-cylinder', '0.0 0.0 193.675', bc = 'vacuum', comment = 'Core barrel outer radius')
    add_surface('shield_OR', 'z-cylinder', '0.0 0.0 199.39', comment = 'Shield panel outer radius')
    add_surface('shield_NWtoSE', 'plane', '1 {0} 0 0'.format(math.tan(math.pi/3)), comment = 'Shield panel cut plane')
    add_surface('shield_SEtoNW', 'plane', '1 {0} 0 0'.format(math.tan(math.pi/6)), comment = 'Shield panel cut plane')
    add_surface('shield_NEtoSW', 'plane', '1 {0} 0 0'.format(math.tan(math.pi/3)), comment = 'Shield panel cut plane')
    add_surface('shield_SWtoNE', 'plane', '1 {0} 0 0'.format(math.tan(math.pi/6)), comment = 'Shield panel cut plane')

def create_fuelpin(fuel_mat):

    # Fuel Pellet
    add_cell(fuel_mat, 
        surfaces = '-{0}'.format(surf_dict['fuelOR'].id), 
        universe = fuel_mat+'pin',
        material = mat_dict[fuel_mat].id,
        comment = 'Fuel pellet')

    # Gas Gap
    add_cell('gap_'+fuel_mat,
        surfaces = '{0} -{1}'.format(surf_dict['fuelOR'].id, surf_dict['cladIR'].id),
        universe = fuel_mat+'pin',
        material = mat_dict['he'].id,
        comment = 'Fuel pin gas gap')

    # Clad
    add_cell('clad_'+fuel_mat,
        surfaces = '{0}'.format(surf_dict['cladIR'].id),
        universe = fuel_mat+'pin',
        material = mat_dict['zr'].id,
        comment = 'Fuel pin clad')

def create_bppin():

    # Inner Air Region
    add_cell('air1BP',
        surfaces = '-{0}'.format(surf_dict['bpIR1'].id),
        universe = 'bp',
        material = mat_dict['air'].id,
        comment = 'BP inner air')

    # Inner Stainless Steel Region
    add_cell('ss1BP',
        surfaces = '{0} -{1}'.format(surf_dict['bpIR1'].id, surf_dict['bpIR2'].id),
        universe = 'bp',
        material = mat_dict['ss'].id,
        comment = 'BP inner stainless')

    # Middle Air Region
    add_cell('air2BP',
        surfaces = '{0} -{1}'.format(surf_dict['bpIR2'].id, surf_dict['bpIR3'].id),
        universe = 'bp',
        material = mat_dict['air'].id,
        comment = 'BP middle air')

    # Borosilicate Glass Region
    add_cell('bsgBP',
        surfaces = '{0} -{1}'.format(surf_dict['bpIR3'].id, surf_dict['bpIR4'].id),
        universe = 'bp',
        material = mat_dict['bsg'].id,
        comment = 'BP borosilicate')

    # Outer Air Region
    add_cell('air3BP',
        surfaces = '{0} -{1}'.format(surf_dict['bpIR4'].id, surf_dict['bpIR5'].id),
        universe = 'bp',
        material = mat_dict['air'].id,
        comment = 'BP outer air')

    # Outer Stainless Steel Region
    add_cell('ss2BP',
        surfaces = '{0} -{1}'.format(surf_dict['bpIR5'].id, surf_dict['bpIR6'].id),
        universe = 'bp',
        material = mat_dict['ss'].id,
        comment = 'BP outer stainless')

    # Moderator Region
    add_cell('modBP',
        surfaces = '{0} -{1}'.format(surf_dict['bpIR6'].id, surf_dict['gtIR'].id),
        universe = 'bp',
        material = mat_dict['h2o_hzp'].id,
        comment = 'BP moderator')

    # Tube Clad
    add_cell('cladBP',
        surfaces = '{0}'.format(surf_dict['gtIR'].id),
        universe = 'bp',
        material = mat_dict['zr'].id,
        comment = 'BP clad')

def create_gtpin():

    # Moderator Region
    add_cell('modGT',
        surfaces = '-{0}'.format(surf_dict['gtIR'].id),
        universe = 'gt',
        material = mat_dict['h2o_hzp'].id,
        comment = 'GT moderator')

    # Tube Clad
    add_cell('cladGT',
        surfaces = '{0}'.format(surf_dict['gtIR'].id),
        universe = 'gt',
        material = mat_dict['zr'].id,
        comment = 'GT clad')

def create_fuelpin_cell(cell_key, pin_key, water_key):

    # Fill static fuel pin
    add_cell('fuelpin_'+cell_key,
        surfaces = '-{0}'.format(surf_dict['cladOR'].id),
        universe = cell_key,
        fill = univ_dict[pin_key].id,
        comment = 'Fuel pin fill for coolant')

    # Fill in water coolant
    add_cell('cool_'+cell_key,
        surfaces = '{0}'.format(surf_dict['cladOR'].id),
        universe = cell_key,
        material = mat_dict[water_key].id,
        comment = 'Coolant around fuel pin')

def create_bppin_cell(cell_key, pin_key, water_key):

    # Fill static bp pin
    add_cell('bppin_'+cell_key,
        surfaces = '-{0}'.format(surf_dict['gtOR'].id),
        universe = cell_key,
        fill = univ_dict[pin_key].id,
        comment = 'BP pin fill for coolant')

    # Fill in water coolant
    add_cell('cool_'+cell_key,
        surfaces = '{0}'.format(surf_dict['gtOR'].id),
        universe = cell_key,
        material = mat_dict[water_key].id,
        comment = 'Coolant around BP pin')

def create_gtpin_cell(cell_key, pin_key, water_key):

    # Fill static gt pin
    add_cell('gtpin_'+cell_key,
        surfaces = '-{0}'.format(surf_dict['gtOR'].id),
        universe = cell_key,
        fill = univ_dict[pin_key].id, 
        comment = 'GT pin fill for coolant')

    # Fill in water coolant
    add_cell('cool_'+cell_key,
        surfaces = '{0}'.format(surf_dict['gtOR'].id),
        universe = cell_key,
        material = mat_dict[water_key].id,
        comment = 'Coolant around GT pin')

def create_baffle():

    # Moderator universe
    add_cell('water_mod',
        surfaces = '',
        universe = 'mod',
        material =  mat_dict['h2o_hzp'].id,
        comment = 'Moderator universe')

    bafmat = 'ss'
    # North baffle 
    add_cell('baffle_N',
        surfaces = '-{0}'.format(surf_dict['bafflefront'].id),
        universe = 'baffle_N',
        material = mat_dict[bafmat].id,
        comment = 'North {0} grid baffle'.format(bafmat))
    add_cell('baffle_N_mod',
        surfaces = '{0}'.format(surf_dict['bafflefront'].id),
        universe = 'baffle_N',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod north of {0} north grid baffle'.format(bafmat))

    # Northeast baffle 
    add_cell('baffle_NE',
        surfaces = '-{0} -{1}'.format(surf_dict['baffleright'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NE',
        material = mat_dict[bafmat].id,
        comment = 'Northeast {0} grid baffle'.format(bafmat))
    add_cell('baffle_NE_n',
        surfaces = '-{0} {1}'.format(surf_dict['baffleright'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NE',
        material = mat_dict[bafmat].id,
        comment = 'North of {0} northeast grid baffle'.format(bafmat))
    add_cell('baffle_NE_e',
        surfaces = '{0} -{1}'.format(surf_dict['baffleright'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NE',
        material = mat_dict[bafmat].id,
        comment = 'East of {0} northeast grid baffle'.format(bafmat))
    add_cell('baffle_NE_mod_ne',
        surfaces = '{0} {1}'.format(surf_dict['baffleright'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NE',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod northeast of {0} northeast grid baffle'.format(bafmat))

    # Northeast corner 
    add_cell('baffle_NEc',
        surfaces = '-{0} -{1}'.format(surf_dict['baffleright'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NEc',
        material = mat_dict[bafmat].id,
        comment = 'Northeast {0} grid baffle'.format(bafmat))
    add_cell('baffle_NE_mod_nc',
        surfaces = '-{0} {1}'.format(surf_dict['baffleright'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NEc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod north of {0} northeast grid baffle'.format(bafmat))
    add_cell('baffle_NE_mod_ec',
        surfaces = '{0} -{1}'.format(surf_dict['baffleright'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NEc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod east of {0} northeast grid baffle'.format(bafmat))
    add_cell('baffle_NE_mod_nec',
        surfaces = '{0} {1}'.format(surf_dict['baffleright'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NEc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod northeast of {0} northeast grid baffle'.format(bafmat))

    # East grid baffle 
    add_cell('baffle_E',
        surfaces = '-{0}'.format(surf_dict['baffleright'].id),
        universe = 'baffle_E',
        material = mat_dict[bafmat].id,
        comment = 'East {0} grid baffle'.format(bafmat))
    add_cell('baffle_E_mod',
        surfaces = '{0}'.format(surf_dict['baffleright'].id),
        universe = 'baffle_E',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod east of {0} east grid baffle'.format(bafmat))

    # Southeast baffle 
    add_cell('baffle_SE',
        surfaces = '-{0} {1}'.format(surf_dict['baffleright'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SE',
        material = mat_dict[bafmat].id,
        comment = 'Southeast {0} grid baffle'.format(bafmat))
    add_cell('baffle_SE_s',
        surfaces = '-{0} -{1}'.format(surf_dict['baffleright'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SE',
        material = mat_dict[bafmat].id,
        comment = 'South of {0} southeast grid baffle'.format(bafmat))
    add_cell('baffle_SE_e',
        surfaces = '{0} {1}'.format(surf_dict['baffleright'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SE',
        material = mat_dict[bafmat].id,
        comment = 'East of {0} southeast grid baffle'.format(bafmat))
    add_cell('baffle_SE_mod_se',
        surfaces = '{0} -{1}'.format(surf_dict['baffleright'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SE',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod southeast of {0} southeast grid baffle'.format(bafmat))

    # Southeast baffle corner
    add_cell('baffle_SEc',
        surfaces = '-{0} {1}'.format(surf_dict['baffleright'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SEc',
        material = mat_dict[bafmat].id,
        comment = 'Southeast {0} grid baffle'.format(bafmat))
    add_cell('baffle_SE_mod_sc',
        surfaces = '-{0} -{1}'.format(surf_dict['baffleright'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SEc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod south of {0} southeast grid baffle'.format(bafmat))
    add_cell('baffle_SE_mod_ec',
        surfaces = '{0} {1}'.format(surf_dict['baffleright'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SEc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod east of {0} southeast grid baffle'.format(bafmat))
    add_cell('baffle_SE_mod_sec',
        surfaces = '{0} -{1}'.format(surf_dict['baffleright'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SEc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod southeast of {0} southeast grid baffle'.format(bafmat))

    # South baffle 
    add_cell('baffle_S',
        surfaces = '{0}'.format(surf_dict['baffleback'].id),
        universe = 'baffle_S',
        material = mat_dict[bafmat].id,
        comment = 'South {0} grid baffle'.format(bafmat))
    add_cell('baffle_S_mod',
        surfaces = '-{0}'.format(surf_dict['baffleback'].id),
        universe = 'baffle_S',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod south of {0} south grid baffle'.format(bafmat))

    # Southwest baffle 
    add_cell('baffle_SW',
        surfaces = '{0} {1}'.format(surf_dict['baffleleft'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SW',
        material = mat_dict[bafmat].id,
        comment = 'Southwest {0} grid baffle'.format(bafmat))
    add_cell('baffle_SW_s',
        surfaces = '{0} -{1}'.format(surf_dict['baffleleft'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SW',
        material = mat_dict[bafmat].id,
        comment = 'South of {0} southwest grid baffle'.format(bafmat))
    add_cell('baffle_SW_w',
        surfaces = '-{0} {1}'.format(surf_dict['baffleleft'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SW',
        material = mat_dict[bafmat].id,
        comment = 'West of {0} southwest grid baffle'.format(bafmat))
    add_cell('baffle_SW_mod_sw',
        surfaces = '-{0} -{1}'.format(surf_dict['baffleleft'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SW',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod southwest of {0} southwest grid baffle'.format(bafmat))

    # Southwest baffle corner 
    add_cell('baffle_SWc',
        surfaces = '{0} {1}'.format(surf_dict['baffleleft'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SWc',
        material = mat_dict[bafmat].id,
        comment = 'Southwest {0} grid baffle'.format(bafmat))
    add_cell('baffle_SW_mod_sc',
        surfaces = '{0} -{1}'.format(surf_dict['baffleleft'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SWc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod south of {0} southwest grid baffle'.format(bafmat))
    add_cell('baffle_SW_mod_wc',
        surfaces = '-{0} {1}'.format(surf_dict['baffleleft'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SWc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod west of {0} southwest grid baffle'.format(bafmat))
    add_cell('baffle_SW_mod_swc',
        surfaces = '-{0} -{1}'.format(surf_dict['baffleleft'].id, surf_dict['baffleback'].id),
        universe = 'baffle_SWc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod southwest of {0} southwest grid baffle'.format(bafmat))

    # West baffle 
    add_cell('baffle_W',
        surfaces = '{0}'.format(surf_dict['baffleleft'].id),
        universe = 'baffle_W',
        material = mat_dict[bafmat].id,
        comment = 'West {0} grid baffle'.format(bafmat))
    add_cell('baffle_W_mod',
        surfaces = '-{0}'.format(surf_dict['baffleleft'].id),
        universe = 'baffle_W',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod west of {0} west grid baffle'.format(bafmat))

    # Northwest baffle
    add_cell('baffle_NW',
        surfaces = '{0} -{1}'.format(surf_dict['baffleleft'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NW',
        material = mat_dict[bafmat].id,
        comment = 'Northwest {0} grid baffle'.format(bafmat))
    add_cell('baffle_NW_s',
        surfaces = '{0} {1}'.format(surf_dict['baffleleft'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NW',
        material = mat_dict[bafmat].id,
        comment = 'North of {0} northwest grid baffle'.format(bafmat))
    add_cell('baffle_NW_w',
        surfaces = '-{0} -{1}'.format(surf_dict['baffleleft'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NW',
        material = mat_dict[bafmat].id,
        comment = 'West of {0} northwest grid baffle'.format(bafmat))
    add_cell('baffle_NW_mod_sw',
        surfaces = '-{0} {1}'.format(surf_dict['baffleleft'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NW',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod northwest of {0} northest grid baffle'.format(bafmat))

    # Northwest baffle corner
    add_cell('baffle_NWc',
        surfaces = '{0} -{1}'.format(surf_dict['baffleleft'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NWc',
        material = mat_dict[bafmat].id,
        comment = 'Northwest {0} grid baffle'.format(bafmat))
    add_cell('baffle_NW_mod_sc',
        surfaces = '{0} {1}'.format(surf_dict['baffleleft'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NWc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod north of {0} northwest grid baffle'.format(bafmat))
    add_cell('baffle_NW_mod_wc',
        surfaces = '-{0} -{1}'.format(surf_dict['baffleleft'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NWc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod west of {0} northwest grid baffle'.format(bafmat))
    add_cell('baffle_NW_mod_swc',
        surfaces = '-{0} {1}'.format(surf_dict['baffleleft'].id, surf_dict['bafflefront'].id),
        universe = 'baffle_NWc',
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod northwest of {0} northest grid baffle'.format(bafmat))

    # Add all of these ids to assemblies for core construction
    assy_dict.update({
       'MOD__' : Assembly(u = univ_dict['mod'].id),
       'GR__N' : Assembly(u = univ_dict['baffle_N'].id),
       'GR_NE' : Assembly(u = univ_dict['baffle_NE'].id),
       'GRNEc' : Assembly(u = univ_dict['baffle_NEc'].id),
       'GR__E' : Assembly(u = univ_dict['baffle_E'].id),
       'GR_SE' : Assembly(u = univ_dict['baffle_SE'].id),
       'GRSEc' : Assembly(u = univ_dict['baffle_SEc'].id),
       'GR__S' : Assembly(u = univ_dict['baffle_S'].id),
       'GR_SW' : Assembly(u = univ_dict['baffle_SW'].id),
       'GRSWc' : Assembly(u = univ_dict['baffle_SWc'].id),
       'GR__W' : Assembly(u = univ_dict['baffle_W'].id),
       'GR_NW' : Assembly(u = univ_dict['baffle_NW'].id),
       'GRNWc' : Assembly(u = univ_dict['baffle_NWc'].id)})

def create_lattice(lat_key, fuel_key, bp_key, gt_key, it_key, bp_config=None, comment = None):

    # Set up pin ids
    fuel_id = univ_dict[fuel_key].id
    bp_id = univ_dict[bp_key].id
    gt_id = univ_dict[gt_key].id
    it_id = univ_dict[it_key].id

    # No grid present in this model so set these all to water
    no_id = univ_dict['mod'].id
    ne_id = univ_dict['mod'].id
    ea_id = univ_dict['mod'].id
    se_id = univ_dict['mod'].id
    so_id = univ_dict['mod'].id
    sw_id = univ_dict['mod'].id
    we_id = univ_dict['mod'].id
    nw_id = univ_dict['mod'].id

    # Calculate coordinates
    lleft = -19.0*pin_pitch / 2.0

    # Set defaults
    univs = { 'fp':fuel_id,
        'pa' : gt_id,
        'pb' : gt_id,
        'pc' : gt_id,
        'pd' : gt_id,
        'pe' : gt_id,
        'pf' : gt_id,
        'pg' : gt_id,
        'ph' : gt_id,
        'pi' : gt_id,
        'pj' : gt_id,
        'pk' : gt_id,
        'pl' : gt_id,
        'pm' : it_id,
        'pn' : gt_id,
        'po' : gt_id,
        'pp' : gt_id,
        'pq' : gt_id,
        'pr' : gt_id,
        'ps' : gt_id,
        'pt' : gt_id,
        'pu' : gt_id,
        'pv' : gt_id,
        'pw' : gt_id,
        'px' : gt_id,
        'py' : gt_id,
        'no' : no_id,
        'ne' : ne_id,
        'ea' : ea_id,
        'se' : se_id,
        'so' : so_id,
        'sw' : sw_id,
        'we' : we_id,
        'nw' : nw_id} 

    # Perform BP configurations
    if bp_config == None:
        pass
    elif bp_config == '6N':
        univs.update({
            'pa' : bp_id,
	    'pc' : bp_id,
            'pd' : bp_id,
            'pe' : bp_id,
            'pf' : bp_id,
            'pj' : bp_id})
    elif bp_config == '6E':
        univs.update({
            'pj' : bp_id,
	    'pt' : bp_id,
            'pe' : bp_id,
            'pv' : bp_id,
            'pc' : bp_id,
            'py' : bp_id})
    elif bp_config == '6S':
        univs.update({
            'py' : bp_id,
	    'pw' : bp_id,
            'pv' : bp_id,
            'pu' : bp_id,
            'pt' : bp_id,
            'pp' : bp_id})
    elif bp_config == '6W':
        univs.update({
            'pp' : bp_id,
	    'pf' : bp_id,
            'pu' : bp_id,
            'pd' : bp_id,
            'pw' : bp_id,
            'pa' : bp_id})
    elif bp_config == '12':
        univs.update({
            'pa' : bp_id,
	    'pc' : bp_id,
            'pd' : bp_id,
            'pe' : bp_id,
            'pf' : bp_id,
            'pj' : bp_id,
            'pp' : bp_id,
	    'pt' : bp_id,
            'pu' : bp_id,
            'pv' : bp_id,
            'pw' : bp_id,
            'py' : bp_id})
    elif bp_config == '15NW':
        univs.update({
            'pa' : bp_id,
	    'pb' : bp_id,
            'pc' : bp_id,
            'pd' : bp_id,
            'pf' : bp_id,
            'pg' : bp_id,
            'ph' : bp_id,
	    'pi' : bp_id,
            'pk' : bp_id,
            'pl' : bp_id,
            'pn' : bp_id,
            'pp' : bp_id,
            'pq' : bp_id,
            'pr' : bp_id,
            'ps' : bp_id})
    elif bp_config == '15NE':
        univs.update({
            'pa' : bp_id,
	    'pb' : bp_id,
            'pc' : bp_id,
            'pe' : bp_id,
            'pg' : bp_id,
            'ph' : bp_id,
            'pi' : bp_id,
	    'pj' : bp_id,
            'pl' : bp_id,
            'pn' : bp_id,
            'po' : bp_id,
            'pq' : bp_id,
            'pr' : bp_id,
            'ps' : bp_id,
            'pt' : bp_id})
    elif bp_config == '15SE':
        univs.update({
            'pg' : bp_id,
	    'ph' : bp_id,
            'pi' : bp_id,
            'pj' : bp_id,
            'pl' : bp_id,
            'pn' : bp_id,
            'po' : bp_id,
	    'pq' : bp_id,
            'pr' : bp_id,
            'ps' : bp_id,
            'pt' : bp_id,
            'pv' : bp_id,
            'pw' : bp_id,
            'px' : bp_id,
            'py' : bp_id})
    elif bp_config == '15SW':
        univs.update({
            'pf' : bp_id,
	    'pg' : bp_id,
            'ph' : bp_id,
            'pi' : bp_id,
            'pk' : bp_id,
            'pl' : bp_id,
            'pn' : bp_id,
	    'pp' : bp_id,
            'pq' : bp_id,
            'pr' : bp_id,
            'ps' : bp_id,
            'pu' : bp_id,
            'pw' : bp_id,
            'px' : bp_id,
            'py' : bp_id})
    elif bp_config == '16':
        univs.update({
            'pa' : bp_id,
	    'pb' : bp_id,
            'pc' : bp_id,
            'pd' : bp_id,
            'pe' : bp_id,
            'pf' : bp_id,
            'pj' : bp_id,
            'pk' : bp_id,
	    'po' : bp_id,
            'pp' : bp_id,
            'pt' : bp_id,
            'pu' : bp_id,
            'pv' : bp_id,
            'pw' : bp_id,
            'px' : bp_id,
            'py' : bp_id})
    elif bp_config == '20':
        univs.update({
            'pa' : bp_id,
            'pb' : bp_id,
	    'pc' : bp_id,
	    'pd' : bp_id,
            'pe' : bp_id,
            'pf' : bp_id,
            'pg' : bp_id,
            'pi' : bp_id,
            'pj' : bp_id,
            'pk' : bp_id,
	    'po' : bp_id,
            'pp' : bp_id,
            'pq' : bp_id,
            'ps' : bp_id,
            'pt' : bp_id,
            'pu' : bp_id,
            'pv' : bp_id,
            'pw' : bp_id,
            'px' : bp_id,
            'py' : bp_id})
    else:
        raise Exception('BP Configuration {0} doesnt exist.'.format(bp_config))

    # Make lattice
    add_lattice(lat_key,
        dimension = '19 19',
        lower_left = '{0} {0}'.format(lleft),
        width = '{0} {0}'.format(pin_pitch),
        universes = pin_lattice.format(**univs),
        comment = comment)

def create_assemblies():

    # Begin loop around fuel assemblies
    for assy in assy_dict.keys():

        # Check for non fuel assembly
        if assy_dict[assy].enr == None:
            continue

        # Sample density
        density = random.uniform(low_density, hzp_density)
        color = -156.0/(hzp_density - low_density) * \
                (density - hzp_density)
        assy_dict[assy].add_density(density)

        # Create a water material with that density
        create_water_material('{0} coolant'.format(assy), density, color)

        # Create pin cells
        enr = assy_dict[assy].enr
        if enr == '1.6':
            fuelpin = 'fuel16pin'
        elif enr == '2.4':
            fuelpin = 'fuel24pin'
        elif enr == '3.1':
            fuelpin = 'fuel31pin'
        else:
            raise Exception ('Fuel enrichment doesnt exist')
        create_fuelpin_cell('{0} fuelpin'.format(assy), fuelpin, '{0} coolant'.format(assy)) 
        create_bppin_cell('{0} bppin'.format(assy), 'bp', '{0} coolant'.format(assy))
        create_gtpin_cell('{0} gtpin'.format(assy), 'gt', '{0} coolant'.format(assy)) 

        # Create lattice
        create_lattice('{0} lattice'.format(assy), '{0} fuelpin'.format(assy), '{0} bppin'.format(assy),
                       '{0} gtpin'.format(assy), '{0} gtpin'.format(assy), bp_config = assy_dict[assy].bp,
                       comment = '{0} lattice'.format(assy))

        # Create cell to put lattice in
        add_cell('{0} lattice fill'.format(assy),
            surfaces = '',
            universe = '{0} assembly'.format(assy),
            fill = lat_dict['{0} lattice'.format(assy)].id,
            comment = '{0} Assembly Cell'.format(assy))
        assy_dict['{0}'.format(assy)].add_universe(univ_dict['{0} assembly'.format(assy)].id)

def create_core():

    # Create Core Lattice
    lleft = -17.0*assy_pitch/2.0
    add_lattice('Core Lattice',
        dimension = '17 17',
        lower_left = '{0} {0}'.format(lleft),
        width = '{0} {0}'.format(assy_pitch),
        universes = assembly_map.format(**assy_dict),
        comment = 'Core Lattice')
     
    # Create core fill cell
    add_cell('core',
        surfaces = '{0} -{1} {2} -{3} -{4}'.format(surf_dict['core_left'].id, surf_dict['core_right'].id,
                                        surf_dict['core_back'].id, surf_dict['core_front'].id, surf_dict['core_barrelIR'].id),
        fill = lat_dict['Core Lattice'].id,
        comment = 'Core fill')

    # Add moderator outside of core
    add_cell('coremodN',
        surfaces = '{0} {1} -{2} -{3}'.format(surf_dict['core_front'].id, surf_dict['core_left'].id, surf_dict['core_right'].id, surf_dict['core_barrelIR'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Moderator around N of core')
    add_cell('coremodS',
        surfaces = '-{0} {1} -{2} -{3}'.format(surf_dict['core_back'].id, surf_dict['core_left'].id, surf_dict['core_right'].id, surf_dict['core_barrelIR'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Moderator around S of core')
    add_cell('coremodNES',
        surfaces = '{0} -{1}'.format(surf_dict['core_right'].id, surf_dict['core_barrelIR'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Moderator around E of core')
    add_cell('coremodSWN',
        surfaces = '-{0} -{1}'.format(surf_dict['core_left'].id, surf_dict['core_barrelIR'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Moderator around W of core')

    # Add in core barrel
    add_cell('core_barrel',
        surfaces = '{0} -{1}'.format(surf_dict['core_barrelIR'].id, surf_dict['core_barrelOR'].id),
        material = mat_dict['ss'].id,
        comment = 'Core Barrel')
     
    # Plot core
    add_plot('plot_axial',
        origin = '0.0 0.0 0.0',
        width = '{0} {0}'.format(2.0*rpv_OR),
        basis = 'xy',
        pixels = '3000 3000',
        filename = 'core_radial')

def create_water_material(key, water_density, color=None):

    # Avagadros Number
    NA = 0.60221415

    # Molar masses of nuclides
    MH1 = 1.0078250
    MH2 = 2.0141018
    MB10 = 10.0129370
    MB11 = 11.0093054
    MO16 = 15.9949146196
    MO17 = 16.9991317
    MO18 = 17.999161

    # Molar masses of natural elements
    MH = 1.00794
    MB = 10.811
    MO = 15.9994

    # Natrual abundances
    aH1 = 0.99985
    aH2 = 0.00015
    aB10 = 0.199
    aB11 = 0.801
    aO16 = 0.99757
    aO17 = 0.00038
    aO18 = 0.00205

    # Boron info
    ppm = 975
    wBph2o = ppm * 10**-6

    # Molecular mass of pure water
    Mh2o = 2*MH + MO

    # Compute number density of pure water
    Nh2o = water_density * NA / Mh2o

    # Compute mass density of borated water
    rhoh2oB = water_density / (1.0 - wBph2o)

    # Compute number densities of elements
    NB = wBph2o * rhoh2oB * NA / MB
    NH = 2.0 * Nh2o
    NO = Nh2o

    # Compute isotopic number densities
    NB10 = aB10 * NB
    NB11 = aB11 * NB
    NH1 = aH1 * NH
    NH2 = aH2 * NH
    NO16 = aO16 * NO
    NO17 = aO17 * NO
    NO18 = aO18 * NO

    mat_h2o = Material(key, 'HZP Water @ {0} g/cc'.format(water_density))
    mat_h2o.add_nuclide('B-10', '71c', str(NB10))
    mat_h2o.add_nuclide('B-11', '71c', str(NB11))
    mat_h2o.add_nuclide('H-1', '71c', str(NH1))
    mat_h2o.add_nuclide('H-2', '71c', str(NH2))
    mat_h2o.add_nuclide('O-16', '71c', str(NO16))
    mat_h2o.add_nuclide('O-17', '71c', str(NO17 + NO18))
    mat_h2o.add_sab('lwtr', '15t')
    if color != None:
        mat_h2o.add_color('{0} {0} 255'.format(int(color)))
    mat_h2o.finalize()

def create_cmfd():

    # Put mesh info in
    dz = (axial_surfaces['taf'] - axial_surfaces['baf'])/float(n_water)
    mx = -assy_pitch/2.0
    my = -assy_pitch/2.0
    mz = axial_surfaces['baf'] - dz
    px = assy_pitch/2.0
    py = assy_pitch/2.0
    pz = axial_surfaces['taf'] + dz
    cmfd.update({'lower_left': '{0} {1} {2}'.format(mx, my, mz)})
    cmfd.update({'upper_right':'{0} {1} {2}'.format(px, py, pz)})
    cmfd.update({'dimension':'1 1 {0}'.format(n_water + 2)})
    map_str = "1\n"
    for i in range(n_water):
        map_str += "2\n"
    map_str += "1\n"
    cmfd.update({'map':map_str})

    # Put water id map in
    water_str = "0\n"
    current_id = -1 
    for key in axial_dict.keys():
        axial = axial_dict[key]
        if axial.water_idx != current_id:
            current_id = axial.water_idx
            water_str += "{0}\n".format(mat_dict['water_{0}'.format(current_id)].id)
    water_str += "0"
    cmfd.update({'water_map':water_str})

    # Put enrichment and bp map together
    enr_str = "0.0\n"
    bp_str = "0\n"
    for i in range(n_water):
        enr_str += "2.4\n"
        bp_str += "12\n"
    enr_str += "0.0"
    bp_str += "0"
    cmfd.update({'enr_map':enr_str})
    cmfd.update({'bp_map':bp_str})

    # Fuel temperature
    temp_low = 600.0
    temp_high = 1200.0
    try:
        temp_slope = (temp_high - temp_low)/float(n_temps - 1)
    except ZeroDivisionError:
        temp_slope = 0.0
    temp_str = "{0}\n".format(temp_low)
    for i in range(n_water):
        temp = temp_slope*i + temp_low
        temp_str += "{0}\n".format(temp)
    temp_str += "{0}".format(temp_low)
    cmfd.update({'fuel_temp':temp_str})

    # Density
    density_str = "0.0\n"
    current_id = -1 
    for key in axial_dict.keys():
        axial = axial_dict[key]
        if axial.water_idx != current_id:
            current_id = axial.water_idx
            density_str += "{0}\n".format(axial.cool_rho)
    density_str += "0.0"
    cmfd.update({'density':density_str})

    # Normalization
    cmfd.update({'norm':n_water})

def write_openmc_input():

############ Geometry File ##############

    # Heading info
    geo_str = ""
    geo_str += \
"""<?xml version="1.0" encoding="UTF-8"?>\n<geometry>\n\n"""

    # Write out surfaces
    for item in surf_dict.keys():
        geo_str += surf_dict[item].write_xml()

    # Write out cells
    geo_str += "\n"
    for item in cell_dict.keys():
        geo_str += cell_dict[item].write_xml()

    # Write out lattices
    geo_str += "\n"
    for item in lat_dict.keys():
        geo_str += lat_dict[item].write_xml()

    # Write out footer info
    geo_str += \
"""\n</geometry>"""
    with open('geometry.xml','w') as fh:
        fh.write(geo_str)

############ Materials File ##############

    # Heading info
    mat_str = ""
    mat_str += \
"""<?xml version="1.0" encoding="UTF-8"?>\n<materials>\n\n"""

    # Write out materials
    for item in mat_dict.keys():
        mat_str += mat_dict[item].write_xml()
        mat_str += "\n"

    # Write out footer info
    mat_str += \
"""</materials>"""
    with open('materials.xml','w') as fh:
        fh.write(mat_str)

############ Settings File ##############

    settings.update({
'xbot' : -17*assy_pitch/2.0,
'ybot' : -17*assy_pitch/2.0,
'zbot' : 0.0,
'xtop' : 17*assy_pitch/2.0,
'ytop' : 17*assy_pitch/2.0,
'ztop' : 10.0,
'entrX' : 17,
'entrY' : 17,
'entrZ' : 1
    })

    set_str = """<?xml version="1.0" encoding="UTF-8"?>
<settings>

  <!-- Parameters for criticality calculation -->
  <eigenvalue batches="{batches}" inactive="{inactive}" particles="{particles}" />

  <!-- Starting source -->
  <source>
    <space type="box">
      <parameters>{xbot} {ybot} {zbot} {xtop} {ytop} {ztop}</parameters>
    </space>
  </source>
  
  <!-- Shannon Entropy -->
  <entropy>
    <dimension> {entrX} {entrY} {entrZ} </dimension>
    <lower_left> {xbot} {ybot} {zbot} </lower_left>
    <upper_right> {xtop} {ytop} {ztop} </upper_right>
  </entropy>

  <!-- Run CMFD -->
  <run_cmfd> {run_cmfd} </run_cmfd>

</settings>""".format(**settings)
    with open('settings.xml','w') as fh:
        fh.write(set_str)

############ Plots File ##############

    plot_str = """<?xml version="1.0" encoding="UTF-8"?>\n"""
    plot_str += """<plots>\n"""
    for item in plot_dict.keys():
        plot_str += plot_dict[item].write_xml()
        plot_str += "\n"
    plot_str += """</plots>""".format(x = assy_pitch+5, y = assy_pitch+5)
    with open('plots.xml','w') as fh:
        fh.write(plot_str)
    return
############ CMFD File ###############

    cmfd_str = """<?xml version="1.0" encoding="UTF-8"?>
<cmfd>

  <!-- This file auto-generated by beavrs.py  -->
  <mesh>
    <lower_left>{lower_left}</lower_left>
    <upper_right>{upper_right}</upper_right>
    <dimension>{dimension}</dimension>
    <albedo>1.0 1.0 1.0 1.0 0.0 0.0</albedo>
    <map>
{map}
    </map>
    <energy>0.0 0.625e-6 20.0</energy>
  </mesh>
  
  <thermal>
    <water_map>
{water_map} 
    </water_map>

    <enr_map>
{enr_map}
    </enr_map>

    <bp_map>
{bp_map}
    </bp_map>

    <fuel_temp>
{fuel_temp}
    </fuel_temp>

    <density>
{density}
    </density>

    <core_power> {power} </core_power>
    <core_flowrate> {flowrate} </core_flowrate>
    <inlet_enthalpy> {inlet_enthalpy} </inlet_enthalpy>
    <n_assemblies> {n_assemblies} </n_assemblies>
    <boron> {boron} </boron>
    <maxthinner> {thinner} </maxthinner>
    <maxthouter> {thouter} </maxthouter>
    <interval> {interval} </interval>
  </thermal>

  <begin> {begin} </begin>
  <active_flush> {active_flush} </active_flush>
  <feedback> {feedback} </feedback>
  <norm> {norm} </norm>
  <downscatter> true </downscatter>
  <power_monitor> true </power_monitor>
</cmfd>
""".format(**cmfd)
    with open('cmfd.xml','w') as fh:
        fh.write(cmfd_str)

if __name__ == '__main__':
    main()
