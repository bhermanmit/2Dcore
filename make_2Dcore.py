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
'run_cmfd' : 'true'
}
cmfd = {
'power' : '3411e6',
'flowrate' : '17083.3',
'inlet_enthalpy' : '1301740.0',
'n_assemblies': '193',
'boron' : '975',
'thinner' : '1',
'thouter' : '1',
'interval' : '1000',
'begin':'500',
'active_flush':'70',
'feedback':'false'
}

# Global data
hzp_density = 0.73986            # Highest density
low_density = 0.66               # Lowest density
hzp_fueltemp = 600.0             # Lowest fuel temp 
high_fueltemp = 1200.0           # Highest fuel temp
pin_pitch = 1.25984              # Pin pitch
assy_pitch = 21.50364            # Assembly pitch
rpv_OR = 251.9

assy_dict.update({'A___5' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'A___6' : Assembly(enr = '3.1', bp = '6W')})
assy_dict.update({'A___7' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'A___8' : Assembly(enr = '3.1', bp = '6W')})
assy_dict.update({'A___9' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'A__10' : Assembly(enr = '3.1', bp = '6W')})
assy_dict.update({'A__11' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'B___3' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'B___4' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'B___5' : Assembly(enr = '3.1', bp = '16')})
assy_dict.update({'B___6' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'B___7' : Assembly(enr = '3.1', bp = '20')})
assy_dict.update({'B___8' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'B___9' : Assembly(enr = '3.1', bp = '20')})
assy_dict.update({'B__10' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'B__11' : Assembly(enr = '3.1', bp = '16')})
assy_dict.update({'B__12' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'B__13' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'C___2' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'C___3' : Assembly(enr = '3.1', bp = '15SW')})
assy_dict.update({'C___4' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'C___5' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'C___6' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'C___7' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'C___8' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'C___9' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'C__10' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'C__11' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'C__12' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'C__13' : Assembly(enr = '3.1', bp = '15NW')})
assy_dict.update({'C__14' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'D___2' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'D___3' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'D___4' : Assembly(enr = '2.4', bp = None)})
assy_dict.update({'D___5' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'D___6' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'D___7' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'D___8' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'D___9' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'D__10' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'D__11' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'D__12' : Assembly(enr = '2.4', bp = None)})
assy_dict.update({'D__13' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'D__14' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'E___1' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'E___2' : Assembly(enr = '3.1', bp = '16')})
assy_dict.update({'E___3' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'E___4' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'E___5' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'E___6' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'E___7' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'E___8' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'E___9' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'E__10' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'E__11' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'E__12' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'E__13' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'E__14' : Assembly(enr = '3.1', bp = '16')})
assy_dict.update({'E__15' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'F___1' : Assembly(enr = '3.1', bp = '6S')})
assy_dict.update({'F___2' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'F___3' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'F___4' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'F___5' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'F___6' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'F___7' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'F___8' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'F___9' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'F__10' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'F__11' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'F__12' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'F__13' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'F__14' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'F__15' : Assembly(enr = '3.1', bp = '6N')})

assy_dict.update({'G___1' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'G___2' : Assembly(enr = '3.1', bp = '20')})
assy_dict.update({'G___3' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'G___4' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'G___5' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'G___6' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'G___7' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'G___8' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'G___9' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'G__10' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'G__11' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'G__12' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'G__13' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'G__14' : Assembly(enr = '3.1', bp = '20')})
assy_dict.update({'G__15' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'H___1' : Assembly(enr = '3.1', bp = '6S')})
assy_dict.update({'H___2' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'H___3' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'H___4' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'H___5' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'H___6' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'H___7' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'H___8' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'H___9' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'H__10' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'H__11' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'H__12' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'H__13' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'H__14' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'H__15' : Assembly(enr = '3.1', bp = '6N')})

assy_dict.update({'J___1' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'J___2' : Assembly(enr = '3.1', bp = '20')})
assy_dict.update({'J___3' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'J___4' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'J___5' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'J___6' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'J___7' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'J___8' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'J___9' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'J__10' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'J__11' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'J__12' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'J__13' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'J__14' : Assembly(enr = '3.1', bp = '20')})
assy_dict.update({'J__15' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'K___1' : Assembly(enr = '3.1', bp = '6S')})
assy_dict.update({'K___2' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'K___3' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'K___4' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'K___5' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'K___6' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'K___7' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'K___8' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'K___9' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'K__10' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'K__11' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'K__12' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'K__13' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'K__14' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'K__15' : Assembly(enr = '3.1', bp = '6N')})

assy_dict.update({'L___1' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'L___2' : Assembly(enr = '3.1', bp = '16')})
assy_dict.update({'L___3' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'L___4' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'L___5' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'L___6' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'L___7' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'L___8' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'L___9' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'L__10' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'L__11' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'L__12' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'L__13' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'L__14' : Assembly(enr = '3.1', bp = '16')})
assy_dict.update({'L__15' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'M___2' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'M___3' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'M___4' : Assembly(enr = '2.4', bp = None)})
assy_dict.update({'M___5' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'M___6' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'M___7' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'M___8' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'M___9' : Assembly(enr = '2.4', bp = '12')})
assy_dict.update({'M__10' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'M__11' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'M__12' : Assembly(enr = '2.4', bp = None)})
assy_dict.update({'M__13' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'M__14' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'N___2' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'N___3' : Assembly(enr = '3.1', bp = '15SE')})
assy_dict.update({'N___4' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'N___5' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'N___6' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'N___7' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'N___8' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'N___9' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'N__10' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'N__11' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'N__12' : Assembly(enr = '2.4', bp = '16')})
assy_dict.update({'N__13' : Assembly(enr = '3.1', bp = '15NE')})
assy_dict.update({'N__14' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'P___3' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'P___4' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'P___5' : Assembly(enr = '3.1', bp = '16')})
assy_dict.update({'P___6' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'P___7' : Assembly(enr = '3.1', bp = '20')})
assy_dict.update({'P___8' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'P___9' : Assembly(enr = '3.1', bp = '20')})
assy_dict.update({'P__10' : Assembly(enr = '1.6', bp = None)})
assy_dict.update({'P__11' : Assembly(enr = '3.1', bp = '16')})
assy_dict.update({'P__12' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'P__13' : Assembly(enr = '3.1', bp = None)})

assy_dict.update({'R___5' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'R___6' : Assembly(enr = '3.1', bp = '6E')})
assy_dict.update({'R___7' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'R___8' : Assembly(enr = '3.1', bp = '6E')})
assy_dict.update({'R___9' : Assembly(enr = '3.1', bp = None)})
assy_dict.update({'R__10' : Assembly(enr = '3.1', bp = '6E')})
assy_dict.update({'R__11' : Assembly(enr = '3.1', bp = None)})

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

water_idmap = """
{MOD__.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4} {GRNWc.wid:>4} {GR__N.wid:>4} {GR__N.wid:>4} {GR__N.wid:>4} {GR__N.wid:>4} {GR__N.wid:>4} {GR__N.wid:>4} {GR__N.wid:>4} {GRNEc.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4} 
{MOD__.wid:>4} {MOD__.wid:>4} {GRNWc.wid:>4} {GR__N.wid:>4} {GR_NW.wid:>4} {L___1.wid:>4} {K___1.wid:>4} {J___1.wid:>4} {H___1.wid:>4} {G___1.wid:>4} {F___1.wid:>4} {E___1.wid:>4} {GR_NE.wid:>4} {GR__N.wid:>4} {GRNEc.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4}
{MOD__.wid:>4} {GRNWc.wid:>4} {GR_NW.wid:>4} {N___2.wid:>4} {M___2.wid:>4} {L___2.wid:>4} {K___2.wid:>4} {J___2.wid:>4} {H___2.wid:>4} {G___2.wid:>4} {F___2.wid:>4} {E___2.wid:>4} {D___2.wid:>4} {C___2.wid:>4} {GR_NE.wid:>4} {GRNEc.wid:>4} {MOD__.wid:>4}
{MOD__.wid:>4} {GR__W.wid:>4} {P___3.wid:>4} {N___3.wid:>4} {M___3.wid:>4} {L___3.wid:>4} {K___3.wid:>4} {J___3.wid:>4} {H___3.wid:>4} {G___3.wid:>4} {F___3.wid:>4} {E___3.wid:>4} {D___3.wid:>4} {C___3.wid:>4} {B___3.wid:>4} {GR__E.wid:>4} {MOD__.wid:>4}
{GRNWc.wid:>4} {GR_NW.wid:>4} {P___4.wid:>4} {N___4.wid:>4} {M___4.wid:>4} {L___4.wid:>4} {K___4.wid:>4} {J___4.wid:>4} {H___4.wid:>4} {G___4.wid:>4} {F___4.wid:>4} {E___4.wid:>4} {D___4.wid:>4} {C___4.wid:>4} {B___4.wid:>4} {GR_NE.wid:>4} {GRNEc.wid:>4}
{GR__W.wid:>4} {R___5.wid:>4} {P___5.wid:>4} {N___5.wid:>4} {M___5.wid:>4} {L___5.wid:>4} {K___5.wid:>4} {J___5.wid:>4} {H___5.wid:>4} {G___5.wid:>4} {F___5.wid:>4} {E___5.wid:>4} {D___5.wid:>4} {C___5.wid:>4} {B___5.wid:>4} {A___5.wid:>4} {GR__E.wid:>4}
{GR__W.wid:>4} {R___6.wid:>4} {P___6.wid:>4} {N___6.wid:>4} {M___6.wid:>4} {L___6.wid:>4} {K___6.wid:>4} {J___6.wid:>4} {H___6.wid:>4} {G___6.wid:>4} {F___6.wid:>4} {E___6.wid:>4} {D___6.wid:>4} {C___6.wid:>4} {B___6.wid:>4} {A___6.wid:>4} {GR__E.wid:>4}
{GR__W.wid:>4} {R___7.wid:>4} {P___7.wid:>4} {N___7.wid:>4} {M___7.wid:>4} {L___7.wid:>4} {K___7.wid:>4} {J___7.wid:>4} {H___7.wid:>4} {G___7.wid:>4} {F___7.wid:>4} {E___7.wid:>4} {D___7.wid:>4} {C___7.wid:>4} {B___7.wid:>4} {A___7.wid:>4} {GR__E.wid:>4}
{GR__W.wid:>4} {R___8.wid:>4} {P___8.wid:>4} {N___8.wid:>4} {M___8.wid:>4} {L___8.wid:>4} {K___8.wid:>4} {J___8.wid:>4} {H___8.wid:>4} {G___8.wid:>4} {F___8.wid:>4} {E___8.wid:>4} {D___8.wid:>4} {C___8.wid:>4} {B___8.wid:>4} {A___8.wid:>4} {GR__E.wid:>4}
{GR__W.wid:>4} {R___9.wid:>4} {P___9.wid:>4} {N___9.wid:>4} {M___9.wid:>4} {L___9.wid:>4} {K___9.wid:>4} {J___9.wid:>4} {H___9.wid:>4} {G___9.wid:>4} {F___9.wid:>4} {E___9.wid:>4} {D___9.wid:>4} {C___9.wid:>4} {B___9.wid:>4} {A___9.wid:>4} {GR__E.wid:>4}
{GR__W.wid:>4} {R__10.wid:>4} {P__10.wid:>4} {N__10.wid:>4} {M__10.wid:>4} {L__10.wid:>4} {K__10.wid:>4} {J__10.wid:>4} {H__10.wid:>4} {G__10.wid:>4} {F__10.wid:>4} {E__10.wid:>4} {D__10.wid:>4} {C__10.wid:>4} {B__10.wid:>4} {A__10.wid:>4} {GR__E.wid:>4}
{GR__W.wid:>4} {R__11.wid:>4} {P__11.wid:>4} {N__11.wid:>4} {M__11.wid:>4} {L__11.wid:>4} {K__11.wid:>4} {J__11.wid:>4} {H__11.wid:>4} {G__11.wid:>4} {F__11.wid:>4} {E__11.wid:>4} {D__11.wid:>4} {C__11.wid:>4} {B__11.wid:>4} {A__11.wid:>4} {GR__E.wid:>4}
{GRSWc.wid:>4} {GR_SW.wid:>4} {P__12.wid:>4} {N__12.wid:>4} {M__12.wid:>4} {L__12.wid:>4} {K__12.wid:>4} {J__12.wid:>4} {H__12.wid:>4} {G__12.wid:>4} {F__12.wid:>4} {E__12.wid:>4} {D__12.wid:>4} {C__12.wid:>4} {B__12.wid:>4} {GR_SE.wid:>4} {GRSEc.wid:>4}
{MOD__.wid:>4} {GR__W.wid:>4} {P__13.wid:>4} {N__13.wid:>4} {M__13.wid:>4} {L__13.wid:>4} {K__13.wid:>4} {J__13.wid:>4} {H__13.wid:>4} {G__13.wid:>4} {F__13.wid:>4} {E__13.wid:>4} {D__13.wid:>4} {C__13.wid:>4} {B__13.wid:>4} {GR__E.wid:>4} {MOD__.wid:>4}
{MOD__.wid:>4} {GRSWc.wid:>4} {GR_SW.wid:>4} {N__14.wid:>4} {M__14.wid:>4} {L__14.wid:>4} {K__14.wid:>4} {J__14.wid:>4} {H__14.wid:>4} {G__14.wid:>4} {F__14.wid:>4} {E__14.wid:>4} {D__14.wid:>4} {C__14.wid:>4} {GR_SE.wid:>4} {GRSEc.wid:>4} {MOD__.wid:>4}
{MOD__.wid:>4} {MOD__.wid:>4} {GRSWc.wid:>4} {GR__S.wid:>4} {GR_SW.wid:>4} {L__15.wid:>4} {K__15.wid:>4} {J__15.wid:>4} {H__15.wid:>4} {G__15.wid:>4} {F__15.wid:>4} {E__15.wid:>4} {GR_SE.wid:>4} {GR__S.wid:>4} {GRSEc.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4}
{MOD__.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4} {GRSWc.wid:>4} {GR__S.wid:>4} {GR__S.wid:>4} {GR__S.wid:>4} {GR__S.wid:>4} {GR__S.wid:>4} {GR__S.wid:>4} {GR__S.wid:>4} {GRSEc.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4} {MOD__.wid:>4}
"""

enr_map = """
{MOD__.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4} {GRNWc.enr:>4} {GR__N.enr:>4} {GR__N.enr:>4} {GR__N.enr:>4} {GR__N.enr:>4} {GR__N.enr:>4} {GR__N.enr:>4} {GR__N.enr:>4} {GRNEc.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4} 
{MOD__.enr:>4} {MOD__.enr:>4} {GRNWc.enr:>4} {GR__N.enr:>4} {GR_NW.enr:>4} {L___1.enr:>4} {K___1.enr:>4} {J___1.enr:>4} {H___1.enr:>4} {G___1.enr:>4} {F___1.enr:>4} {E___1.enr:>4} {GR_NE.enr:>4} {GR__N.enr:>4} {GRNEc.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4}
{MOD__.enr:>4} {GRNWc.enr:>4} {GR_NW.enr:>4} {N___2.enr:>4} {M___2.enr:>4} {L___2.enr:>4} {K___2.enr:>4} {J___2.enr:>4} {H___2.enr:>4} {G___2.enr:>4} {F___2.enr:>4} {E___2.enr:>4} {D___2.enr:>4} {C___2.enr:>4} {GR_NE.enr:>4} {GRNEc.enr:>4} {MOD__.enr:>4}
{MOD__.enr:>4} {GR__W.enr:>4} {P___3.enr:>4} {N___3.enr:>4} {M___3.enr:>4} {L___3.enr:>4} {K___3.enr:>4} {J___3.enr:>4} {H___3.enr:>4} {G___3.enr:>4} {F___3.enr:>4} {E___3.enr:>4} {D___3.enr:>4} {C___3.enr:>4} {B___3.enr:>4} {GR__E.enr:>4} {MOD__.enr:>4}
{GRNWc.enr:>4} {GR_NW.enr:>4} {P___4.enr:>4} {N___4.enr:>4} {M___4.enr:>4} {L___4.enr:>4} {K___4.enr:>4} {J___4.enr:>4} {H___4.enr:>4} {G___4.enr:>4} {F___4.enr:>4} {E___4.enr:>4} {D___4.enr:>4} {C___4.enr:>4} {B___4.enr:>4} {GR_NE.enr:>4} {GRNEc.enr:>4}
{GR__W.enr:>4} {R___5.enr:>4} {P___5.enr:>4} {N___5.enr:>4} {M___5.enr:>4} {L___5.enr:>4} {K___5.enr:>4} {J___5.enr:>4} {H___5.enr:>4} {G___5.enr:>4} {F___5.enr:>4} {E___5.enr:>4} {D___5.enr:>4} {C___5.enr:>4} {B___5.enr:>4} {A___5.enr:>4} {GR__E.enr:>4}
{GR__W.enr:>4} {R___6.enr:>4} {P___6.enr:>4} {N___6.enr:>4} {M___6.enr:>4} {L___6.enr:>4} {K___6.enr:>4} {J___6.enr:>4} {H___6.enr:>4} {G___6.enr:>4} {F___6.enr:>4} {E___6.enr:>4} {D___6.enr:>4} {C___6.enr:>4} {B___6.enr:>4} {A___6.enr:>4} {GR__E.enr:>4}
{GR__W.enr:>4} {R___7.enr:>4} {P___7.enr:>4} {N___7.enr:>4} {M___7.enr:>4} {L___7.enr:>4} {K___7.enr:>4} {J___7.enr:>4} {H___7.enr:>4} {G___7.enr:>4} {F___7.enr:>4} {E___7.enr:>4} {D___7.enr:>4} {C___7.enr:>4} {B___7.enr:>4} {A___7.enr:>4} {GR__E.enr:>4}
{GR__W.enr:>4} {R___8.enr:>4} {P___8.enr:>4} {N___8.enr:>4} {M___8.enr:>4} {L___8.enr:>4} {K___8.enr:>4} {J___8.enr:>4} {H___8.enr:>4} {G___8.enr:>4} {F___8.enr:>4} {E___8.enr:>4} {D___8.enr:>4} {C___8.enr:>4} {B___8.enr:>4} {A___8.enr:>4} {GR__E.enr:>4}
{GR__W.enr:>4} {R___9.enr:>4} {P___9.enr:>4} {N___9.enr:>4} {M___9.enr:>4} {L___9.enr:>4} {K___9.enr:>4} {J___9.enr:>4} {H___9.enr:>4} {G___9.enr:>4} {F___9.enr:>4} {E___9.enr:>4} {D___9.enr:>4} {C___9.enr:>4} {B___9.enr:>4} {A___9.enr:>4} {GR__E.enr:>4}
{GR__W.enr:>4} {R__10.enr:>4} {P__10.enr:>4} {N__10.enr:>4} {M__10.enr:>4} {L__10.enr:>4} {K__10.enr:>4} {J__10.enr:>4} {H__10.enr:>4} {G__10.enr:>4} {F__10.enr:>4} {E__10.enr:>4} {D__10.enr:>4} {C__10.enr:>4} {B__10.enr:>4} {A__10.enr:>4} {GR__E.enr:>4}
{GR__W.enr:>4} {R__11.enr:>4} {P__11.enr:>4} {N__11.enr:>4} {M__11.enr:>4} {L__11.enr:>4} {K__11.enr:>4} {J__11.enr:>4} {H__11.enr:>4} {G__11.enr:>4} {F__11.enr:>4} {E__11.enr:>4} {D__11.enr:>4} {C__11.enr:>4} {B__11.enr:>4} {A__11.enr:>4} {GR__E.enr:>4}
{GRSWc.enr:>4} {GR_SW.enr:>4} {P__12.enr:>4} {N__12.enr:>4} {M__12.enr:>4} {L__12.enr:>4} {K__12.enr:>4} {J__12.enr:>4} {H__12.enr:>4} {G__12.enr:>4} {F__12.enr:>4} {E__12.enr:>4} {D__12.enr:>4} {C__12.enr:>4} {B__12.enr:>4} {GR_SE.enr:>4} {GRSEc.enr:>4}
{MOD__.enr:>4} {GR__W.enr:>4} {P__13.enr:>4} {N__13.enr:>4} {M__13.enr:>4} {L__13.enr:>4} {K__13.enr:>4} {J__13.enr:>4} {H__13.enr:>4} {G__13.enr:>4} {F__13.enr:>4} {E__13.enr:>4} {D__13.enr:>4} {C__13.enr:>4} {B__13.enr:>4} {GR__E.enr:>4} {MOD__.enr:>4}
{MOD__.enr:>4} {GRSWc.enr:>4} {GR_SW.enr:>4} {N__14.enr:>4} {M__14.enr:>4} {L__14.enr:>4} {K__14.enr:>4} {J__14.enr:>4} {H__14.enr:>4} {G__14.enr:>4} {F__14.enr:>4} {E__14.enr:>4} {D__14.enr:>4} {C__14.enr:>4} {GR_SE.enr:>4} {GRSEc.enr:>4} {MOD__.enr:>4}
{MOD__.enr:>4} {MOD__.enr:>4} {GRSWc.enr:>4} {GR__S.enr:>4} {GR_SW.enr:>4} {L__15.enr:>4} {K__15.enr:>4} {J__15.enr:>4} {H__15.enr:>4} {G__15.enr:>4} {F__15.enr:>4} {E__15.enr:>4} {GR_SE.enr:>4} {GR__S.enr:>4} {GRSEc.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4}
{MOD__.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4} {GRSWc.enr:>4} {GR__S.enr:>4} {GR__S.enr:>4} {GR__S.enr:>4} {GR__S.enr:>4} {GR__S.enr:>4} {GR__S.enr:>4} {GR__S.enr:>4} {GRSEc.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4} {MOD__.enr:>4}
"""

bp_map = """
{MOD__.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4} {GRNWc.bp:>4} {GR__N.bp:>4} {GR__N.bp:>4} {GR__N.bp:>4} {GR__N.bp:>4} {GR__N.bp:>4} {GR__N.bp:>4} {GR__N.bp:>4} {GRNEc.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4} 
{MOD__.bp:>4} {MOD__.bp:>4} {GRNWc.bp:>4} {GR__N.bp:>4} {GR_NW.bp:>4} {L___1.bp:>4} {K___1.bp:>4} {J___1.bp:>4} {H___1.bp:>4} {G___1.bp:>4} {F___1.bp:>4} {E___1.bp:>4} {GR_NE.bp:>4} {GR__N.bp:>4} {GRNEc.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4}
{MOD__.bp:>4} {GRNWc.bp:>4} {GR_NW.bp:>4} {N___2.bp:>4} {M___2.bp:>4} {L___2.bp:>4} {K___2.bp:>4} {J___2.bp:>4} {H___2.bp:>4} {G___2.bp:>4} {F___2.bp:>4} {E___2.bp:>4} {D___2.bp:>4} {C___2.bp:>4} {GR_NE.bp:>4} {GRNEc.bp:>4} {MOD__.bp:>4}
{MOD__.bp:>4} {GR__W.bp:>4} {P___3.bp:>4} {N___3.bp:>4} {M___3.bp:>4} {L___3.bp:>4} {K___3.bp:>4} {J___3.bp:>4} {H___3.bp:>4} {G___3.bp:>4} {F___3.bp:>4} {E___3.bp:>4} {D___3.bp:>4} {C___3.bp:>4} {B___3.bp:>4} {GR__E.bp:>4} {MOD__.bp:>4}
{GRNWc.bp:>4} {GR_NW.bp:>4} {P___4.bp:>4} {N___4.bp:>4} {M___4.bp:>4} {L___4.bp:>4} {K___4.bp:>4} {J___4.bp:>4} {H___4.bp:>4} {G___4.bp:>4} {F___4.bp:>4} {E___4.bp:>4} {D___4.bp:>4} {C___4.bp:>4} {B___4.bp:>4} {GR_NE.bp:>4} {GRNEc.bp:>4}
{GR__W.bp:>4} {R___5.bp:>4} {P___5.bp:>4} {N___5.bp:>4} {M___5.bp:>4} {L___5.bp:>4} {K___5.bp:>4} {J___5.bp:>4} {H___5.bp:>4} {G___5.bp:>4} {F___5.bp:>4} {E___5.bp:>4} {D___5.bp:>4} {C___5.bp:>4} {B___5.bp:>4} {A___5.bp:>4} {GR__E.bp:>4}
{GR__W.bp:>4} {R___6.bp:>4} {P___6.bp:>4} {N___6.bp:>4} {M___6.bp:>4} {L___6.bp:>4} {K___6.bp:>4} {J___6.bp:>4} {H___6.bp:>4} {G___6.bp:>4} {F___6.bp:>4} {E___6.bp:>4} {D___6.bp:>4} {C___6.bp:>4} {B___6.bp:>4} {A___6.bp:>4} {GR__E.bp:>4}
{GR__W.bp:>4} {R___7.bp:>4} {P___7.bp:>4} {N___7.bp:>4} {M___7.bp:>4} {L___7.bp:>4} {K___7.bp:>4} {J___7.bp:>4} {H___7.bp:>4} {G___7.bp:>4} {F___7.bp:>4} {E___7.bp:>4} {D___7.bp:>4} {C___7.bp:>4} {B___7.bp:>4} {A___7.bp:>4} {GR__E.bp:>4}
{GR__W.bp:>4} {R___8.bp:>4} {P___8.bp:>4} {N___8.bp:>4} {M___8.bp:>4} {L___8.bp:>4} {K___8.bp:>4} {J___8.bp:>4} {H___8.bp:>4} {G___8.bp:>4} {F___8.bp:>4} {E___8.bp:>4} {D___8.bp:>4} {C___8.bp:>4} {B___8.bp:>4} {A___8.bp:>4} {GR__E.bp:>4}
{GR__W.bp:>4} {R___9.bp:>4} {P___9.bp:>4} {N___9.bp:>4} {M___9.bp:>4} {L___9.bp:>4} {K___9.bp:>4} {J___9.bp:>4} {H___9.bp:>4} {G___9.bp:>4} {F___9.bp:>4} {E___9.bp:>4} {D___9.bp:>4} {C___9.bp:>4} {B___9.bp:>4} {A___9.bp:>4} {GR__E.bp:>4}
{GR__W.bp:>4} {R__10.bp:>4} {P__10.bp:>4} {N__10.bp:>4} {M__10.bp:>4} {L__10.bp:>4} {K__10.bp:>4} {J__10.bp:>4} {H__10.bp:>4} {G__10.bp:>4} {F__10.bp:>4} {E__10.bp:>4} {D__10.bp:>4} {C__10.bp:>4} {B__10.bp:>4} {A__10.bp:>4} {GR__E.bp:>4}
{GR__W.bp:>4} {R__11.bp:>4} {P__11.bp:>4} {N__11.bp:>4} {M__11.bp:>4} {L__11.bp:>4} {K__11.bp:>4} {J__11.bp:>4} {H__11.bp:>4} {G__11.bp:>4} {F__11.bp:>4} {E__11.bp:>4} {D__11.bp:>4} {C__11.bp:>4} {B__11.bp:>4} {A__11.bp:>4} {GR__E.bp:>4}
{GRSWc.bp:>4} {GR_SW.bp:>4} {P__12.bp:>4} {N__12.bp:>4} {M__12.bp:>4} {L__12.bp:>4} {K__12.bp:>4} {J__12.bp:>4} {H__12.bp:>4} {G__12.bp:>4} {F__12.bp:>4} {E__12.bp:>4} {D__12.bp:>4} {C__12.bp:>4} {B__12.bp:>4} {GR_SE.bp:>4} {GRSEc.bp:>4}
{MOD__.bp:>4} {GR__W.bp:>4} {P__13.bp:>4} {N__13.bp:>4} {M__13.bp:>4} {L__13.bp:>4} {K__13.bp:>4} {J__13.bp:>4} {H__13.bp:>4} {G__13.bp:>4} {F__13.bp:>4} {E__13.bp:>4} {D__13.bp:>4} {C__13.bp:>4} {B__13.bp:>4} {GR__E.bp:>4} {MOD__.bp:>4}
{MOD__.bp:>4} {GRSWc.bp:>4} {GR_SW.bp:>4} {N__14.bp:>4} {M__14.bp:>4} {L__14.bp:>4} {K__14.bp:>4} {J__14.bp:>4} {H__14.bp:>4} {G__14.bp:>4} {F__14.bp:>4} {E__14.bp:>4} {D__14.bp:>4} {C__14.bp:>4} {GR_SE.bp:>4} {GRSEc.bp:>4} {MOD__.bp:>4}
{MOD__.bp:>4} {MOD__.bp:>4} {GRSWc.bp:>4} {GR__S.bp:>4} {GR_SW.bp:>4} {L__15.bp:>4} {K__15.bp:>4} {J__15.bp:>4} {H__15.bp:>4} {G__15.bp:>4} {F__15.bp:>4} {E__15.bp:>4} {GR_SE.bp:>4} {GR__S.bp:>4} {GRSEc.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4}
{MOD__.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4} {GRSWc.bp:>4} {GR__S.bp:>4} {GR__S.bp:>4} {GR__S.bp:>4} {GR__S.bp:>4} {GR__S.bp:>4} {GR__S.bp:>4} {GR__S.bp:>4} {GRSEc.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4} {MOD__.bp:>4}
"""

density_map = """
{MOD__.density:>4} {MOD__.density:>4} {MOD__.density:>4} {MOD__.density:>4} {GRNWc.density:>4} {GR__N.density:>4} {GR__N.density:>4} {GR__N.density:>4} {GR__N.density:>4} {GR__N.density:>4} {GR__N.density:>4} {GR__N.density:>4} {GRNEc.density:>4} {MOD__.density:>4} {MOD__.density:>4} {MOD__.density:>4} {MOD__.density:>4} 
{MOD__.density:>4} {MOD__.density:>4} {GRNWc.density:>4} {GR__N.density:>4} {GR_NW.density:>4} {L___1.density:>4} {K___1.density:>4} {J___1.density:>4} {H___1.density:>4} {G___1.density:>4} {F___1.density:>4} {E___1.density:>4} {GR_NE.density:>4} {GR__N.density:>4} {GRNEc.density:>4} {MOD__.density:>4} {MOD__.density:>4}
{MOD__.density:>4} {GRNWc.density:>4} {GR_NW.density:>4} {N___2.density:>4} {M___2.density:>4} {L___2.density:>4} {K___2.density:>4} {J___2.density:>4} {H___2.density:>4} {G___2.density:>4} {F___2.density:>4} {E___2.density:>4} {D___2.density:>4} {C___2.density:>4} {GR_NE.density:>4} {GRNEc.density:>4} {MOD__.density:>4}
{MOD__.density:>4} {GR__W.density:>4} {P___3.density:>4} {N___3.density:>4} {M___3.density:>4} {L___3.density:>4} {K___3.density:>4} {J___3.density:>4} {H___3.density:>4} {G___3.density:>4} {F___3.density:>4} {E___3.density:>4} {D___3.density:>4} {C___3.density:>4} {B___3.density:>4} {GR__E.density:>4} {MOD__.density:>4}
{GRNWc.density:>4} {GR_NW.density:>4} {P___4.density:>4} {N___4.density:>4} {M___4.density:>4} {L___4.density:>4} {K___4.density:>4} {J___4.density:>4} {H___4.density:>4} {G___4.density:>4} {F___4.density:>4} {E___4.density:>4} {D___4.density:>4} {C___4.density:>4} {B___4.density:>4} {GR_NE.density:>4} {GRNEc.density:>4}
{GR__W.density:>4} {R___5.density:>4} {P___5.density:>4} {N___5.density:>4} {M___5.density:>4} {L___5.density:>4} {K___5.density:>4} {J___5.density:>4} {H___5.density:>4} {G___5.density:>4} {F___5.density:>4} {E___5.density:>4} {D___5.density:>4} {C___5.density:>4} {B___5.density:>4} {A___5.density:>4} {GR__E.density:>4}
{GR__W.density:>4} {R___6.density:>4} {P___6.density:>4} {N___6.density:>4} {M___6.density:>4} {L___6.density:>4} {K___6.density:>4} {J___6.density:>4} {H___6.density:>4} {G___6.density:>4} {F___6.density:>4} {E___6.density:>4} {D___6.density:>4} {C___6.density:>4} {B___6.density:>4} {A___6.density:>4} {GR__E.density:>4}
{GR__W.density:>4} {R___7.density:>4} {P___7.density:>4} {N___7.density:>4} {M___7.density:>4} {L___7.density:>4} {K___7.density:>4} {J___7.density:>4} {H___7.density:>4} {G___7.density:>4} {F___7.density:>4} {E___7.density:>4} {D___7.density:>4} {C___7.density:>4} {B___7.density:>4} {A___7.density:>4} {GR__E.density:>4}
{GR__W.density:>4} {R___8.density:>4} {P___8.density:>4} {N___8.density:>4} {M___8.density:>4} {L___8.density:>4} {K___8.density:>4} {J___8.density:>4} {H___8.density:>4} {G___8.density:>4} {F___8.density:>4} {E___8.density:>4} {D___8.density:>4} {C___8.density:>4} {B___8.density:>4} {A___8.density:>4} {GR__E.density:>4}
{GR__W.density:>4} {R___9.density:>4} {P___9.density:>4} {N___9.density:>4} {M___9.density:>4} {L___9.density:>4} {K___9.density:>4} {J___9.density:>4} {H___9.density:>4} {G___9.density:>4} {F___9.density:>4} {E___9.density:>4} {D___9.density:>4} {C___9.density:>4} {B___9.density:>4} {A___9.density:>4} {GR__E.density:>4}
{GR__W.density:>4} {R__10.density:>4} {P__10.density:>4} {N__10.density:>4} {M__10.density:>4} {L__10.density:>4} {K__10.density:>4} {J__10.density:>4} {H__10.density:>4} {G__10.density:>4} {F__10.density:>4} {E__10.density:>4} {D__10.density:>4} {C__10.density:>4} {B__10.density:>4} {A__10.density:>4} {GR__E.density:>4}
{GR__W.density:>4} {R__11.density:>4} {P__11.density:>4} {N__11.density:>4} {M__11.density:>4} {L__11.density:>4} {K__11.density:>4} {J__11.density:>4} {H__11.density:>4} {G__11.density:>4} {F__11.density:>4} {E__11.density:>4} {D__11.density:>4} {C__11.density:>4} {B__11.density:>4} {A__11.density:>4} {GR__E.density:>4}
{GRSWc.density:>4} {GR_SW.density:>4} {P__12.density:>4} {N__12.density:>4} {M__12.density:>4} {L__12.density:>4} {K__12.density:>4} {J__12.density:>4} {H__12.density:>4} {G__12.density:>4} {F__12.density:>4} {E__12.density:>4} {D__12.density:>4} {C__12.density:>4} {B__12.density:>4} {GR_SE.density:>4} {GRSEc.density:>4}
{MOD__.density:>4} {GR__W.density:>4} {P__13.density:>4} {N__13.density:>4} {M__13.density:>4} {L__13.density:>4} {K__13.density:>4} {J__13.density:>4} {H__13.density:>4} {G__13.density:>4} {F__13.density:>4} {E__13.density:>4} {D__13.density:>4} {C__13.density:>4} {B__13.density:>4} {GR__E.density:>4} {MOD__.density:>4}
{MOD__.density:>4} {GRSWc.density:>4} {GR_SW.density:>4} {N__14.density:>4} {M__14.density:>4} {L__14.density:>4} {K__14.density:>4} {J__14.density:>4} {H__14.density:>4} {G__14.density:>4} {F__14.density:>4} {E__14.density:>4} {D__14.density:>4} {C__14.density:>4} {GR_SE.density:>4} {GRSEc.density:>4} {MOD__.density:>4}
{MOD__.density:>4} {MOD__.density:>4} {GRSWc.density:>4} {GR__S.density:>4} {GR_SW.density:>4} {L__15.density:>4} {K__15.density:>4} {J__15.density:>4} {H__15.density:>4} {G__15.density:>4} {F__15.density:>4} {E__15.density:>4} {GR_SE.density:>4} {GR__S.density:>4} {GRSEc.density:>4} {MOD__.density:>4} {MOD__.density:>4}
{MOD__.density:>4} {MOD__.density:>4} {MOD__.density:>4} {MOD__.density:>4} {GRSWc.density:>4} {GR__S.density:>4} {GR__S.density:>4} {GR__S.density:>4} {GR__S.density:>4} {GR__S.density:>4} {GR__S.density:>4} {GR__S.density:>4} {GRSEc.density:>4} {MOD__.density:>4} {MOD__.density:>4} {MOD__.density:>4} {MOD__.density:>4}
"""

fueltemp_map = """
{MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {GRNWc.fueltemp:>4} {GR__N.fueltemp:>4} {GR__N.fueltemp:>4} {GR__N.fueltemp:>4} {GR__N.fueltemp:>4} {GR__N.fueltemp:>4} {GR__N.fueltemp:>4} {GR__N.fueltemp:>4} {GRNEc.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4} 
{MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {GRNWc.fueltemp:>4} {GR__N.fueltemp:>4} {GR_NW.fueltemp:>4} {L___1.fueltemp:>4} {K___1.fueltemp:>4} {J___1.fueltemp:>4} {H___1.fueltemp:>4} {G___1.fueltemp:>4} {F___1.fueltemp:>4} {E___1.fueltemp:>4} {GR_NE.fueltemp:>4} {GR__N.fueltemp:>4} {GRNEc.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4}
{MOD__.fueltemp:>4} {GRNWc.fueltemp:>4} {GR_NW.fueltemp:>4} {N___2.fueltemp:>4} {M___2.fueltemp:>4} {L___2.fueltemp:>4} {K___2.fueltemp:>4} {J___2.fueltemp:>4} {H___2.fueltemp:>4} {G___2.fueltemp:>4} {F___2.fueltemp:>4} {E___2.fueltemp:>4} {D___2.fueltemp:>4} {C___2.fueltemp:>4} {GR_NE.fueltemp:>4} {GRNEc.fueltemp:>4} {MOD__.fueltemp:>4}
{MOD__.fueltemp:>4} {GR__W.fueltemp:>4} {P___3.fueltemp:>4} {N___3.fueltemp:>4} {M___3.fueltemp:>4} {L___3.fueltemp:>4} {K___3.fueltemp:>4} {J___3.fueltemp:>4} {H___3.fueltemp:>4} {G___3.fueltemp:>4} {F___3.fueltemp:>4} {E___3.fueltemp:>4} {D___3.fueltemp:>4} {C___3.fueltemp:>4} {B___3.fueltemp:>4} {GR__E.fueltemp:>4} {MOD__.fueltemp:>4}
{GRNWc.fueltemp:>4} {GR_NW.fueltemp:>4} {P___4.fueltemp:>4} {N___4.fueltemp:>4} {M___4.fueltemp:>4} {L___4.fueltemp:>4} {K___4.fueltemp:>4} {J___4.fueltemp:>4} {H___4.fueltemp:>4} {G___4.fueltemp:>4} {F___4.fueltemp:>4} {E___4.fueltemp:>4} {D___4.fueltemp:>4} {C___4.fueltemp:>4} {B___4.fueltemp:>4} {GR_NE.fueltemp:>4} {GRNEc.fueltemp:>4}
{GR__W.fueltemp:>4} {R___5.fueltemp:>4} {P___5.fueltemp:>4} {N___5.fueltemp:>4} {M___5.fueltemp:>4} {L___5.fueltemp:>4} {K___5.fueltemp:>4} {J___5.fueltemp:>4} {H___5.fueltemp:>4} {G___5.fueltemp:>4} {F___5.fueltemp:>4} {E___5.fueltemp:>4} {D___5.fueltemp:>4} {C___5.fueltemp:>4} {B___5.fueltemp:>4} {A___5.fueltemp:>4} {GR__E.fueltemp:>4}
{GR__W.fueltemp:>4} {R___6.fueltemp:>4} {P___6.fueltemp:>4} {N___6.fueltemp:>4} {M___6.fueltemp:>4} {L___6.fueltemp:>4} {K___6.fueltemp:>4} {J___6.fueltemp:>4} {H___6.fueltemp:>4} {G___6.fueltemp:>4} {F___6.fueltemp:>4} {E___6.fueltemp:>4} {D___6.fueltemp:>4} {C___6.fueltemp:>4} {B___6.fueltemp:>4} {A___6.fueltemp:>4} {GR__E.fueltemp:>4}
{GR__W.fueltemp:>4} {R___7.fueltemp:>4} {P___7.fueltemp:>4} {N___7.fueltemp:>4} {M___7.fueltemp:>4} {L___7.fueltemp:>4} {K___7.fueltemp:>4} {J___7.fueltemp:>4} {H___7.fueltemp:>4} {G___7.fueltemp:>4} {F___7.fueltemp:>4} {E___7.fueltemp:>4} {D___7.fueltemp:>4} {C___7.fueltemp:>4} {B___7.fueltemp:>4} {A___7.fueltemp:>4} {GR__E.fueltemp:>4}
{GR__W.fueltemp:>4} {R___8.fueltemp:>4} {P___8.fueltemp:>4} {N___8.fueltemp:>4} {M___8.fueltemp:>4} {L___8.fueltemp:>4} {K___8.fueltemp:>4} {J___8.fueltemp:>4} {H___8.fueltemp:>4} {G___8.fueltemp:>4} {F___8.fueltemp:>4} {E___8.fueltemp:>4} {D___8.fueltemp:>4} {C___8.fueltemp:>4} {B___8.fueltemp:>4} {A___8.fueltemp:>4} {GR__E.fueltemp:>4}
{GR__W.fueltemp:>4} {R___9.fueltemp:>4} {P___9.fueltemp:>4} {N___9.fueltemp:>4} {M___9.fueltemp:>4} {L___9.fueltemp:>4} {K___9.fueltemp:>4} {J___9.fueltemp:>4} {H___9.fueltemp:>4} {G___9.fueltemp:>4} {F___9.fueltemp:>4} {E___9.fueltemp:>4} {D___9.fueltemp:>4} {C___9.fueltemp:>4} {B___9.fueltemp:>4} {A___9.fueltemp:>4} {GR__E.fueltemp:>4}
{GR__W.fueltemp:>4} {R__10.fueltemp:>4} {P__10.fueltemp:>4} {N__10.fueltemp:>4} {M__10.fueltemp:>4} {L__10.fueltemp:>4} {K__10.fueltemp:>4} {J__10.fueltemp:>4} {H__10.fueltemp:>4} {G__10.fueltemp:>4} {F__10.fueltemp:>4} {E__10.fueltemp:>4} {D__10.fueltemp:>4} {C__10.fueltemp:>4} {B__10.fueltemp:>4} {A__10.fueltemp:>4} {GR__E.fueltemp:>4}
{GR__W.fueltemp:>4} {R__11.fueltemp:>4} {P__11.fueltemp:>4} {N__11.fueltemp:>4} {M__11.fueltemp:>4} {L__11.fueltemp:>4} {K__11.fueltemp:>4} {J__11.fueltemp:>4} {H__11.fueltemp:>4} {G__11.fueltemp:>4} {F__11.fueltemp:>4} {E__11.fueltemp:>4} {D__11.fueltemp:>4} {C__11.fueltemp:>4} {B__11.fueltemp:>4} {A__11.fueltemp:>4} {GR__E.fueltemp:>4}
{GRSWc.fueltemp:>4} {GR_SW.fueltemp:>4} {P__12.fueltemp:>4} {N__12.fueltemp:>4} {M__12.fueltemp:>4} {L__12.fueltemp:>4} {K__12.fueltemp:>4} {J__12.fueltemp:>4} {H__12.fueltemp:>4} {G__12.fueltemp:>4} {F__12.fueltemp:>4} {E__12.fueltemp:>4} {D__12.fueltemp:>4} {C__12.fueltemp:>4} {B__12.fueltemp:>4} {GR_SE.fueltemp:>4} {GRSEc.fueltemp:>4}
{MOD__.fueltemp:>4} {GR__W.fueltemp:>4} {P__13.fueltemp:>4} {N__13.fueltemp:>4} {M__13.fueltemp:>4} {L__13.fueltemp:>4} {K__13.fueltemp:>4} {J__13.fueltemp:>4} {H__13.fueltemp:>4} {G__13.fueltemp:>4} {F__13.fueltemp:>4} {E__13.fueltemp:>4} {D__13.fueltemp:>4} {C__13.fueltemp:>4} {B__13.fueltemp:>4} {GR__E.fueltemp:>4} {MOD__.fueltemp:>4}
{MOD__.fueltemp:>4} {GRSWc.fueltemp:>4} {GR_SW.fueltemp:>4} {N__14.fueltemp:>4} {M__14.fueltemp:>4} {L__14.fueltemp:>4} {K__14.fueltemp:>4} {J__14.fueltemp:>4} {H__14.fueltemp:>4} {G__14.fueltemp:>4} {F__14.fueltemp:>4} {E__14.fueltemp:>4} {D__14.fueltemp:>4} {C__14.fueltemp:>4} {GR_SE.fueltemp:>4} {GRSEc.fueltemp:>4} {MOD__.fueltemp:>4}
{MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {GRSWc.fueltemp:>4} {GR__S.fueltemp:>4} {GR_SW.fueltemp:>4} {L__15.fueltemp:>4} {K__15.fueltemp:>4} {J__15.fueltemp:>4} {H__15.fueltemp:>4} {G__15.fueltemp:>4} {F__15.fueltemp:>4} {E__15.fueltemp:>4} {GR_SE.fueltemp:>4} {GR__S.fueltemp:>4} {GRSEc.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4}
{MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {GRSWc.fueltemp:>4} {GR__S.fueltemp:>4} {GR__S.fueltemp:>4} {GR__S.fueltemp:>4} {GR__S.fueltemp:>4} {GR__S.fueltemp:>4} {GR__S.fueltemp:>4} {GR__S.fueltemp:>4} {GRSEc.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4} {MOD__.fueltemp:>4}
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
    create_cmfd()

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

    # Carbon steel ASTM A533 Grade B
    mat_cs = Material('cs', 'Carbon steel')
    mat_cs.add_nuclide('C-Nat', '71c', '9.7772E-04')
    mat_cs.add_nuclide('Si-28', '71c', '4.2417E-04')
    mat_cs.add_nuclide('Si-29', '71c', '2.1548E-05')
    mat_cs.add_nuclide('Si-30', '71c', '1.4221E-05')
    mat_cs.add_nuclide('Mn-55', '71c', '1.1329E-03')
    mat_cs.add_nuclide('P-31',  '71c', '3.7913E-05')
    mat_cs.add_nuclide('Mo-92', '71c', '3.7965E-05')
    mat_cs.add_nuclide('Mo-94', '71c', '2.3725E-05')
    mat_cs.add_nuclide('Mo-96', '71c', '4.2875E-05')
    mat_cs.add_nuclide('Mo-97', '71c', '2.4573E-05')
    mat_cs.add_nuclide('Mo-98', '71c', '6.2179E-05')
    mat_cs.add_nuclide('Mo-100','71c', '2.4856E-05')
    mat_cs.add_nuclide('Fe-54', '71c', '4.7714E-03')
    mat_cs.add_nuclide('Fe-56', '71c', '7.4900E-02')
    mat_cs.add_nuclide('Fe-57', '71c', '1.7298E-03')
    mat_cs.add_nuclide('Fe-58', '71c', '2.3020E-04')
    mat_cs.add_nuclide('Ni-58', '71c', '2.9965E-04')
    mat_cs.add_nuclide('Ni-60', '71c', '1.1543E-04')
    mat_cs.add_nuclide('Ni-61', '71c', '5.0175E-06')
    mat_cs.add_nuclide('Ni-62', '71c', '1.5998E-05')
    mat_cs.add_nuclide('Ni-64', '71c', '4.0742E-06')
    mat_cs.add_color('0 0 0')
    mat_cs.finalize()

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
    add_surface('core_barrelOR', 'z-cylinder', '0.0 0.0 193.675', comment = 'Core barrel outer radius')
    add_surface('shield_OR', 'z-cylinder', '0.0 0.0 199.39', comment = 'Shield panel outer radius')
    add_surface('shield_NESW60', 'plane', '1 {0} 0 0'.format(math.tan(math.pi/3)), comment = 'Shield panel cut plane')
    add_surface('shield_NESW30', 'plane', '1 {0} 0 0'.format(math.tan(math.pi/6)), comment = 'Shield panel cut plane')
    add_surface('shield_NWSE60', 'plane', '1 {0} 0 0'.format(-math.tan(math.pi/3)), comment = 'Shield panel cut plane')
    add_surface('shield_NWSE30', 'plane', '1 {0} 0 0'.format(-math.tan(math.pi/6)), comment = 'Shield panel cut plane')
    add_surface('rpv_IR', 'z-cylinder', '0.0 0.0 230.09', comment = 'RPV inner radius')
    add_surface('rpv_OR', 'z-cylinder', '0.0 0.0 251.9', bc = 'vacuum', comment = 'RPV outer radius')
    add_surface('floor', 'z-plane', '-100.0', bc = 'reflective', comment = 'Lowest plane')
    add_surface('ceiling', 'z-plane', '100.0', bc = 'reflective', comment = 'Hightest plane')

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

def create_lattice(assy, fuel_key, bp_key, gt_key, it_key, water_key = None, bp_config=None, comment = None):

    # Set up pin ids
    fuel_id = univ_dict[fuel_key].id
    bp_id = univ_dict[bp_key].id
    gt_id = univ_dict[gt_key].id
    it_id = univ_dict[it_key].id

    # No grid present in this model so set these all to water
    if water_key == None:
        water_univ = 'mod'
    else:
        add_cell(water_key,
            surfaces = '',
            universe = water_key,
            material =  mat_dict[water_key].id,
            comment = '{0} universe'.format(water_key))
        water_univ = water_key
    no_id = univ_dict[water_univ].id
    ne_id = univ_dict[water_univ].id
    ea_id = univ_dict[water_univ].id
    se_id = univ_dict[water_univ].id
    so_id = univ_dict[water_univ].id
    sw_id = univ_dict[water_univ].id
    we_id = univ_dict[water_univ].id
    nw_id = univ_dict[water_univ].id

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
    if bp_config == '0':
        pass
    elif bp_config == '6N':
        univs.update({
            'pa' : bp_id,
	    'pc' : bp_id,
            'pd' : bp_id,
            'pe' : bp_id,
            'pf' : bp_id,
            'pj' : bp_id})
        assy_dict[assy].bp = '6'
    elif bp_config == '6E':
        univs.update({
            'pj' : bp_id,
	    'pt' : bp_id,
            'pe' : bp_id,
            'pv' : bp_id,
            'pc' : bp_id,
            'py' : bp_id})
        assy_dict[assy].bp = '6'
    elif bp_config == '6S':
        univs.update({
            'py' : bp_id,
	    'pw' : bp_id,
            'pv' : bp_id,
            'pu' : bp_id,
            'pt' : bp_id,
            'pp' : bp_id})
        assy_dict[assy].bp = '6'
    elif bp_config == '6W':
        univs.update({
            'pp' : bp_id,
	    'pf' : bp_id,
            'pu' : bp_id,
            'pd' : bp_id,
            'pw' : bp_id,
            'pa' : bp_id})
        assy_dict[assy].bp = '6'
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
        assy_dict[assy].bp = '15'
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
        assy_dict[assy].bp = '15'
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
        assy_dict[assy].bp = '15'
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
        assy_dict[assy].bp = '15'
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
    add_lattice(assy,
        dimension = '19 19',
        lower_left = '{0} {0}'.format(lleft),
        width = '{0} {0}'.format(pin_pitch),
        universes = pin_lattice.format(**univs),
        comment = comment)

def create_assemblies():

    # Begin loop around fuel assemblies
    for assy in assy_dict.keys():

        # Check for non fuel assembly
        if assy_dict[assy].enr == '0.0':
            continue

        # Sample density
        density = random.uniform(low_density, hzp_density)
        color = -156.0/(hzp_density - low_density) * \
                (density - hzp_density)
        assy_dict[assy].add_density(density)

        # Create a water material with that density
        create_water_material(assy, density, color)
        assy_dict[assy].add_waterid(mat_dict['{0} water'.format(assy)].id)

        # Sample fuel temperature
        fueltemp = random.uniform(hzp_fueltemp, high_fueltemp)
        assy_dict[assy].add_fueltemp(fueltemp)

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
        create_fuelpin_cell('{0} fuelpin'.format(assy), fuelpin, '{0} water'.format(assy)) 
        create_bppin_cell('{0} bppin'.format(assy), 'bp', '{0} water'.format(assy))
        create_gtpin_cell('{0} gtpin'.format(assy), 'gt', '{0} water'.format(assy)) 

        # Create lattice
        create_lattice(assy, '{0} fuelpin'.format(assy), '{0} bppin'.format(assy),
                       '{0} gtpin'.format(assy), '{0} gtpin'.format(assy), water_key = '{0} water'.format(assy),
                       bp_config = assy_dict[assy].bp, comment = '{0} lattice'.format(assy))

        # Create cell to put lattice in
        add_cell('{0} lattice fill'.format(assy),
            surfaces = '',
            universe = '{0} assembly'.format(assy),
            fill = lat_dict[assy].id,
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
        surfaces = '{0} -{1} {2} -{3} -{4} {5} -{6}'.format(surf_dict['core_left'].id, surf_dict['core_right'].id,
                                        surf_dict['core_back'].id, surf_dict['core_front'].id, surf_dict['core_barrelIR'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        fill = lat_dict['Core Lattice'].id,
        comment = 'Core fill')

    # Add moderator outside of core
    add_cell('coremodN',
        surfaces = '{0} {1} -{2} -{3} {4} -{5}'.format(surf_dict['core_front'].id, surf_dict['core_left'].id, surf_dict['core_right'].id, surf_dict['core_barrelIR'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Moderator around N of core')
    add_cell('coremodS',
        surfaces = '-{0} {1} -{2} -{3} {4} -{5}'.format(surf_dict['core_back'].id, surf_dict['core_left'].id, surf_dict['core_right'].id, surf_dict['core_barrelIR'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Moderator around S of core')
    add_cell('coremodNES',
        surfaces = '{0} -{1} {2} -{3}'.format(surf_dict['core_right'].id, surf_dict['core_barrelIR'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Moderator around E of core')
    add_cell('coremodSWN',
        surfaces = '-{0} -{1} {2} -{3}'.format(surf_dict['core_left'].id, surf_dict['core_barrelIR'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Moderator around W of core')

    # Add in core barrel
    add_cell('core_barrel',
        surfaces = '{0} -{1} {2} -{3}'.format(surf_dict['core_barrelIR'].id, surf_dict['core_barrelOR'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['ss'].id,
        comment = 'Core Barrel')

    # Add shield panel ring
    add_cell('panelNE',
        surfaces = '{0} -{1} {2} -{3} {4} -{5}'.format(surf_dict['core_barrelOR'].id, surf_dict['shield_OR'].id, surf_dict['shield_NESW60'].id, surf_dict['shield_NESW30'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['ss'].id,
        comment = 'Northeast Shield Panel')
    add_cell('panelmodE',
        surfaces = '{0} -{1} {2} -{3} {4} -{5}'.format(surf_dict['core_barrelOR'].id, surf_dict['shield_OR'].id, surf_dict['shield_NESW30'].id, surf_dict['shield_NWSE30'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod East Shield Panel')
    add_cell('panelSE',
       surfaces = '{0} -{1} {2} -{3} {4} -{5}'.format(surf_dict['core_barrelOR'].id, surf_dict['shield_OR'].id, surf_dict['shield_NWSE30'].id, surf_dict['shield_NWSE60'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
       material = mat_dict['ss'].id,
       comment = 'Southeast Shield Panel')
    add_cell('panelmodS',
        surfaces = '{0} -{1}  {2} {3} {4} -{5}'.format(surf_dict['core_barrelOR'].id, surf_dict['shield_OR'].id, surf_dict['shield_NWSE60'].id, surf_dict['shield_NESW60'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod South Shield Panel')
    add_cell('panelSW',
        surfaces = '{0} -{1} -{2} {3} {4} -{5}'.format(surf_dict['core_barrelOR'].id, surf_dict['shield_OR'].id, surf_dict['shield_NESW60'].id, surf_dict['shield_NESW30'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['ss'].id,
        comment = 'Southwest Shield Panel')
    add_cell('panelmodW',
        surfaces = '{0} -{1} -{2} {3} {4} -{5}'.format(surf_dict['core_barrelOR'].id, surf_dict['shield_OR'].id, surf_dict['shield_NESW30'].id, surf_dict['shield_NWSE30'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod West Shield Panel')
    add_cell('panelNW',
        surfaces = '{0} -{1} -{2} {3} {4} -{5}'.format(surf_dict['core_barrelOR'].id, surf_dict['shield_OR'].id, surf_dict['shield_NWSE30'].id, surf_dict['shield_NWSE60'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['ss'].id,
        comment = 'Northwest Shield Panel')
    add_cell('panelmodN',
        surfaces = '{0} -{1} -{2} -{3} {4} -{5}'.format(surf_dict['core_barrelOR'].id, surf_dict['shield_OR'].id, surf_dict['shield_NWSE60'].id, surf_dict['shield_NESW60'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod North Shield Panel')

    # Add moderator before rpv
    add_cell('modrpv',
        surfaces = '{0} -{1} {2} -{3}'.format(surf_dict['shield_OR'].id, surf_dict['rpv_IR'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['h2o_hzp'].id,
        comment = 'Mod before RPV')

    # Add rpv
    add_cell('rpv',
        surfaces = '{0} -{1} {2} -{3}'.format(surf_dict['rpv_IR'].id, surf_dict['rpv_OR'].id,
                                        surf_dict['floor'].id, surf_dict['ceiling'].id),
        material = mat_dict['cs'].id,
        comment = 'RPV')

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

    mat_h2o = Material('{0} water'.format(key), 'Water @ {0} g/cc in {1}'.format(water_density, key))
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
    mx = -17.0*assy_pitch/2.0
    my = -17.0*assy_pitch/2.0
    mz = -100.0 
    px = 17.0*assy_pitch/2.0
    py = 17.0*assy_pitch/2.0
    pz = 100.0 
    cmfd.update({'lower_left': '{0} {1} {2}'.format(mx, my, mz)})
    cmfd.update({'upper_right':'{0} {1} {2}'.format(px, py, pz)})
    cmfd.update({'dimension':'17 17 1'})
    map_str = """
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1""" 
    cmfd.update({'map':map_str})

    # Put water id map in
    cmfd.update({'water_map':water_idmap.format(**assy_dict)})

    # Put enrichment and bp map together
    cmfd.update({'enr_map':enr_map.format(**assy_dict)})
    cmfd.update({'bp_map':bp_map.format(**assy_dict)})

    # Fuel temperature
    cmfd.update({'fuel_temp':fueltemp_map.format(**assy_dict)})

    # Density
    cmfd.update({'density':density_map.format(**assy_dict)})

    # Normalization
    cmfd.update({'norm':cmfd['n_assemblies']})

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
'zbot' : -100.0,
'xtop' : 17*assy_pitch/2.0,
'ytop' : 17*assy_pitch/2.0,
'ztop' : 100.0,
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

############ CMFD File ###############

    cmfd_str = """<?xml version="1.0" encoding="UTF-8"?>
<cmfd>

  <!-- This file auto-generated by beavrs.py  -->
  <mesh>
    <lower_left>{lower_left}</lower_left>
    <upper_right>{upper_right}</upper_right>
    <dimension>{dimension}</dimension>
    <albedo>0.0 0.0 0.0 0.0 1.0 1.0</albedo>
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
