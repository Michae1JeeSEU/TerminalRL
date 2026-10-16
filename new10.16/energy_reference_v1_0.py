"""Reconstructed independent scalar reference from FT-RS-S1-1.0 Appendix A.

The referenced program was absent from the supplied three-file package. This
reconstruction uses only the standard library, no production-kernel imports.
Its output is a formula check, not a scenario qualification result.
"""
import math
import json

def power(v,a):
    x,y,z=v; ax,ay,az=a; rho=1.225; area=math.pi*(1.575/2)**2
    xy=math.sqrt(x*x+y*y)
    tx=620*ax+rho*2.11*xy*x/2
    ty=620*ay+rho*2.11*xy*y/2
    tz=620*(az+9.80665)+rho*1.47*abs(z)*z/2
    thrust=math.sqrt(tx*tx+ty*ty+tz*tz)
    if thrust<=0 or tz<=0: raise ValueError('ENERGY_MODEL_DOMAIN')
    work=tx*x+ty*y+tz*z
    normal=work/thrust; plane=math.sqrt(max(0,x*x+y*y+z*z-normal*normal))
    rhs=thrust/(32*rho*area)
    if normal/math.sqrt(2*rhs)<-.28: raise ValueError('ENERGY_MODEL_DOMAIN')
    left=max(0,-normal); right=max(1,left+1,2*math.sqrt(rhs)+abs(normal))
    def equation(q): return q*math.sqrt(plane*plane+(normal+q)**2)-rhs
    if equation(left)>max(1,rhs)*1e-12: raise ValueError('ENERGY_MODEL_DOMAIN')
    while equation(right)<0: right*=2
    for _ in range(80):
        middle=(left+right)/2
        if equation(middle)>0: right=middle
        else: left=middle
    induced=(left+right)/2
    profile=16*rho/8*.055*.0089*area*120.48**3*.97
    shaft=2*thrust*induced+work+profile
    if shaft<=0: raise ValueError('ENERGY_MODEL_DOMAIN')
    return {'battery_w':shaft/.9,'shaft_w':shaft,'current_a':shaft/(.9*85),'induced_velocity_mps':induced}

def checkpoints():
    cases={'hover':((0,0,0),(0,0,0)),'descent_1p5':((0,0,-1.5),(0,0,0))}
    for name,r in (('outer',600),('middle',450),('inner',300)):
        cases[name]=((0,-.035*r,0),(-.035**2*r,0,0))
    return {name:power(*state) for name,state in cases.items()}

if __name__=='__main__':
    print(json.dumps({'provenance':'RECONSTRUCTED_FROM_APPENDIX_A_NOT_SUPPLIED_ORIGINAL','checkpoints':checkpoints()},indent=2))
