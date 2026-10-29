# Split the C* template into its dipole and even (quadrupole-dominated) parts about the fixed axis; fit separate amplitudes.
exec(open('sn_test.py').read().split("c_iso=chi2(r0)")[0])
from numpy.polynomial import legendre as LG
f=template('s1_U1_F0.2_P0.9_X0.6')
for axname,ax in (('frame axis (0,+15)',vec(0,15)),('Migkas-fit axis (17,-7)',vec(17,-7))):
    mu_=n@ax;psi=np.degrees(np.arccos(np.clip(mu_,-1,1)))
    ps=np.linspace(2.5,177.5,71);full=np.array([[f([[p_,zz]])[0] for p_ in ps] for zz in [0.05]])[0]
    cos=np.cos(np.radians(ps));c=LG.legfit(cos,full,6);odd=c.copy();odd[0::2]=0;even=c.copy();even[1::2]=0;even[0]=0
    td=-5/np.log(10)*LG.legval(mu_,odd);te=-5/np.log(10)*LG.legval(mu_,even)
    o=np.ones_like(r0);P=Ci-np.outer(Ci@o,o@Ci)/(o@Ci@o);X=np.c_[td,te];F=X.T@P@X;bt=np.linalg.solve(F,X.T@P@r0);se=np.sqrt(np.diag(np.linalg.inv(F)))
    print(f'{axname}: template dipole part {100*c[1]:+.1f}% (cos psi), even part P2 {100*c[2]:+.1f}%; fitted amplitudes: dipole {bt[0]:+.2f}+-{se[0]:.2f}, even {bt[1]:+.2f}+-{se[1]:.2f} (prediction 1, 1)')
