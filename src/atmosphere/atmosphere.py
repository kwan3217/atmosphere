"""
Base class for atmosphere models. These take a geometric height above a reference surface
and return the atmosphere properties at that height.

All of these functions and classes are documented to use np.ndarray, but will accept scalars
(float or int) as well and will return float results in these cases.
"""
from dataclasses import dataclass

import numpy as np
from numpy import log as ln


@dataclass
class AirProperties:
    Altitude:np.ndarray      # Geometric Altitude above reference surface, m
    Temperature:np.ndarray   # units:K
    Pressure:np.ndarray      # units:Pa
    Density:np.ndarray       # units:kg/m**3
    VSound:np.ndarray        # units:m/s
    MolWt:np.ndarray         # units:kg/kmol (=g/mol)
    Geopotential:np.ndarray  # units:geopotenital meters (m'). In a uniform gravity field,
                        # an object which is pushed up gains gravitational potential
                        # energy as mgh. Since real gravity fields are not uniform,
                        # but much more closely approximated as inverse square, raising
                        # an object will not gain as much potential energy, because there
                        # is less gravity at the top end of its travel. An object at a
                        # geopotential altitude h will have as much potential energy
                        # as it would have if it was raised h meters in a uniform gravity
                        # field. This will in general be different from the geometric
                        # altitude z -- an object will have to be raised *more* geometric
                        # height to get an equivalent amount of geopotential height, since
                        # there is less gravity and therefore less potential energy at the top.
    MolTemp:np.ndarray       # units:K'. This is the temperature scaled by the molecular weight
    Gravity:np.ndarray       # units:m/s**2
    PScaleHeight:np.ndarray  # units:m
    rhoScaleHeight:np.ndarray# units:m
    NumberDensity:np.ndarray # units:1/m**3
    MolVel:np.ndarray        #units:m/s
    MeanFreePath:np.ndarray  #units:m
    ColFreq:np.ndarray       #units:Hz
    Viscosity:np.ndarray     #units:Ns/m**2
    KinViscosity:np.ndarray  #units:m**2/s
    ThermalCond:np.ndarray   #units:W/(m*K)
    SpecHeatRatio:np.ndarray #unitless
    GasNumberDensity:list[np.ndarray]
    GasMolWeight:list[np.ndarray]
    GasName:list[str]


def barometric_lapse(h:np.ndarray,*,P0:np.ndarray,T0:np.ndarray,h0:np.ndarray,L:np.ndarray,g0:np.ndarray,M:np.ndarray)->np.ndarray:
    """
    Calculate the pressure at a given altitude in a layer of atmosphere with a linear temperature lapse.

    :param h: Altitude at which to calculate the pressure
    :param h0: Reference altitude, altitude of the base of the layer with linear temperature lapse
    :param P0: Pressure at the reference altitude
    :param T0: Temperature at the reference altitude
    :param L: Vertical temperature lapse rate
    :param g0: acceleration of gravity
    :param M: Molecular mass, kg/kmol
    :return:
    """
    if L==0:
        P=P0*np.exp(-g0*M*(h-h0)/(SimpleAtmosphere.Rs*T0))
    else:
        P=P0*((T0+(h-h0)*L)/T0)**(-g0*M/(SimpleAtmosphere.Rs*L))
    return P


class Atmosphere:
    def calc_props(self,Z:np.ndarray):
        raise NotImplementedError


class SimpleAtmosphere(Atmosphere):
    k=1.380_649e-23 #Boltzmann constant, 2019 value (exact), J/K
    Na=6.022_140_76e26 #Avogadro's number, 2019 value (exact), 1/kmol
    Rs=k*Na #Ideal gas constant,J/(K*kmol)
    def __init__(self):
        pass
    def temp(self,alt:np.ndarray)->np.ndarray:
        raise NotImplementedError
    def mol_weight(self,alt:np.ndarray)->np.ndarray:
        raise NotImplementedError
    def pressure(self,alt:np.ndarray)->np.ndarray:
        raise NotImplementedError
    def viscosity(self,alt:np.ndarray):
        """
        Calculate dynamic viscosity \mu at given geometric altitude
        :param alt:
        :return: Viscosity calculated from Sutherland's law, an empirical
                 approximation for how viscosity varies with temperature
        """
        T=self.temp(alt)
        return (self.beta*T**1.5)/(T+self.S)
    def density(self,alt:np.ndarray):
        """

        :param alt:
        :return:
        """
        T=self.temp(alt)
        return np.where(T==0.0,
                        0.0,
                        self.mol_weight(alt)*self.pressure(alt)/(self.Rs*T))
    def vsound(self,alt:np.ndarray):
        M=self.mol_weight(alt)
        return np.where(M==0.0,0.0,np.sqrt(self.gamma*self.Rs*self.temp(alt)/M))
    def mean_free_path(self,alt:np.ndarray):
        P=self.pressure(alt)
        return np.where(P==0,
            float('inf'),
            1.0 / (np.sqrt(2.0) * np.pi * self.Na * self.sigma**2) * (self.Rs / self.mol_weight(0) * self.mol_weight(alt)
            * self.mol_temp(alt) / self.pressure(alt)))
    def mol_temp(self,alt:np.ndarray):
        return self.temp(alt)*self.mol_weight(alt)/self.mol_weight(0)
    def pres_scale_height(self,alt:np.ndarray):
        return np.where(self.mol_weight(alt)>0,self.Rs * self.temp(alt) / (self.mol_weight(alt) * self.g0),np.inf)
    def dens_scale_height(self,alt:np.ndarray):
        Hp=self.pres_scale_height(alt)
        dz=1.0
        dlnT=ln(self.temp(alt+dz))-ln(self.temp(alt))
        dlnTdz=dlnT/dz
        Hrho=Hp/(1+Hp*dlnTdz)
        return np.where(np.isfinite(Hp),Hrho,np.inf)
    def gravity(self,Z:np.ndarray):
        return self.g0*(self.r0/(self.r0+Z))**2
    def geopotential(self,Z:np.ndarray):
        return self.r0*Z/(self.r0+Z)
    def calc_props(self,Z:np.ndarray):
        return AirProperties(Altitude=Z,Temperature=self.temp(Z),Pressure=self.pressure(Z),Density=self.density(Z),
                             VSound=self.vsound(Z),MolWt=self.mol_weight(Z),Geopotential=self.geopotential(Z),
                             MolTemp=self.mol_temp(Z),Gravity=self.gravity(Z),PScaleHeight=self.pres_scale_height(Z),
                             rhoScaleHeight=self.dens_scale_height(Z),
                             NumberDensity=np.where(self.density(Z)>0,self.density(Z)/self.mol_weight(Z)*self.Na,0),MolVel=None,
                             MeanFreePath=self.mean_free_path(Z),ColFreq=None,Viscosity=self.viscosity(Z),
                             KinViscosity=None,ThermalCond=None,SpecHeatRatio=self.gamma,GasNumberDensity=[],
                             GasMolWeight=[],GasName=[])


class SimpleEarthAtmosphere(SimpleAtmosphere):
    gamma=1.40 #Specific heat ratio, unitless, theoretical value for diatomic gases
    S=110.4 # Sutherland's constant, K, for Earth air. Constant has a different value for each gas or mixture of gases
    beta=1.458e-6 #Viscosity coefficient beta, kg/(s*m*sqrt(K))
    sigma=3.65e-10 #Effective collision diameter, m
    g0=9.80665 #Acceleration of gravity at reference surface, m/s**2
    r0=6_356_766.0 #Reference radius of Earth, from USSA1976.
    Zlimit=150_000.0 #Limit of atmosphere model, m
    def temp(self,alt:np.ndarray):
        return np.where(
            alt<11000,
                288.14 - 0.00694 * alt,
            np.where(alt<25000,
                216.64,
            np.where(alt<=self.Zlimit,
                141.89 + 0.00299 * alt,
                0.0
        )))
    def mol_weight(self,alt:np.ndarray):
        return np.where(alt<=self.Zlimit,28.9644,0)
    def pressure(self,alt:np.ndarray):
        return np.where(
            alt<11000,
                101290*(self.temp(alt)/288.08)**5.256,
            np.where(alt<25000,
                22650*np.exp(1.73-0.000157*alt),
            np.where(alt<=self.Zlimit,
                2488*(self.temp(alt)/216.6)**-11.388,
                0.0
        )))


