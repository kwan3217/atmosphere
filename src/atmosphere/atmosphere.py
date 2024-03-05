"""
Base class for atmosphere models. These take a geometric height above a reference surface
and return the atmosphere properties at that height.
"""
from dataclasses import dataclass

import numpy as np
from numpy import log as ln


@dataclass
class AirProperties:
    Altitude:float      # Geometric Altitude above reference surface, m
    Temperature:float   # units:K
    Pressure:float      # units:Pa
    Density:float       # units:kg/m**3
    VSound:float        # units:m/s
    MolWt:float         # units:kg/kmol (=g/mol)
    Geopotential:float  # units:geopotenital meters (m'). In a uniform gravity field,
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
    MolTemp:float       # units:K'. This is the temperature scaled by the molecular weight
    Gravity:float       # units:m/s**2
    PScaleHeight:float  # units:m
    rhoScaleHeight:float# units:m
    NumberDensity:float # units:1/m**3
    MolVel:float        #units:m/s
    MeanFreePath:float  #units:m
    ColFreq:float       #units:Hz
    Viscosity:float     #units:Ns/m**2
    KinViscosity:float  #units:m**2/s
    ThermalCond:float   #units:W/(m*K)
    SpecHeatRatio:float #unitless
    GasNumberDensity:list[float]
    GasMolWeight:list[float]
    GasName:list[str]


def barometric_lapse(h:float,*,P0:float,T0:float,h0:float,L:float,g0:float,M:float)->float:
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
    def calc_props(self,Z:float):
        raise NotImplementedError


class SimpleAtmosphere(Atmosphere):
    k=1.380_649e-23 #Boltzmann constant, 2019 value (exact), J/K
    Na=6.022_140_76e26 #Avogadro's number, 2019 value (exact), 1/kmol
    Rs=k*Na #Ideal gas constant,J/(K*kmol)
    def __init__(self):
        pass
    def temp(self,alt:float)->float:
        raise NotImplementedError
    def mol_weight(self,alt:float)->float:
        raise NotImplementedError
    def pressure(self,alt:float)->float:
        raise NotImplementedError
    def viscosity(self,alt:float):
        """
        Calculate dynamic viscosity \mu at given geometric altitude
        :param alt:
        :return: Viscosity calculated from Sutherland's law, an empirical
                 approximation for how viscosity varies with temperature
        """
        T=self.temp(alt)
        return (self.beta*T**1.5)/(T+self.S)
    def density(self,alt:float):
        """

        :param alt:
        :return:
        """
        T=self.temp(alt)
        if T==0.0:
            return 0.0
        return self.mol_weight(alt)*self.pressure(alt)/(self.Rs*T)
    def vsound(self,alt:float):
        M=self.mol_weight(alt)
        if M==0.0:
            return 0.0
        return np.sqrt(self.gamma*self.Rs*self.temp(alt)/M)
    def mean_free_path(self,alt:float):
        return 1.0 / (np.sqrt(2.0) * np.pi * self.Na * self.sigma**2) * (self.Rs / self.mol_weight(0) * self.mol_weight(alt)
            * self.mol_temp(alt) / self.pressure(alt))
    def mol_temp(self,alt:float):
        return self.temp(alt)*self.mol_weight(alt)/self.mol_weight(0)
    def pres_scale_height(self,alt:float):
        return self.Rs * self.temp(alt) / (self.mol_weight(alt) * self.g0)
    def dens_scale_height(self,alt:float):
        Hp=self.pres_scale_height(alt)
        dz=1.0
        dlnT=ln(self.temp(alt+dz))-ln(self.temp(alt))
        dlnTdz=dlnT/dz
        Hrho=Hp/(1+Hp*dlnTdz)
        return Hrho
    def gravity(self,Z:float):
        return self.g0*(self.r0/(self.r0+Z))**2
    def geopotential(self,Z:float):
        return self.r0*Z/(self.r0+Z)
    def calc_props(self,Z:float):
        return AirProperties(Altitude=Z,Temperature=self.temp(Z),Pressure=self.pressure(Z),Density=self.density(Z),
                             VSound=self.vsound(Z),MolWt=self.mol_weight(Z),Geopotential=self.geopotential(Z),
                             MolTemp=self.mol_temp(Z),Gravity=self.gravity(Z),PScaleHeight=self.pres_scale_height(Z),
                             rhoScaleHeight=self.dens_scale_height(Z),
                             NumberDensity=self.density(Z)/self.mol_weight(Z)*self.Na,MolVel=None,
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
    def temp(self,alt:float):
        if alt<11000:
            return 288.14-0.00694*alt
        elif alt<25000:
            return 216.64
        elif alt<=self.Zlimit:
            return 141.89+0.00299*alt
        else:
            return 0.0
    def mol_weight(self,alt:float):
        return 28.9644
    def pressure(self,alt:float):
        if alt<11000:
            return 101290*(self.temp(alt)/288.08)**5.256
        elif alt<25000:
            return 22650*np.exp(1.73-0.000157*alt)
        elif alt<=self.Zlimit:
            return 2488*(self.temp(alt)/216.6)**-11.388
        else:
            return 0.0


def main():
    print(SimpleEarthAtmosphere().calc_props(0.0))
    print(SimpleEarthAtmosphere().calc_props(12000.0))
    print(SimpleEarthAtmosphere().calc_props(149999.0))


if __name__=='__main__':
    main()