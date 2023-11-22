import numpy as np
import pandas as pd
#from pythermalcomfort.models import pmv_ppd
#from pythermalcomfort.utilities import v_relative
from xgboost.sklearn import XGBRegressor
#from pykalman import KalmanFilter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class ThermalComfort():
    def __init__(self, pmv_model, env_variables):
        """
        pmv_model: json path of XGBoost model
        env_variables: dict() contains keys:-  
                v(air speed), rh(relative humidity), 
                tdb(temp dry bulb/room temp), tr(radiant temp)
        met: int metabolic rate
        """
        self.pmv_model = XGBRegressor()
        self.pmv_model.load_model(pmv_model)
        self.env_variables = env_variables
#        self.met = met
#        self.age_enconding = {'children': 0, 'youth': 1, "young adults": 2 ,'middle-aged adults': 3, 'old adults': 4}
#        self.gen_enconding = {'Male': 1, 'Female': 0}
#        self.race_encoding = {'Asian': 0, 'Caucasian': 1, 'Indian': 2}
#        self.signals = SignalSmoothing()

    def v_relative(self, v, met):
        """Estimates the relative air speed which combines the average air speed of
        the space plus the relative air speed caused by the body movement. Vag is assumed to
        be 0 for metabolic rates equal and lower than 1 met and otherwise equal to
        Vag = 0.3 (M – 1) (m/s)
        Parameters
        ----------
        v : float or array-like
            air speed measured by the sensor, [m/s]
        met : float
            metabolic rate, [met]
        Returns
        -------
        vr  : float or array-like
            relative air speed, [m/s]
        """
    
        return np.where(met > 1, np.around(v + 0.3 * (met - 1), 3), v)
    
    def model_inference(self, iclos, personal_details, trackids, met=1.1):
#        print("THERMALLLLL ", trackids)
        pmvs = np.full(len(trackids), None) #[]
        
#        for i, (clo, p) in enumerate(zip(iclos, personal_details)):
        for i in range(len(trackids)):
            t_id = trackids[i] #p['id']
            
            #pmv=0 for unknwon tracks
            
#            print("Estimating Thermal comfort")
            p = personal_details[i]
            iclo = iclos[i]

#            iclo = self.signals.iclo_smoothing(clo['iclo'], t_id)
            vr = float(self.v_relative(v=self.env_variables['v'], met=met))

            modelip = pd.DataFrame([[iclo, met, vr, self.env_variables['rh'], 
                                    self.env_variables['tdb'], self.env_variables['tr'],
                                    p['age'], p['gender'], p['race']]], 
                                columns=['gtClo', 'gtMet', 'gtAS', 'gtRelHum',
                                                'gtTA', 'gtRT','Age', 'Gender', 
                                                'Race'])
            
            pmv = self.pmv_model.predict(modelip)[0]
            pmv = self.signals.pmv_smoothing(pmv, t_id)
            pmvs[i] = pmv
            #pmvs.append(pmv)
            
        return pmvs


#class SignalSmoothing():
#    def __init__(self):
#        self.iclo_m = dict() ## mean
#        self.iclo_c = dict() ## cov
#        self.iclo_kfilter = dict() ## kalman filter
#        
#        self.pmv_m = dict()
#        self.pmv_c = dict()
#        self.pmv_kfilter = dict()
#        
#        self.cov = 1
#        self.kalmanTransCov = 0.5
#        
#    def iclo_smoothing(self,iclo,personid):
#        if str(personid) not in self.iclo_m.keys():
#            self.iclo_m[str(personid)] = iclo
#            self.iclo_c[str(personid)] = self.cov
#            self.iclo_kfilter[str(personid)] = KalmanFilter(transition_matrices=[1],
#                              observation_matrices=[1],
#                              initial_state_mean=iclo,
#                              initial_state_covariance=self.cov,
#                              observation_covariance=10,
#                              transition_covariance=self.kalmanTransCov)
#            return iclo
#        else:
#            temp_state_means, temp_state_covariances = (self.iclo_kfilter[str(personid)].filter_update(self.iclo_m[str(personid)],
#                                    self.iclo_c[str(personid)],
#                                    observation = iclo,
#                                    observation_covariance = np.asarray([5])))
#            self.iclo_m[str(personid)], self.iclo_c[str(personid)] = temp_state_means, temp_state_covariances
#            
#            return self.iclo_m[str(personid)][0][0]
#
#    def pmv_smoothing(self, pmv, personid):
#        if str(personid) not in self.pmv_m.keys():
#            self.pmv_m[str(personid)] = pmv
#            self.pmv_c[str(personid)] = self.cov
#            self.pmv_kfilter[str(personid)] = KalmanFilter(transition_matrices=[1],
#                              observation_matrices=[1],
#                              initial_state_mean=pmv,
#                              initial_state_covariance=self.cov,
#                              observation_covariance=10,
#                              transition_covariance=self.kalmanTransCov)
#            return pmv
#        else:
#            temp_state_means, temp_state_covariances = (self.pmv_kfilter[str(personid)].filter_update(self.pmv_m[str(personid)],
#                                    self.pmv_c[str(personid)],
#                                    observation = pmv,
#                                    observation_covariance = np.asarray([5])))
#            self.pmv_m[str(personid)], self.pmv_c[str(personid)] = temp_state_means, temp_state_covariances
#            return self.pmv_m[str(personid)][0][0]
#        
