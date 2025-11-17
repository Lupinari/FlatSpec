# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:33:57 2025

@author: Lupinari
"""

import time
import socket
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyvisa as visa
from scipy.constants import c
from scipy.optimize import curve_fit

import HEDS
from hedslib.heds_types import *


class Osa(object):
    '''Class for communication with Yokogawa Optical Spectrum Analyzer'''

    MAX_SAMPLING_POINT = 50001

    def __init__(self, host_ip, port, usr, pwd, intf_type='LAN', default_config=False):
        self.intf_type = intf_type
        self.host_ip = host_ip
        self.port = port
        self.usr = usr
        self.pwd = pwd

        self.connect()
            
        if default_config:
            self.set_wavelength_x()
            self.set_wavelength_center_nm(1550)
            self.set_wavelength_span_nm(40)
            self.set_wavelength_resolution_nm(0.02)
            self.abort_sweep()
        

    def connect(self):
        if self.intf_type == 'LAN':
            self.device = socket.socket()

            self.device.connect((self.host_ip, self.port))
            time.sleep(0.6)

            send = 'open "' + self.usr + '"\n'
            self.device.send(send.encode('utf-8'))
            self.device.recv(self.MAX_SAMPLING_POINT)
            time.sleep(0.6)

            send = self.pwd + '\n'
            self.device.send(send.encode('utf-8'))
            self.device.recv(self.MAX_SAMPLING_POINT)
            time.sleep(0.6)
            
            self.sampling_point = self.MAX_SAMPLING_POINT
            self.sampling_point = self.get_sampling_point()
        else:
            resource = self.host_ip
            
            rm = visa.ResourceManager()
            self.device = rm.open_resource(resource)

    def __del__(self):
        self.device.close()
        
    def disconnect(self):
        self.device.close()

    def write(self, cmd_str):
        if self.intf_type == 'LAN':
            cmd_str = cmd_str + '\n'
            return self.device.send(cmd_str.encode('utf-8'))
        else:
            return self.device.write(cmd_str)

    def read(self):
        if self.intf_type == 'LAN':
            return self.device.recv(self.sampling_point)
        else:
            return self.device.read()
    
    def query(self, cmd_str):
        if self.intf_type == 'LAN':
            self.write(cmd_str)
            result = self.read()
            return result
        else:
            return self.device.query(cmd_str)

# General commands

    def set_sampling_point(self,points):
        'Get OSA sampling point. '

        send = f':SENSE:SWEEP:POINTS {points}'
        self.write(send)
        
    def get_sampling_point(self):
        'Get OSA sampling point. '

        send = ':SENSE:SWEEP:POINTS?'
        sp = self.query(send)

        return int(sp)

    def set_wavelength_center_nm(self, wave_center):
        'Set wavelenght center in nm'
        send = ':SENSe:WAVelength:CENTer {:f}NM'.format(wave_center)
        self.write(send)

    def set_frequency_center_ghz(self, freq_center):
        'Set frequency center in GHz'
        send = ':SENSe:WAVelength:CENTer {:f}GHZ'.format(freq_center)
        self.write(send)

    def data(self):
        'Read Data'
        send = ':CALCulate:DATA?'
        return self.query(send)

    def set_frequency_x(self):
        'Set X unit in frequency'
        send = ':UNIT:X FREQuency'
        self.write(send)

    def set_wavelength_x(self):
        'Set X unit in wavelength'
        send = ':UNIT:X WAVelength'
        self.write(send)

    def start_sweep(self):
        'Start sweep'
        send = ':INITiate'
        self.write(send)

    def abort_sweep(self):
        'Stop sweep'
        send = ':abort'
        self.write(send)

    def set_sweep_mode(self, mode):
        'Set sweep mode: '
        send = {'single': ':INITiate:SMODe SINGle',
                'repeat': ':INITiate:SMODe REPeat',
                'auto': ':INITiate:SMODe AUTO'}[mode]
        self.write(send)

    def set_maximum_marker(self):
        'Put a maximum marker on signal'
        send = ':CALCulate:MARKer:MAXimum'
        self.write(send)

    def get_x_value(self, idx):
        'Get X value of marker'
        send = ':CALCulate:MARKer:X? {:d}'.format(idx)
        return float(self.query(send))

    def set_wavelength_span_nm(self, value):
        'Set sweep span in nm'
        self.set_wavelength_x()
        send = ':SENSe:WAVelength:SPAN {:.2f}NM'.format(value)
        self.write(send)

    def set_wavelength_resolution_nm(self, value):
        'Set bandwidth resolution in nm'
        self.set_wavelength_x()
        send = ':SENSe:BWIDth:RESolution {:.2f}NM'.format(value)
        self.write(send)

    def get_operation_status(self):
        'Get the content of the operation status Event register'
        send = ':STATus:OPERation:EVENt?'
        return float(self.query(send))

    def check_operation(self):
        'Check if operation has been completed'
        k = 0
        send = ':STAT:OPER:EVEN?'
        t1 = time.time()
        t2 = 0
        print('Check operation')
        i = 0
        while k == 0:
            self.write(send)
            time.sleep(0.1)
            k = int(self.read())
            #print(f'k = {k}')
            k = k & 1
            #print(f'k masked = {k}')
            t2 = time.time()
            if (t2 - t1 > 10):
                k = 0
                break
            i+=1
        print(f'operation finished after {t2-t1} seconds and {i} checks')
        return k

    def set_sense(self, mode):
        'Set the measurement sensitivity'
        send = {'normal auto': ':INITiate:SMODe NAUT',
                'normal': ':INITiate:SMODe NORM',
                'high': ':INITiate:SMODe HIGH1'}[mode]
        self.write(send)

    def peak_to_ref_level(self):
        'Set peak level as reference'  # and returns the reference power'
        send = ':CALC:MARK:MAX:SRL'
        self.write(send)
        time.sleep(0.4)
#        send = 'CALC:CAT POW'
#        self.s.send(send.encode('utf-8'))
#        send = 'CALC:IMM'
#        self.s.send(send.encode('utf-8'))
#        return self.data()

    def max_lambda(self):
        'Find and return the lambda relative to the maximum power'
        # Set the search function to multi search
        send = ':CALCulate:MARKer:MSEarch 1'
        self.write(send)
        send = ':CALCulate:MARKer:AUTO 1'  # Set the auto search function
        self.write(send)
        # Set the sort order of the multi search detection list
        send = ':CALCulate:MARKer:MSEarch:SORT 1'
        self.write(send)
        send = ':CALCulate:MARKer:MSEarch:THResh 10'  # Set the multi search threshold
        self.write(send)
        send = ':CALCULATE:MARKER:X? 0'  # Queries the lambda of the marker
        self.write(send)
        return float(self.read())

    def get_smsr(self):
        'Set SMSR measuring mode and return the calculated smsr'
        'Returns a vector containing, respectively, the peak lambda, peak level, 2nd peak lambda, 2nd peak level, delta lambda and delta level (SMSR)'
        send = ':CALCULATE:PARAMETER:SMSR:MODE SMSR1'
        self.write(send)
        send = ':CALC:IMM'
        self.write(send)
        data_vector = self.data()
        data_vector = data_vector.decode()
        data_float = np.zeros(len(data_vector.split(','))
                              )  # Data comes as CSV string
        count = 0
        # Separating the CSV numbers and converting to float
        for i in data_vector.split(','):
            data_float[count] = float(i)
            count = count + 1
        return data_float
    
    def get_power(self):
        'Set POWER measuring mode and return the Total Power'
        'Returns a vector containing total power'
        
        send = ':CALCULATE:CATEGORY POWER'
        self.write(send)
        send = ':CALC:IMM'
        self.write(send)
        data_vector = self.data()
        
        data_vector = data_vector.decode() # Data comes as CSV string
        data_float = np.zeros(len(data_vector.split(','))) 
        count = 0
        # Separating the CSV numbers and converting to float
        for i in data_vector.split(','):
            data_float[count] = float(i)
            count = count + 1
        return data_float
    
    
    def smsr_operation(self):
        'Procedure to measure the SMSR'
        self.set_wavelength_resolution_nm(0.02)
        self.set_sweep_mode('single')
        self.start_sweep()
        self.check_operation()
        self.peak_to_ref_level()
        lmbda = self.max_lambda()

        # Put here the OSA check connection

        self.set_sense('high')
        # Zoom in at maximum lambda with  3nm span
        self.set_wavelength_center_nm(lmbda * 1e9)
        self.set_wavelength_span_nm(6)
        self.start_sweep()
        self.check_operation()
        smsr = self.get_smsr()

        # Put here the OSA check connection

        self.set_sweep_mode('repeat')
        self.set_sense('normal')

        return smsr

    def get_trace_x(self, trace):
        'Get trace x axis data of the specified trace: '
        tr = {'A': 'TRA',
              'B': 'TRB',
              'C': 'TRC',
              'D': 'TRD',
              'E': 'TRE',
              'F': 'TRF',
              'G': 'TRG'}[trace]

        send = ':TRACE:X? {}'.format(tr)
        raw_data = self.query(send)
        while raw_data[-2:] != b'\r\n':
            raw_data += self.read()

        raw_data = raw_data.decode()
        data_float = np.zeros(len(raw_data.split(','))
                              )  # Data comes as CSV string
        count = 0
        # Separating the CSV numbers and converting to float
        for i in raw_data.split(','):
            data_float[count] = float(i)
            count += 1
        return data_float

    def get_trace_y(self, trace):
        'Get trace y axis data of the specified trace: '
        tr = {'A': 'TRA',
              'B': 'TRB',
              'C': 'TRC',
              'D': 'TRD',
              'E': 'TRE',
              'F': 'TRF',
              'G': 'TRG'}[trace]

        send = ':TRACE:Y? {}'.format(tr)
        raw_data = self.query(send)
        while raw_data[-2:] != b'\r\n':
            raw_data += self.read()

        raw_data = raw_data.decode()
        data_float = np.zeros(len(raw_data.split(','))
                              )  # Data comes as CSV string
        count = 0
        # Separating the CSV numbers and converting to float
        for i in raw_data.split(','):
            data_float[count] = float(i)
            count += 1
        return data_float
    
    def set_noise_area(self, val):
        send = f':CALCULATE:PARAMETER:WDM:NAREA {val:.2f}NM'
        self.write(send)
    
    def get_osnr(self):
        'Set OSNR measuring mode and return the calculated osnr'
        send = ':CALCULATE:PARAMETER:WDM:SPOWER PEAK'
        self.write(send)
        send = ':CALC:IMM'
        self.write(send)
        data_vector = self.data()
        if self.intf_type == 'LAN':
            data_vector = data_vector.decode()
        data_float = np.zeros(len(data_vector.split(','))
                              )  # Data comes as CSV string
        count = 0
        # Separating the CSV numbers and converting to float
        for i in data_vector.split(','):
            data_float[count] = float(i)
            count = count + 1
        return data_float
    
    def get_raw_osnr(self):
        return self.get_osnr()[-1]
    
    def osnr_operation(self, *lmbda, power_threshold=None,resolution=0.1,bw=100,span=3):
        'Procedure to measure the OSNR'
        self.set_wavelength_resolution_nm(resolution)
        self.set_sweep_mode('single')
        self.start_sweep()
        self.check_operation()
        self.peak_to_ref_level()
        
        if len(lmbda) > 1:
            raise TypeError("osnr_operation()) expected at most 1 argument, got %d"
                            % (len(lmbda) + 1))
        elif len(lmbda) == 1:
            lmbda = lmbda[0]/1e9
        else:
            lmbda = self.max_lambda()    

        self.set_sense('high')
        # Zoom in at maximum lambda with  3nm span
        self.set_wavelength_center_nm(lmbda * 1e9)
        self.set_wavelength_span_nm(span)
        self.start_sweep()
        self.check_operation()
        
        if power_threshold is not None:
            pwr = self.get_power()
            if pwr[0] < power_threshold:
                return None
        
        signal_bw = bw # GHz
        noise_area = (signal_bw*(lmbda*1e9)**2)/(2*c)
        self.set_noise_area(noise_area*1.1) # Noise Area Calculated + 10%
        osnr = self.get_osnr()
    
        self.set_sweep_mode('repeat')
        self.set_sense('normal')

        return osnr
    
    def capture_spectrum_operation(self, trace):
        'Procedure to capture the spectrum of a channel'
        self.set_wavelength_resolution_nm(0.1)
        self.set_wavelength_center_nm(1550)
        self.set_wavelength_span_nm(50)
        self.set_sweep_mode('single')
        self.start_sweep()
        self.check_operation()
        self.peak_to_ref_level()
        lmbda = self.max_lambda()

        # Zoom in at maximum lambda with  3nm span
        self.set_sense('high')
        self.set_wavelength_resolution_nm(0.02)
        self.set_wavelength_center_nm(lmbda * 1e9)
        self.set_wavelength_span_nm(3)
        self.start_sweep()
        self.check_operation()
        
        # capture traces
        lda = self.get_trace_x(trace)
        pwr = self.get_trace_y(trace)
        
        self.set_sweep_mode('repeat')
        self.set_sense('normal')
        
        return lda, pwr
        
    def set_attenuation(self, mode):
        'Set OSA internal attenuation value: '
        send = {'on': ':SENSe:SETting:ATTenuator ON',
                'off': ':SENSe:SETting:ATTenuator OFF'}[mode]
        self.write(send)
    
    def set_connector_type(self, connector):
        'Set connector type (PC, APC, etc) to correctly define OSA internal setup loss: '
        
    def status_clear(self):
        'Clear the OSA internal buffer: '
        send = '*CLS'
        self.write(send)
    
    def set_level_shift(self, value):
        'Set level shifting in dB'
        send = ':SENSe:CORRection:LEVel:SHIFt {:f}DB'.format(value)
        self.write(send)
        
    def get_level_shift(self):
        'Get level shifting in dB'
        send = ':SENSe:CORRection:LEVel:SHIFt?'
        return float(self.query(send))  

###############################################################################

######### Initialize SLM ########
def Boot_SLM():
    pixel_array_len = 1920  #length of pixel array
    pixel_array = np.arange(0, pixel_array_len, 1, dtype=int)
        
        
    # Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
    err = HEDS.SDK.Init(4,0)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        
    # Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
    slm = HEDS.SLM.Init("", True, 0.0)
    assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())
        
    # Configure the calculated image size in pixel:
    dataWidth = pixel_array_len
    dataHeight = 1
        
    # Create an image pixel data field in memory to be shown on an SLM. Each pixel contains an 8-bit integer gray value:
    data = HEDS.SLMDataField(dataWidth, dataHeight, HEDSDTFMT_INT_U8, HEDSSHF_PresentFitScreen)
        
    print('inicializou SLM')
    return data,slm,dataWidth
###############################################################################

######### Initialize OSA ########
osa = Osa('192.168.1.2',10001,'user','yokogawa')
def Boot_OSA(wl_center,wl_span,sampling_pt,wl_res,sense,sweep_mode):
    osa.abort_sweep()
    osa.set_wavelength_center_nm(wl_center)
    osa.set_wavelength_span_nm(wl_span)
    osa.set_sampling_point(sampling_pt)
    osa.set_wavelength_resolution_nm(wl_res)
    osa.set_sense(sense) #normal or high\n",
    osa.set_sweep_mode(sweep_mode)

    #refx = osa.get_trace_x('D')
    #refy = osa.get_trace_y('D')

    print('inicializou OSA')
###############################################################################

def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def Der_sine_function(x,A,B,C):
    return A*B*np.cos(B*x+C)
                  

######### Calibration: Gama ########
def Gama_Calibration(Delta_GS):
    GS_min=0 #Low end of Gray Scale
    GS_max=255 #High end of GS
    Delta_GS #Step size for GS sweep
    GS_Levels = int((GS_max-GS_min)/Delta_GS)+1 #How many GS levels will be measured
    
    GS = GS_min #Inicial GS
    
    Data_points = osa.get_sampling_point() #OSA data points number
    Gama_Specs = np.zeros((GS_Levels+1,Data_points)) #array of arrays containing one spectrum for each GS level Gama_Specs[0][] contains wavelength (y axis) of the spectrum
    
    osa.start_sweep()      
    refx = osa.get_trace_x('D')
    
    Gama_Specs[0][:]=refx[:] #first row contains wavelength y axis
    
    GS_l=0 # Variable counting GS Levels

    
    while GS_l<GS_Levels:
        x=0
        for x in range(0, dataWidth):
            err = data.setPixel(x,0, GS)
            assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    
        err = slm.showImageData(data)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        osa.start_sweep()    
        refy = osa.get_trace_y('D')
        Gama_Specs[GS_l+1][:]=refy[:] # List of arrays, spectrum for each Gray Scale Level (GS_l)
    
        plt.figure()
        plt.title("Gama %d, GS=%d"%(GS_l,GS))
        plt.plot(refx*1e9,refy)
        plt.grid()
        plt.xlabel('Wavelength (nm)')
        #plt.ylabel('Power (dB)')
        #plt.ylim(-80,-40)
        plt.ylabel('Power (nW)')
        plt.ylim(0,3.8e-6)
        plt.show()
    
        GS=GS+Delta_GS
        GS_l=GS_l+1
        np.save('GamaSpecs_%d%d'%(SERIES,DATANAME), Gama_Specs)
    print('Gray Scale Calibration took: %d s'%(time.time()-time_i))
###############################################################################

######### Calibration: Wavelenght per Pixel (SLM) ########
def WL_Calibration(Col_w,Col_ii,Col_sampled):
    
    ## the function starts with a 'blank screen' with value GS_blank. Each collumn 
    ## (containing Col_w pixels) will be assigned value GS_col. OSA takes a spectrum. 
    ## SLM is set to 'blank' again, and the next collumn is assigned valeu GS_col. 
    ## OSA takes a spectrum. And so on, until the SLM surface is completelly swept. 
    
    GS_blank=90 #'Background' reference Gray Scale 
    GS_col=0   # Collumn reference Gray Scale
    Col_w    # Pixel collumn width
    Col_i=Col_ii    # Collumn counter 
    dataWidth=1920 ## !!!
    points_for_wl_calibration=0
    Col_n = int(dataWidth/Col_w) # Number of collumns
    #defining how many collumns of pixels will be used in the calibration (how many points we'll have in the line fitting for the wl calibration)
    
    Col_skip=int(((dataWidth/Col_w)-Col_sampled)/(Col_sampled-1))
    
    Data_points = osa.get_sampling_point() #OSA data points number
    WLCalib_Specs = np.zeros((Col_n+2,Data_points)) #array of arrays containing one spectrum for each highlighted collumn and one more for the reference spec 'blank screen'[1][]. WLCalib_Specs[0][] contains wavelength (x axis) of the spectrum
    
    # set SLM to 'blank surface' value = GS_blank ##
    x=Col_i*Col_w
    for x in range(0, dataWidth):
        err = data.setPixel(x,0, GS_blank)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    err = slm.showImageData(data)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    osa.start_sweep()      
    refx = osa.get_trace_x('D')
    refy = osa.get_trace_y('D')

    WLCalib_Specs[0][:]=refx[:] #first row contains wavelength x axis
    WLCalib_Specs[1][:]=refy[:] #second row contains reference spec 'blank screen'
    
    GS_i=0
    GS_min=0 #Low end of Gray Scale
    GS_max=256 #High end of GS
    Delta_GS=20 #Step size for GS sweep
    GS_Levels = int((GS_max-GS_min)/Delta_GS) #How many GS levels will be measured
    

    x=Col_i*Col_w
    while Col_i<Col_n:
        ## Sweeping collumns##
        plt.figure()
        plt.title('WL Collumn %d (%d pix per coll)'%(Col_i,Col_w))
        #plt.plot(refx*1e9,refy)
        plt.grid()
        plt.xlabel('Wavelength (nm)')
        #plt.ylabel('Power (dB)')
        #plt.ylim(-80,-35)
        plt.ylabel('Power (nW)')
        plt.ylim(0,9e-6)
        
        GS_col=0
    
        # set SLM to 'blank surface' value = GS_blank
        Pix_i=0
        while Pix_i<Col_w: 
            err = data.setPixel(x,0, GS_blank)
            assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
            Pix_i=Pix_i+1
            x=x+1
        err = slm.showImageData(data)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        
        #taking reference spec for that collumn
        osa.start_sweep()  
        refy_0 = osa.get_trace_y('D')
        
        #changing the selected collumn to different GS values 
        while GS_col<GS_max:
            Pix_0=Col_i*Col_w #Starting pixel of the collumn
            x=Col_i*Col_w
            Pix_i=0 #pixel counter inside collumn
            while Pix_i<Col_w: 
                err = data.setPixel(x,0, GS_col) 
                assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
                x=x+1
                Pix_i=Pix_i+1
            err = slm.showImageData(data)
            assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
            x=Col_i*Col_w
        # OSA reading spec
            osa.start_sweep()  
            refy = osa.get_trace_y('D')
            WLCalib_Specs[Col_i+2]=WLCalib_Specs[Col_i+2]+refy
            
            #if np.sum(refy)<np.sum(WLCalib_Specs[1]):
            #    WLCalib_Specs[Col_i+2][:]=WLCalib_Specs[Col_i+2][:]-refy[:] #index Col_i+1 to skip row 0 that contains y axis
            #else:
            #    WLCalib_Specs[Col_i+2][:]=WLCalib_Specs[Col_i+2][:]+refy[:] #index Col_i+1 to skip row 0 that contains y axis
    
            #plt.figure()
            #plt.title('WL Collumn %d'%Col_i)
            plt.plot(refx*1e9,refy)
            #plt.grid()
            #plt.xlabel('Wavelength (nm)')
            ##plt.ylabel('Power (dB)')
            ##plt.ylim(-80,-40)
            #plt.ylabel('Power (nW)')
            #plt.ylim(0,2.5e-5)
        
            #plt.show()
            
            GS_col=GS_col+Delta_GS
            #Pix_0=x #Starting pixel of the collumn
            #Pix_i=0 #pixel counter inside collumn
        WLCalib_Specs[Col_i+2]=WLCalib_Specs[Col_i+2]/refy_0 #divide the sum of the spectra taken by the reference spectrum. 
        WLCalib_Specs[Col_i+2]=np.absolute(WLCalib_Specs[Col_i+2]-np.average(WLCalib_Specs[Col_i+2])) ## subtracts the avarege so that it is centered in zero and takes the absolute value, so that the peak for identification is always positive.
        if np.max(WLCalib_Specs[Col_i+2])>10*np.std(WLCalib_Specs[Col_i+2]):points_for_wl_calibration=points_for_wl_calibration+1 ## counts how many spectra we can use while identifying peaks for the wl calibration. 
        np.save('CalibSpecs_%d%d'%(SERIES,DATANAME),WLCalib_Specs)
        Col_i=Col_i+1+Col_skip
        plt.show()
        
        
    ### identifying wl per pixel ###
        
    WLCalib_wlpercoll = np.zeros((2,points_for_wl_calibration)) #array of arrays containing wl [0][] and corresponding pixel collumn [1][] for wl calibration WLCalib_wlpercoll[0][] contains wavelength (y axis) of the spectrum
    Col_i=Col_ii
    i=0
    while Col_i<Col_n:
        if np.max(WLCalib_Specs[Col_i+2])>10*np.std(WLCalib_Specs[Col_i+2]):
            #WLCalib_wlpercoll[1][i]=Col_i
            WLCalib_wlpercoll[0][i]=WLCalib_Specs[0][np.argmax(WLCalib_Specs[Col_i+2])]
            WLCalib_wlpercoll[1][i]=(Col_i*Col_w+Col_w/2)#int(Col_i*Col_w+Col_w/2) # changing x axis from pixel collumn to center pixel of the collumn
            i=i+1
        Col_i=Col_i+1+Col_skip
    

    
    
    
    
    
    #WLCalib_wlpercoll = np.zeros((2,len(WLCalib_Specs[:])-2)) #array of arrays containing one spectrum for each GS level WLCalib_Specs[0][] contains wavelength (y axis) of the spectrum
       
    #Col_i2=0 #contador de colunas (identificador de cada espectro)
    #Col_i2=2 # Col_i=0: wavelength (x axis); Col_i=1: Reference spec

    #while Col_i2<len(WLCalib_Specs):
    #    wl_locator=np.argmax(WLCalib_Specs[Col_i2])
    #    WLCalib_wlpercoll[0][Col_i2-2]=WLCalib_Specs[0][wl_locator]
    #    WLCalib_wlpercoll[1][Col_i2-2]=Col_i2-2
    #    #WLCalib_coll[Col_i-2]=Col_i-2
    #    #print(wl_locator)
    #    Col_i2=Col_i2+1
    ###

    
    fit_coef=np.polyfit(WLCalib_wlpercoll[1], WLCalib_wlpercoll[0], deg=1) #the fitting is done considering number of pixel (x axis) VS wavelength (y axis) 
    np.save('fit_coef_%d%d'%(SERIES,DATANAME),fit_coef)
    
    np.save('WLperColl_%d%d'%(SERIES,DATANAME),WLCalib_wlpercoll)
    plt.figure()
    plt.title('wl per pixel collumn')
    plt.scatter(WLCalib_wlpercoll[1],WLCalib_wlpercoll[0],color='r')
    plt.plot(WLCalib_wlpercoll[1],fit_coef[1]+fit_coef[0]*WLCalib_wlpercoll[1],color='b')
    plt.ylabel('wavelength')
    plt.xlabel('Pixel number')
    plt.show()
    
    print('Wavelength Calibration took: %d s'%(time.time()-time_i))
    return(WLCalib_Specs,WLCalib_wlpercoll,fit_coef,Col_skip)
###############################################################################

### defining where to equalize the spectrum ###     
def Ruler(Gama_Specs):

    Intens_max=np.zeros(len(Gama_Specs[0]))
    Intens_min=np.zeros(len(Gama_Specs[0]))
    InCollumn=np.zeros(len(Gama_Specs)-1)
    IntensperColl=np.zeros((len(Gama_Specs[0]),len(Gama_Specs)-1)) ## List of arrays, each array contains the intensities (varying with gray scale) in one collumn of sample point
    x=0
    while x in range(len(Gama_Specs[0])):
        y=0
        while y in range(len(Gama_Specs)-1):
            InCollumn[y]=Gama_Specs[y+1][x]
            y=y+1
    
        Intens_max[x]=np.max(InCollumn)
        Intens_min[x]=np.min(InCollumn)
        IntensperColl[x]=InCollumn
        x=x+1

    #plt.figure()
    #plt.title('min and max')
    #plt.plot(Gama_Specs[0], Intens_max)
    #plt.plot(Gama_Specs[0], Intens_min)
    #plt.grid(True)
    #plt.show()

    min_min=np.min(Intens_min)
    max_max=np.max(Intens_max)

    ruler=min_min
    delta_ruler=(max_max-min_min)/100 # define the ruler step with which it will sweep to find the optimum intensity to flatten 
    ruler_ref=0
    counter_ref=0

    i=0
    while ruler<max_max:
        counter=0
        i=0
        while i in range(len(Intens_min)):
            if ruler>Intens_min[i] and ruler<Intens_max[i]:
                counter=counter+1
            if counter>=counter_ref:
                counter_ref=counter
                ruler_ref=ruler
            i=i+1
        ruler=ruler+delta_ruler
        
    plt.figure()
    plt.title('min and max')
    plt.plot(Gama_Specs[0], Intens_max)
    plt.plot(Gama_Specs[0], Intens_min)
    plt.plot([5e-7,5.7e-7], [ruler_ref,ruler_ref])
    plt.grid(True)
    plt.show()
    np.save('IntenseperColl_%d%d'%(SERIES,DATANAME),IntensperColl)
    
    return(ruler_ref,IntensperColl)
###############################################################################

###  Fitting Sine to Intensity x Gray Scale  ###
def FitSin_IntensXGS(Gama_Specs,GS_min,GS_max,Delta_GS ):
    
    Sin_param=np.zeros((len(Gama_Specs[0]),4))
    IntensperColl= np.zeros((len(Gama_Specs[0]),len(Gama_Specs)-1)) 
    
    GS_min=0 #Low end of Gray Scale
    GS_max=256 #High end of GS
    Delta_GS=10 # !!! Step size for GS sweep   
    GS_Levels = int((GS_max-GS_min)/Delta_GS)+1 #How many GS levels will be measured 
    
    GS_Levels=np.arange(GS_min,GS_max ,Delta_GS)
    
    y=0
    while y in range(len(Gama_Specs[0])):
        x=1 
        while x in range(len(Gama_Specs)):
            IntensperColl[y][x-1]=Gama_Specs[x][y]
            x=x+1
            
        ## calculating initial guess ##
        A_i=np.max(IntensperColl[y])-np.min(IntensperColl[y])
        B_i=np.pi/abs(GS_Levels[np.argmax(IntensperColl[y])]-GS_Levels[np.argmin(IntensperColl[y])])
        C_i=0
        D_i=(np.max(IntensperColl[y])+np.min(IntensperColl[y]))/2
        
        params, covariance = curve_fit(sine_function,GS_Levels,IntensperColl[y],p0=[A_i, B_i, C_i, D_i])
        # Extract the fitted parameters
        A_fit, B_fit, C_fit, D_fit = params
        #y_fit = sine_function(GS_Levels , A_fit, B_fit, C_fit, D_fit)
        Sin_param[y]=[A_fit, B_fit, C_fit, D_fit]
    
        y=y+1
    np.save('Sin_param_%d%d'%(SERIES,DATANAME),Sin_param)
    return(Sin_param,GS_Levels)
###############################################################################


###########################         INPUTS          ###########################

## OSA ##
WL_i=5e-7 #inicial wavelength (m)
WL_f=5.7e-7 #final wavelength (m)
WL_span=WL_f-WL_i
WL_center=(WL_f+WL_i)/2
sample_points=801 # OSA sample points

## SLM ##
dataWidth=1920 ##!!! pixels on slm

## For WL calibrations ##
Col_w=15 # width of pixel collumn 
Col_i=0 # starting collumn for calibration
Col_sampled=15 # how many samples to use in the wl calibration

## For Gama calibrations ##
Delta_GS=10 #step size with which Intensity vs. Grayscale Calibration will occur
GS_max=256
GS_min=0

## data name ##
DATANAME=2708
SERIES=3
LOOPS=200 #number of loops

###############################################################################


time_i=time.time()

data,slm,dataWidth=Boot_SLM()
Boot_OSA(WL_center*1e9 ,WL_span*1e9 ,sample_points ,0.5,'high','single') #(wl_center,wl_span,sampling_pt,wl_res,sense,sweep_mode)

WLCalib_Specs,WLCalib_wlpercoll,fit_coef,Col_skip=WL_Calibration(Col_w,Col_i,Col_sampled)


time_i=time.time()
Gama_Calibration(Delta_GS) # Delta gray scale



####### Equalizing ########

print('Start Equalizing')
time_i=time.time()
Gama_Specs=np.load('GamaSpecs_%d%d.npy'%(SERIES,DATANAME))
ruler_ref,IntensperColl=Ruler(Gama_Specs) 
Delta_ruler=ruler_ref*0.01 # range within is accetable and still considered flat

np.save('RulerRef_%d%d'%(SERIES,DATANAME),ruler_ref)

GS_fullrange=np.arange(GS_min,GS_max,1)
GS_measured=np.arange(GS_min,GS_max,Delta_GS) 

Sin_param,GS_Levels=FitSin_IntensXGS(Gama_Specs,GS_min,GS_max,Delta_GS )

Pixels=np.arange(0,dataWidth,1)

Initial_correction_pix=np.zeros(dataWidth)
SamplePoints=np.arange(0,sample_points ,1)

InCollumn_fullrange=np.zeros((len(Gama_Specs[0]),len(GS_fullrange))) ## List of arrays, each array contains the intensities (varying with gray scale) in one collumn of sample point
Initial_correction=np.zeros(len(InCollumn_fullrange))
ToFindCorr=np.zeros((len(Gama_Specs[0]),len(GS_fullrange)))

#### interpolate values in InCollumn so that we have info on each Gray Scale level (Delta_GS=1)
x=0
while x in range(len(IntensperColl)):
    InCollumn_fullrange[x]=np.interp(GS_fullrange, GS_measured, IntensperColl[x]) ##!!!
    y=0
    gs_double=0
    found=0
    found2=0
    intensity_tocompare=InCollumn_fullrange[x][0]
    if (intensity_tocompare-ruler_ref)>0: #intensity on GS=0 above reference level
        while y in range(2*len(GS_fullrange)) and found==0: 
            Sin_thisGS=sine_function(gs_double, Sin_param[x][0], Sin_param[x][1], Sin_param[x][2], Sin_param[x][3])
            Sin_lastGS=sine_function(gs_double-0.5, Sin_param[x][0], Sin_param[x][1], Sin_param[x][2], Sin_param[x][3])
            Sin_nextGS=sine_function(gs_double+0.5, Sin_param[x][0], Sin_param[x][1], Sin_param[x][2], Sin_param[x][3])
            
            if Sin_thisGS -ruler_ref<=0: #if intensity curve crosses ref level
                Initial_correction[x]=int(gs_double)
                found=1
            if Sin_thisGS<Sin_lastGS and found2==0:
                corr=gs_double
                if Sin_thisGS<Sin_nextGS:
                    found2=1
            gs_double=gs_double+0.5    
            y=y+1
        if found==0: #sine never crosses and is always above the ref level 
            Initial_correction[x]=int(corr)
            
            
    if (intensity_tocompare-ruler_ref)<0: #intensity on GS=0 bellow reference level
        while y in range(2*len(GS_fullrange)) and found==0: 
            Sin_thisGS=sine_function(gs_double, Sin_param[x][0], Sin_param[x][1], Sin_param[x][2], Sin_param[x][3])
            Sin_lastGS=sine_function(gs_double-0.5, Sin_param[x][0], Sin_param[x][1], Sin_param[x][2], Sin_param[x][3])
            Sin_nextGS=sine_function(gs_double+0.5, Sin_param[x][0], Sin_param[x][1], Sin_param[x][2], Sin_param[x][3])
            
            if Sin_thisGS-ruler_ref>=0: #if intensity curve crosses ref level
                Initial_correction[x]=int(gs_double)
                found=1
            if Sin_thisGS>Sin_lastGS and found2==0:
                corr=gs_double
                if Sin_thisGS>Sin_nextGS:
                    found2=1
            gs_double=gs_double+0.5    
            y=y+1
        if found==0: #sine never crosses and is always bellow the ref level 
            Initial_correction[x]=int(corr)
    if (intensity_tocompare-ruler_ref)==0:  Initial_correction[x]=int(gs_double)
    
        
    x=x+1

np.save('Initial_correction_%d%d.npy'%(SERIES,DATANAME),Initial_correction)
            
   
x=0
Pixel_c=0 #pixel counter; at the end will contain how many pixels are reflecting the spectra collected by the OSA
Pixel_i='empty' #to flag in which pixel of the SLM the spectrum taken by the OSA starts
Pixel_f=0 #to flag in which pixel of the SLM the spectrum taken by the OSA finishes

### First static correction ###
while x in range(len(Pixels)):
    if fit_coef[1]+fit_coef[0]*Pixels[x]>=WL_i and fit_coef[1]+fit_coef[0]*Pixels[x]<=WL_f: ## if the pixel is reflecting a wl in range of the OSA measurements.
        Pixel_c=Pixel_c+1
        Initial_correction_pix[x]=Initial_correction[round(((fit_coef[1]+fit_coef[0]*x-WL_i)/(WL_span))*(len(SamplePoints)-1))]
        if Pixel_i=='empty': Pixel_i=Pixels[x]
        if Pixel_f<x:Pixel_f=x
    err = data.setPixel(x,0, int(Initial_correction_pix[x]))
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    x=x+1
err = slm.showImageData(data)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)


plt.figure()
plt.title('Initial Correction')
plt.plot(Pixels,Initial_correction_pix)
plt.ylim(GS_min,GS_max)
plt.grid(True)
plt.show()

Correction=Initial_correction
Correction_pix=Initial_correction_pix
Correction_pix_last=Initial_correction_pix
LoopCounter=0
np.save('InLoopCorrPix_%d%d%d'%(SERIES,DATANAME,LoopCounter),Correction_pix)
print('Initial Correction calculated')
print('Entering Loop')



### Correction Loop  interpolating OSA sample points to pixel array ###
OSAinPix=np.zeros((2,(Pixel_f-Pixel_i+1)))
OSAinPix[0]=np.linspace(WL_i,WL_f,(Pixel_f-Pixel_i+1))
while True: 
    osa.start_sweep()      
    refy = osa.get_trace_y('D')
    refx = osa.get_trace_x('D')
    np.save('InLoopSpec_%d%d%d'%(SERIES,DATANAME,LoopCounter),refy)
    OSAinPix[1]=np.interp(OSAinPix[0],refx,refy)
    np.save('OSAinPix_%d%d%d'%(SERIES,DATANAME,LoopCounter),OSAinPix)
    Shift_GS=np.round(10/(LoopCounter+1))+1
    Shift_GS2=0.5
    Shift_GS3=0.25
    x=Pixel_i
    while x<=Pixel_f:
        i=x-Pixel_i
        SP=round((fit_coef[1]+fit_coef[0]*x-WL_i)/((WL_f-WL_i)/(len(SamplePoints)-1))) ## Sample Point for pixel x
        if OSAinPix[1][i]<(ruler_ref-Delta_ruler): ##if intensity is bellow target value
            if Der_sine_function(Correction_pix_last[x], Sin_param[SP][0], Sin_param[SP][1], Sin_param[SP][2])>0: 
                if Correction_pix[x]+Shift_GS<=GS_max: 
                    Correction_pix[x]=Correction_pix[x]+Shift_GS
                    
                if Correction_pix[x-1]+Shift_GS2<=GS_max:
                    Correction_pix[x-1]=Correction_pix[x-1]+Shift_GS2
                if Correction_pix[x+1]+Shift_GS2<=GS_max:
                    Correction_pix[x+1]=Correction_pix[x+1]+Shift_GS2   
                    
                if Correction_pix[x-2]+Shift_GS3<=GS_max:
                    Correction_pix[x-2]=Correction_pix[x-2]+Shift_GS3
                if Correction_pix[x+2]+Shift_GS3<=GS_max:
                    Correction_pix[x+2]=Correction_pix[x+2]+Shift_GS3
                    
            elif Der_sine_function(Correction_pix_last[x], Sin_param[SP][0], Sin_param[SP][1], Sin_param[SP][2])<0:
                if Correction_pix[x]-Shift_GS>=GS_min: 
                    Correction_pix[x]=Correction_pix[x]-Shift_GS
                    
                if Correction_pix[x-1]-Shift_GS2>=GS_min:
                    Correction_pix[x-1]=Correction_pix[x-1]-Shift_GS2
                if Correction_pix[x+1]-Shift_GS2>=GS_min:
                    Correction_pix[x+1]=Correction_pix[x+1]-Shift_GS2
                    
                if Correction_pix[x-2]-Shift_GS3>=GS_min:
                    Correction_pix[x-2]=Correction_pix[x-2]-Shift_GS3
                if Correction_pix[x+2]-Shift_GS3>=GS_min:
                    Correction_pix[x+2]=Correction_pix[x+2]-Shift_GS3
                
        if OSAinPix[1][i]>(ruler_ref+Delta_ruler): ##if intensity is above target value
            if Der_sine_function(Correction_pix_last[x], Sin_param[SP][0], Sin_param[SP][1], Sin_param[SP][2])>0:
                if Correction_pix[x]-Shift_GS>=GS_min: 
                    Correction_pix[x]=Correction_pix[x]-Shift_GS
                    
                if Correction_pix[x-1]-Shift_GS2>=GS_min:
                    Correction_pix[x-1]=Correction_pix[x-1]-Shift_GS2
                if Correction_pix[x+1]-Shift_GS2>=GS_min:
                    Correction_pix[x+1]=Correction_pix[x+1]-Shift_GS2
                    
                if Correction_pix[x-2]-Shift_GS3>=GS_min:
                    Correction_pix[x-2]=Correction_pix[x-2]-Shift_GS3
                if Correction_pix[x+2]-Shift_GS3>=GS_min:
                    Correction_pix[x+2]=Correction_pix[x+2]-Shift_GS3
                
            elif Der_sine_function(Correction_pix_last[x], Sin_param[SP][0], Sin_param[SP][1], Sin_param[SP][2])<0:
                if Correction_pix[x]+Shift_GS<=GS_max: 
                    Correction_pix[x]=Correction_pix[x]+Shift_GS
                    
                if Correction_pix[x-1]+Shift_GS2<=GS_max:
                    Correction_pix[x-1]=Correction_pix[x-1]+Shift_GS2
                if Correction_pix[x+1]+Shift_GS2<=GS_max:
                    Correction_pix[x+1]=Correction_pix[x+1]+Shift_GS2  
                    
                if Correction_pix[x-2]+Shift_GS3<=GS_max:
                    Correction_pix[x-2]=Correction_pix[x-2]+Shift_GS3
                if Correction_pix[x+2]+Shift_GS3<=GS_max:
                    Correction_pix[x+2]=Correction_pix[x+2]+Shift_GS3
        x=x+1        
    x=0
    while x in range(len(Pixels)):
        err = data.setPixel(x,0, round(Correction_pix[x]))
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        x=x+1
    err = slm.showImageData(data)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    plt.figure()
    plt.title('Correction for Loop %d'%LoopCounter)
    plt.plot(Pixels,Correction_pix)
    plt.ylim(GS_min,GS_max)
    plt.grid(True)
    plt.show()            
    np.save('InLoopCorrPix_%d%d%d'%(SERIES,DATANAME,LoopCounter),Correction_pix)
    LoopCounter=LoopCounter+1
    Correction_pix_last=Correction_pix
    print('The loop ran for %d laps, and it took %d s'%(LoopCounter-1,time.time()-time_i))
    if LoopCounter>LOOPS: 
        print('The loop ran for %d laps, and it took %d s'%(LoopCounter-1,time.time()-time_i))
        sys.exit()





###############################################################################