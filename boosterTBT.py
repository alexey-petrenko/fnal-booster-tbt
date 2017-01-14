import numpy as np

# Print all BPM names:
def getBPMnames(fname):
    names = []
    with open(fname, 'r') as f:
        line = f.readline()
        while line != "":
            line = f.readline()
            if line.startswith("***NEW BPM"):
                bpm = f.readline()[:-1]
                names.append(bpm)
    return names

def readBPM(fname, BPMname):
    x = np.array([])
    with open(fname, 'r') as f:
        line = f.readline()
        while line != "":
            line = f.readline()
            if line.startswith(BPMname):
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line[:-1] != " " and line != "":
                    line = f.readline()
                    arr = np.fromstring(line, dtype=float, sep=" ")
                    x = np.append(x, arr)
                break
    return x

def detrend(x, Tkick=500, kick_delay=2):
    x1 = np.zeros(np.size(x))
    i_max = len(x)-1
    for i in range(0,i_max+1):
        i1 = int(np.floor(i/Tkick)*Tkick) - kick_delay
        i1 = np.max([0,i1])
        i2 = np.min([i1 + Tkick, i_max])
        x_lin = (x[i1]*(i2-i) + x[i2]*(i-i1))/(i2-i1)
        x1[i]= x[i] - x_lin    
    return x1

def find_FFTpeak(fname, BPMname="B:HST02S", f_window=[0.1, 0.5], T0 = 3000-2, dT = 500):
    x = readBPM(fname, BPMname)
    Turns = np.arange(T0,T0+dT)
    x = np.take(x,Turns)
    N = len(x)
    FFTx = np.fft.fft(x)/N
    FFTx = FFTx[range(int(N/2))]
    FFTampl = abs(FFTx)
    i_min = max(int(f_window[0]*N), 0)
    i_max = min(int(f_window[1]*N), len(FFTampl)-1)
    FFTampl[:i_min] = 0
    FFTampl[i_max:] = 0
    i = np.argmax(FFTampl)
    tune = 0.5*i/(N/2)

    #import matplotlib.pyplot as plt
    
    #freq = 0.5*np.arange(0, N/2)/(N/2)
    #plt.plot(freq, FFTx)
    #plt.plot([tune,tune],[0,0.1],color='red')
    #plt.ylabel("FFT ( "+BPMname+" )")
    #plt.xlabel('Tune')
    ##plt.xlim(tune-0.01,tune+0.01)
    #plt.grid()
    #plt.show()
    
    return tune, FFTx[i]
