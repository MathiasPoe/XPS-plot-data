"""
Software to plot XPS data, aquired with SpecsLab Prodigy, Version 4.43.2-r73078

Save spectrum as .vms and .xy

The fitting of the date is done with Fityk (https://fityk.nieto.pl/) in the following way:
- Open filename.xy
- Fit background with function f(x).
- Create ideal background from function f and datapoints and save this background data as filename.backgr
- Subtract background from data
- Fit peaks in data
- Save peak parameters as filename.peaks
"""

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import tkinter as tk
import os
from tkinter import filedialog
import sys


def load_fit_parameters(parameter_file):
    parameters = pd.read_csv(parameter_file, delimiter='\t', engine='python')
    parameter_df = pd.DataFrame(parameters)
    parameter_df['Number'] = [item.split()[0] for item in parameter_df['# PeakType']]
    parameter_df['PeakType'] = [item.split()[1] for item in parameter_df['# PeakType']]
    parameter_df.drop(labels='# PeakType', axis=1, inplace=True)
    df_keys = [key.replace('#', '').replace(' ', '').replace('.', '') for key in list(parameter_df.keys())]
    parameter_df.columns = df_keys
    parameter_df.set_index('Number', inplace=True)
    return parameter_df


def load_data(data_file):
    # number of header lines must be adjusted
    data = pd.read_csv(data_file, header=43, delimiter='  ', engine='python')
    data_df_temp = pd.DataFrame(data)
    Energies = data_df_temp['#'].index.values
    Counts = data_df_temp['#'].values
    data_df = pd.DataFrame({'BindingEnergy': Energies, 'Counts': Counts})
    data_df.set_index('BindingEnergy', inplace=True)
    data_df.sort_index(axis=0)
    return data_df


def load_background(data_background):
    background = pd.read_csv(data_background, delimiter=' ', header=None, engine='python')
    background_df = pd.DataFrame(background)
    keys = ['Counts{}'.format(n) for n, i in enumerate(background_df.keys())]
    keys[0] = 'BindingEnergy'
    background_df.columns = keys
    background_df.set_index('BindingEnergy', inplace=True)
    background_df.sort_index(axis=0)
    counts_new = np.zeros(len(background_df.index.values))
    for key in background_df:
        counts_new += background_df[key]
    background_df_new = pd.DataFrame({'BindingEnergy': background_df.index.values, 'Counts': counts_new})
    background_df_new.set_index('BindingEnergy', inplace=True)
    return background_df_new


# fit models
def PseudoVoigt(x, height, center, hwhm, shape):
    return height * (((1 - shape) * np.exp(- np.log(2) * ((x - center) / hwhm) ** 2)) + shape / (1 + ((x - center) / hwhm) ** 2))


def Linear(x, offset, slope):
    return slope * x + offset


def Lorentzian(x, height, center, shape):
    return height / (1 + ((x - center) / shape) ** 2)


def Gaussian(x, height, center, shape):
    return height * np.exp(- np.log(2) * ((x - center) / shape) ** 2)


def Sigmoid(x, lower, upper, xmid, wsig):
    return lower + (upper - lower) / (1 + np.exp(-(x - xmid) / wsig))


def Cubic(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*x**2 + a3*x**3


# get path
root = tk.Tk()
root.withdraw()
path = filedialog.askdirectory()
root.destroy()
os.chdir(path)
print(path)

# get filename(s)
root = tk.Tk()
root.withdraw()
dat = filedialog.askopenfilenames(filetypes=[('XPS-file', '*.vms'), ('All files', '*.*')])
root.destroy()
file_names = [data.split('/')[-1].split('.')[0] for data in dat]

for file_name in file_names:
    print(file_name)
    plt.figure(figsize=(10, 6), dpi=100, tight_layout=True)
    plt.xlabel('Binding Energy [eV]')
    plt.ylabel('Counts [arb. unit]')
    ax = plt.gca()
    ax.invert_xaxis()
    try:
        data = load_data(file_name + '.xy')
        data.sort_index()
        try:
            background = load_background(file_name + '.backgr')
            data['Counts'] -= background['Counts']
        except FileNotFoundError:
            print('No background data found.')
            pass
        plt.plot(data.index.values, data['Counts'], '.', mfc='none', mew=2, ms=13,
                 label='Data: {}'.format(file_name.split(' - ')[-1]), alpha=0.4)
        try:
            parameter = load_fit_parameters(file_name + '.peaks')
            fit_sum = np.zeros(len(data.index.values))
            for number, PT in enumerate(parameter['PeakType']):
                if PT == 'PseudoVoigt':
                    paras = np.array(parameter['parameters'][number].split(), float)
                    plt.plot(data.index.values, PseudoVoigt(data.index.values, *paras),
                             label='{}_{}\nCenter: {}'.format(PT, number, paras[1]))
                    fit_sum += PseudoVoigt(data.index.values, *paras)
                if PT == 'Linear':
                    paras = np.array(parameter['parameters'][number].split(), float)
                    plt.plot(data.index.values, Linear(data.index.values, *paras), label='{}_{}'.format(PT, number))
                    fit_sum += Linear(data.index.values, *paras)
                if PT == 'Lorentzian':
                    paras = np.array(parameter['parameters'][number].split(), float)
                    plt.plot(data.index.values, Lorentzian(data.index.values, *paras),
                             label='{}_{}\nCenter: {}'.format(PT, number, paras[1]))
                    fit_sum += Lorentzian(data.index.values, *paras)
                if PT == 'Gaussian':
                    paras = np.array(parameter['parameters'][number].split(), float)
                    plt.plot(data.index.values, Gaussian(data.index.values, *paras),
                             label='{}_{}\nCenter: {}'.format(PT, number, paras[1]))
                    fit_sum += Gaussian(data.index.values, *paras)
                if PT == 'Sigmoid':
                    paras = np.array(parameter['parameters'][number].split(), float)
                    plt.plot(data.index.values, Sigmoid(data.index.values, *paras),
                            label='{}_{}\nCenter: {}'.format(PT, number, paras[2]))
                    fit_sum += Sigmoid(data.index.values, *paras)
                if PT == 'Cubic':
                    paras = np.array(parameter['parameters'][number].split(), float)
                    plt.plot(data.index.values, Cubic(data.index.values, *paras), label='{}_{}'.format(PT, number))
                    fit_sum += Cubic(data.index.values, *paras)
            if number > 0:
                plt.plot(data.index.values, fit_sum, label='Sum of fits')
        except FileNotFoundError:
            print('No fit parameter found.')
            pass
    except FileNotFoundError:
        sys.exit('No .xy file from this data found!')
    plt.legend()
    plt.axhline(y=0, color='k')
    plt.savefig(file_name + '.png')
    print('Image saved to ' + file_name + '.png')
    plt.show()
