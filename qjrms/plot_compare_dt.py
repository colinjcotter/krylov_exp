import numpy as np
import matplotlib.pyplot as plt
import data_lev5_dt025
import data_lev5_dt0125
import data_lev5_dt0375
from data_lev5_dt025 import *
from data_lev5_dt0125 import *
from data_lev5_dt0375 import *

# standard model results
eta_1350s = [0.00186908343558815, 0.0029236867056528708, 0.004092992645657903, 0.004259656570998618]
u_1350s = [0.0020939275941023795, 0.0032455969161292696, 0.004584794354643949, 0.006699830988348994]

eta_900s = [0.0011678662042723888, 0.0017860061124668875, 0.002234454669789315, 0.00274443210083982]
u_900s = [0.0012973803949937785, 0.0019736018123054943, 0.0024921658929822365, 0.003069718801010646]

eta_450s = [0.000482154999575752, 0.0007650883758127529, 0.0010018594716208551, 0.001313043595896499] 
u_450s = [0.0005315923730401088, 0.0008401858482484554, 0.0011048694866359933, 0.0012381093634108319]

# parameters
PPP = 4
DTS = 22.5
DAY_MAX = 1.0
DAYS_TO_PLOT = [1.0]
AVERAGING_WINDOW_MAX0 = 1.0
AVERAGING_WINDOW_MAX1 = 1.0
AVERAGING_WINDOW_MAX2 = 1.0
LOGLOG = True
LINEWIDTH = 1
DRAW_ZOOM = False
AVERAGING_WINDOW_ZOOM = 1.0

# set days and averaging window
DAY_INT = 0.25
DAYS = []
imax = int(DAY_MAX/DAY_INT)
i = 1
while i <= imax:
    DAYS.append(i*DAY_INT)
    i += 1

AVERAGING_WINDOW_ALL0 = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15, 17.5, 20]
AVERAGING_WINDOW0 = []
for k in AVERAGING_WINDOW_ALL0:
    if k <= AVERAGING_WINDOW_MAX0:
        AVERAGING_WINDOW0.append(k)

AVERAGING_WINDOW_ALL1 = [0.1, 0.10625, 0.1125, 0.11875, 0.125, 0.13125, 0.1375, 0.14375, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15, 17.5, 20]
AVERAGING_WINDOW1 = []
for k in AVERAGING_WINDOW_ALL1:
    if k <= AVERAGING_WINDOW_MAX1:
        AVERAGING_WINDOW1.append(k)

AVERAGING_WINDOW_ALL2 = [0.375, 0.4125, 0.45, 0.4875, 0.525, 0.5625, 0.6, 0.6375, 0.675, 0.7125, 0.75, 0.825, 0.9, 0.975, 1.05]
AVERAGING_WINDOW2 = []
for k in AVERAGING_WINDOW_ALL2:
    if k <= AVERAGING_WINDOW_MAX2:
        AVERAGING_WINDOW2.append(k)

# input data
etadict0 = {}
udict0 = {}
input_lev5_dt025(etadict0, udict0, ppp=PPP, dts=DTS)
DT0 = 900

etadict1 = {}
udict1 = {}
input_lev5_dt0125(etadict1, udict1, ppp=PPP, dts=DTS)
DT1 = 450

etadict2 = {}
udict2 = {}
input_lev5_dt0375(etadict2, udict2, ppp=PPP, dts=DTS)
DT2 = 1350

# sort data by day
etanorm0 = {}
unorm0 = {}
i = 0
for day in DAYS:
    etanorm0[day] = []
    unorm0[day] = []
    for key in AVERAGING_WINDOW0:
        etanorm0[day].append(etadict0[key][i])
        unorm0[day].append(udict0[key][i])
    i += 1

etanorm1 = {}
unorm1 = {}
i = 0
for day in DAYS:
    etanorm1[day] = []
    unorm1[day] = []
    for key in AVERAGING_WINDOW1:
        etanorm1[day].append(etadict1[key][i])
        unorm1[day].append(udict1[key][i])
    i += 1

etanorm2 = {}
unorm2 = {}
i = 0
for day in DAYS:
    etanorm2[day] = []
    unorm2[day] = []
    for key in AVERAGING_WINDOW2:
        etanorm2[day].append(etadict2[key][i])
        unorm2[day].append(udict2[key][i])
    i += 1

etanorm_900s = {}
unorm_900s = {}
i = 0
for day in DAYS:
    etanorm_900s[day] = []
    unorm_900s[day] = []
    for key in AVERAGING_WINDOW0:
        etanorm_900s[day].append(eta_900s[i])
        unorm_900s[day].append(u_900s[i])
    i += 1

etanorm_1350s = {}
unorm_1350s = {}
i = 0
for day in DAYS:
    etanorm_1350s[day] = []
    unorm_1350s[day] = []
    for key in AVERAGING_WINDOW2:
        etanorm_1350s[day].append(eta_1350s[i])
        unorm_1350s[day].append(u_1350s[i])
    i += 1

etanorm_450s = {}
unorm_450s = {}
i = 0
for day in DAYS:
    etanorm_450s[day] = []
    unorm_450s[day] = []
    for key in AVERAGING_WINDOW1:
        etanorm_450s[day].append(eta_450s[i])
        unorm_450s[day].append(u_450s[i])
    i += 1

##################################
#     PLOTTONG ENTIRE FIGURE     #
##################################
markers = ["+", "s", "o", "v", "^", "<", ">", "1", "2", "3"]
colors = ["black", "red", "blue", "green", "magenta", "yellow", "black", "cyan"]

plt.figure(num=1,figsize=(8,7))

i = 0
for day in DAYS_TO_PLOT:
    if LOGLOG:
        plt.loglog(AVERAGING_WINDOW2, unorm_1350s[day], color=colors[i], linestyle="dotted", linewidth=LINEWIDTH, label = r'dt = '+str(DT2)+' s (standard)')
        plt.loglog(AVERAGING_WINDOW0, unorm_900s[day], color=colors[i], linestyle="solid", linewidth=LINEWIDTH, label = r'dt = '+str(DT0)+' s (standard)')
        plt.loglog(AVERAGING_WINDOW1, unorm_450s[day], color=colors[i], linestyle="dashed", linewidth=LINEWIDTH, label = r'dt = '+str(DT1)+' s (standard)')
        plt.loglog(AVERAGING_WINDOW2, unorm2[day], marker=markers[4], color=colors[i], markersize=3, linestyle="dotted", linewidth=LINEWIDTH, label = r'dt = '+str(DT2)+' s (averaged)')
        plt.loglog(AVERAGING_WINDOW0, unorm0[day], marker=markers[4], color=colors[i], markersize=3, linestyle="solid", linewidth=LINEWIDTH, label = r'dt = '+str(DT0)+' s (averaged)')
        plt.loglog(AVERAGING_WINDOW1, unorm1[day], marker=markers[4], color=colors[i], markersize=3, linestyle="dashed", linewidth=LINEWIDTH, label = r'dt = '+str(DT1)+' s (averaged)')
    else:
        plt.plot(AVERAGING_WINDOW2, unorm2[day], marker=markers[0], color=colors[i], markersize=5, linestyle="dotted", linewidth=LINEWIDTH, label = r'dt = '+str(DT2)+' s')
        plt.plot(AVERAGING_WINDOW0, unorm0[day], marker=markers[0], color=colors[i], markersize=5, linestyle="solid", linewidth=LINEWIDTH, label = r'dt = '+str(DT0)+' s')
        plt.plot(AVERAGING_WINDOW1, unorm1[day], marker=markers[0], color=colors[i], markersize=5, linestyle="dashed", linewidth=LINEWIDTH, label = r'dt = '+str(DT1)+' s')
        plt.plot(AVERAGING_WINDOW2, unorm_1350s[day], color=colors[i], linestyle="dotted", linewidth=LINEWIDTH, label = r'dt = '+str(DT2)+' s')
        plt.plot(AVERAGING_WINDOW0, unorm_900s[day], color=colors[i], linestyle="solid", linewidth=LINEWIDTH, label = r'dt = '+str(DT0)+' s')
        plt.plot(AVERAGING_WINDOW1, unorm_450s[day], color=colors[i], linestyle="dashed", linewidth=LINEWIDTH, label = r'dt = '+str(DT1)+' s')
    i += 1

plt.title('Error in $u$ at day 1', fontsize=18)
plt.legend(prop={'size':10}, loc="lower right")

plt.xlabel(r'Averaging Window $T$ (hour)', fontsize=18)
plt.ylabel(r'Normalised error', fontsize=18)
plt.ylim(1.e-5,1.e-2)
plt.yscale('log')


plt.savefig('lev5_dt_compare_u.eps')
plt.show()

plt.figure(num=2,figsize=(8,7))

i = 0
for day in DAYS_TO_PLOT:
    if LOGLOG:
        plt.loglog(AVERAGING_WINDOW2, etanorm_1350s[day], color=colors[i], linestyle="dotted", linewidth=LINEWIDTH, label = r'dt = '+str(DT2)+' s (standard)')
        plt.loglog(AVERAGING_WINDOW0, etanorm_900s[day], color=colors[i], linestyle="solid", linewidth=LINEWIDTH, label = r'dt = '+str(DT0)+' s (standard)')
        plt.loglog(AVERAGING_WINDOW1, etanorm_450s[day], color=colors[i], linestyle="dashed", linewidth=LINEWIDTH, label = r'dt = '+str(DT1)+' s (standard)')
        plt.loglog(AVERAGING_WINDOW2, etanorm2[day], marker=markers[0], color=colors[i], markersize=5, linestyle="dotted", linewidth=LINEWIDTH, label = r'dt = '+str(DT2)+' s (averaged)')
        plt.loglog(AVERAGING_WINDOW0, etanorm0[day], marker=markers[0], color=colors[i], markersize=5, linestyle="solid", linewidth=LINEWIDTH, label = r'dt = '+str(DT0)+' s (averaged)')
        plt.loglog(AVERAGING_WINDOW1, etanorm1[day], marker=markers[0], color=colors[i], markersize=5, linestyle="dashed", linewidth=LINEWIDTH, label = r'dt = '+str(DT1)+' s (averaged)')
    else:
        plt.plot(AVERAGING_WINDOW2, etanorm2[day], marker=markers[0], color=colors[i], markersize=5, linestyle="dotted", linewidth=LINEWIDTH, label = r'dt = '+str(DT2)+' s')
        plt.plot(AVERAGING_WINDOW0, etanorm0[day], marker=markers[0], color=colors[i], markersize=5, linestyle="solid", linewidth=LINEWIDTH, label = r'dt = '+str(DT0)+' s')
        plt.plot(AVERAGING_WINDOW1, etanorm1[day], marker=markers[0], color=colors[i], markersize=5, linestyle="dashed", linewidth=LINEWIDTH, label = r'dt = '+str(DT1)+' s')
        plt.plot(AVERAGING_WINDOW2, etanorm_1350s[day], color=colors[i], linestyle="dotted", linewidth=LINEWIDTH, label = r'dt = '+str(DT2)+' s')
        plt.plot(AVERAGING_WINDOW0, etanorm_900s[day], color=colors[i], linestyle="solid", linewidth=LINEWIDTH, label = r'dt = '+str(DT0)+' s')
        plt.plot(AVERAGING_WINDOW1, etanorm_450s[day], color=colors[i], linestyle="dashed", linewidth=LINEWIDTH, label = r'dt = '+str(DT1)+' s')
    i += 1

plt.title('Error in $\eta$ at day 1', fontsize=18)
plt.legend(prop={'size':10}, loc="lower right")

plt.xlabel(r'Averaging Window $T$ (hour)', fontsize=18)
plt.ylabel(r'Normalised error', fontsize=18)
plt.ylim(1.e-5,1.e-2)
plt.yscale('log')

plt.savefig('lev5_dt_compare_eta.eps')
plt.show()
