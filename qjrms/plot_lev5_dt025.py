import numpy as np
import matplotlib.pyplot as plt
import data_lev5_dt025
from data_lev5_dt025 import *

# shared parameters
PPP = 4
DTS = 22.5
DAY_INT = 0.25
LOGLOG = True

# set parameters
DAY_MAX0 = 1
DAYS_TO_PLOT0 = [1]
AVERAGING_WINDOW_MAX0 = 10

DAY_MAX1 = 1
DAYS_TO_PLOT1 = [1]
AVERAGING_WINDOW_MAX1 = 2

DAY_MAX2 = 5
DAYS_TO_PLOT2 = [5,4,3,2]
AVERAGING_WINDOW_MAX2 = 1

# set days
DAYS0 = []
imax0 = int(DAY_MAX0/DAY_INT)
i = 1
while i <= imax0:
    DAYS0.append(i*DAY_INT)
    i += 1

DAYS1 = []
imax1 = int(DAY_MAX1/DAY_INT)
i = 1
while i <= imax1:
    DAYS1.append(i*DAY_INT)
    i += 1

DAYS2 = []
imax2 = int(DAY_MAX2/DAY_INT)
i = 1
while i <= imax2:
    DAYS2.append(i*DAY_INT)
    i += 1

# set averaging window
AVERAGING_WINDOW_ALL = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
AVERAGING_WINDOW0 = []
for k in AVERAGING_WINDOW_ALL:
    if k <= AVERAGING_WINDOW_MAX0:
        AVERAGING_WINDOW0.append(k)

AVERAGING_WINDOW1 = []
for k in AVERAGING_WINDOW_ALL:
    if k <= AVERAGING_WINDOW_MAX1:
        AVERAGING_WINDOW1.append(k)

AVERAGING_WINDOW2 = []
for k in AVERAGING_WINDOW_ALL:
    if k <= AVERAGING_WINDOW_MAX2:
        AVERAGING_WINDOW2.append(k)

# input data
etadict0 = {}
udict0 = {}
input_lev5_dt025(etadict0, udict0, ppp=PPP, dts=DTS)

etadict1 = {}
udict1 = {}
input_lev5_dt025(etadict1, udict1, ppp=PPP, dts=DTS)

etadict2 = {}
udict2 = {}
input_lev5_dt025(etadict2, udict2, ppp=PPP, dts=DTS)

# sort data by day
etanorm0 = {}
unorm0 = {}
i = 0
for day in DAYS0:
    etanorm0[day] = []
    unorm0[day] = []
    for key in AVERAGING_WINDOW0:
        etanorm0[day].append(etadict0[key][i])
        unorm0[day].append(udict0[key][i])
    i += 1

etanorm1 = {}
unorm1 = {}
i = 0
for day in DAYS1:
    etanorm1[day] = []
    unorm1[day] = []
    for key in AVERAGING_WINDOW1:
        etanorm1[day].append(etadict1[key][i])
        unorm1[day].append(udict1[key][i])
    i += 1

etanorm2 = {}
unorm2 = {}
i = 0
for day in DAYS2:
    etanorm2[day] = []
    unorm2[day] = []
    for key in AVERAGING_WINDOW2:
        etanorm2[day].append(etadict2[key][i])
        unorm2[day].append(udict2[key][i])
    i += 1

##################################
#     PLOTTONG ENTIRE FIGURE     #
##################################
markers = ["+", "s", "o", "v", "^", "<", ">", "1", "2", "3"]
colors = ["black", "red", "blue", "green", "magenta", "darkorange", "brown", "yellowgreen", "purple"]

plt.figure(num=1,figsize=(8,6))

i = 0
for day in DAYS_TO_PLOT0:
    if LOGLOG:
        plt.loglog(AVERAGING_WINDOW0, unorm0[day], color=colors[i], linewidth=1., marker=markers[4], markersize=3, markeredgecolor=colors[i], label = r'$u$')
        plt.loglog(AVERAGING_WINDOW0, etanorm0[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, markeredgecolor=colors[i], label = r'$\eta$')
    else:
        plt.plot(AVERAGING_WINDOW0, unorm0[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, label = r'day '+str(day))
        plt.plot(AVERAGING_WINDOW0, etanorm0[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, label = r'day '+str(day))
    i += 1

plt.title('Error at day 1', fontsize=18)
plt.legend(prop={'size':13}, loc="lower right")

plt.xlabel(r'Averaging Window $T$ (hour)', fontsize=18)
plt.ylabel(r'Normalised error', fontsize=18)
plt.ylim(1.e-5,1.e-1)

plt.savefig('lev5_dt025.eps')
plt.show()

# plt.figure(num=2,figsize=(8,6))

# i = 0
# for day in DAYS_TO_PLOT0:
#     if LOGLOG:
#         plt.loglog(AVERAGING_WINDOW0, etanorm0[day], color=colors[i], linewidth=1., marker=markers[4], markersize=3, markeredgecolor=colors[i], label = r'day = '+str(day))
#     else:
#         plt.plot(AVERAGING_WINDOW0, etanorm0[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, markeredgecolor=colors[i], label = r'day = '+str(day))
#     i += 1

# plt.title('Error in $\eta$', fontsize=18)
# plt.legend(prop={'size':12}, loc="lower right")

# plt.xlabel(r'Averaging Window', fontsize=18)
# plt.ylabel(r'Normalised error', fontsize=18)
# plt.ylim(1.e-5,1.e-1)

# plt.savefig('lev5_dt025_eta_day025.eps')
# plt.show()

plt.figure(num=3,figsize=(6,6))

i = 4
for day in DAYS_TO_PLOT2:
    if LOGLOG:
        plt.loglog(AVERAGING_WINDOW2, unorm2[day], color=colors[i], linewidth=1., marker=markers[4], markersize=3, markeredgecolor=colors[i], label = r'day '+str(day))
    else:
        plt.plot(AVERAGING_WINDOW2, unorm2[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, label = r'day '+str(day))
    i -= 1

# for day in DAYS_TO_PLOT1:
#     if LOGLOG:
#         plt.loglog(AVERAGING_WINDOW1, unorm1[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, label = r'day = '+str(day))
#     else:
#         plt.plot(AVERAGING_WINDOW1, unorm1[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, label = r'day = '+str(day))
#     i -= 1

for day in DAYS_TO_PLOT0:
    if LOGLOG:
        plt.loglog(AVERAGING_WINDOW0, unorm0[day], color=colors[i], linewidth=1., marker=markers[4], markersize=3, markeredgecolor=colors[i], label = r'day '+str(day))
    else:
        plt.plot(AVERAGING_WINDOW0, unorm0[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, label = r'day '+str(day))

plt.title('Error in $u$', fontsize=18)
plt.legend(prop={'size':13}, loc="lower left")

plt.xlabel(r'Averaging Window $T$ (hour)', fontsize=18)
plt.ylabel(r'Normalised error', fontsize=18)
plt.xlim(1.e-1,1.e0)
plt.ylim(1.e-5,1.e-1)

plt.savefig('lev5_dt025_u.eps')
plt.show()

plt.figure(num=4,figsize=(6,6))

i = 4
for day in DAYS_TO_PLOT2:
    if LOGLOG:
        plt.loglog(AVERAGING_WINDOW2, etanorm2[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, markeredgecolor=colors[i], label = r'day '+str(day))
    else:
        plt.plot(AVERAGING_WINDOW2, etanorm2[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5,  label = r'day '+str(day))
    i -= 1

# for day in DAYS_TO_PLOT1:
#     if LOGLOG:
#         plt.loglog(AVERAGING_WINDOW1, etanorm1[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5,  label = r'day = '+str(day))
#     else:
#         plt.plot(AVERAGING_WINDOW1, etanorm1[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5,  label = r'day = '+str(day))
#     i -= 1

for day in DAYS_TO_PLOT0:
    if LOGLOG:
        plt.loglog(AVERAGING_WINDOW0, etanorm0[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5, markeredgecolor=colors[i], label = r'day '+str(day))
    else:
        plt.plot(AVERAGING_WINDOW0, etanorm0[day], color=colors[i], linewidth=1., marker=markers[0], markersize=5,  label = r'day '+str(day))

plt.title('Error in $\eta$', fontsize=18)
plt.legend(prop={'size':13}, loc="lower left")

plt.xlabel(r'Averaging Window $T$ (hour)', fontsize=18)
plt.ylabel(r'Normalised error', fontsize=18)
plt.xlim(1.e-1,1.e0)
plt.ylim(1.e-5,1.e-1)

plt.savefig('lev5_dt025_eta.eps')
plt.show()
