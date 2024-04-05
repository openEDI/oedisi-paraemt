% check 240-bus current balance at bus 6335
clc;clear;

v6335 = 1.065074*exp(1i*(101.2052)/180*pi);
v6305 = 1.06*exp(1i*(100.4084)/180*pi);
rl = 0+0.0005j;
ii = (v6335-v6305)/rl

S6335 = v6335*conj(ii)
S6305 = -v6305*conj(ii)