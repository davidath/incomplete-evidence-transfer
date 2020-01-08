#! /bin/bash

make run test CONF=ini/mnist/real3/skip1/real3.ini CONF2=ini/mnist/real3/skip1/real3sae.ini RUN=0 PREF=real3_skip1_
make run test CONF=ini/mnist/real3/skip2/real3.ini CONF2=ini/mnist/real3/skip2/real3sae.ini RUN=0 PREF=real3_skip2_
./train.py ini/mnist/2real/0.1/2real.ini 0 ini/mnist/2real/0.1/2realsae1.ini ini/mnist/2real/0.1/2realsae2.ini && make test CONF=ini/mnist/2real/0.1/2real.ini RUN=0 PREF=2real_0.1_
