#! /bin/bash

make run test CONF=ini/mnist/real4/skip1/real4.ini CONF2=ini/mnist/real4/skip1/real4sae.ini RUN=0 PREF=real4_skip1_
make run test CONF=ini/mnist/real4/skip2/real4.ini CONF2=ini/mnist/real4/skip2/real4sae.ini RUN=0 PREF=real4_skip2_
./train.py ini/mnist/2real/skip1/2real.ini 0 ini/mnist/2real/skip1/2realsae1.ini ini/mnist/2real/skip1/2realsae2.ini && make test CONF=ini/mnist/2real/skip1/2real.ini RUN=0 PREF=2real_skip1_
make run test CONF=ini/mnist/real10/0.1/real10.ini CONF2=ini/mnist/real10/0.1/real10sae.ini RUN=0 PREF=real10_0.1_

