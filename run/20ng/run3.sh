#! /bin/bash

make run test CONF=ini/20ng/real6/skip1/real6.ini CONF2=ini/20ng/real6/skip1/real6sae.ini RUN=0 PREF=real6_skip1_
make run test CONF=ini/20ng/real6/skip2/real6.ini CONF2=ini/20ng/real6/skip2/real6sae.ini RUN=0 PREF=real6_skip2_
./train.py ini/20ng/2real/skip1/2real.ini 0 ini/20ng/2real/skip1/2realsae1.ini ini/20ng/2real/skip1/2realsae2.ini && make test CONF=ini/20ng/2real/skip1/2real.ini RUN=0 PREF=2real_skip1_
make run test CONF=ini/20ng/real20/0.1/real20.ini CONF2=ini/20ng/real20/0.1/real20sae.ini RUN=0 PREF=real20_0.1_

