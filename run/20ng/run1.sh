#! /bin/bash

make run test CONF=ini/20ng/real5/skip1/real5.ini CONF2=ini/20ng/real5/skip1/real5sae.ini RUN=0 PREF=real5_skip1_
make run test CONF=ini/20ng/real5/skip2/real5.ini CONF2=ini/20ng/real5/skip2/real5sae.ini RUN=0 PREF=real5_skip2_
./train.py ini/20ng/2real/0.1/2real.ini 0 ini/20ng/2real/0.1/2realsae1.ini ini/20ng/2real/0.1/2realsae2.ini && make test CONF=ini/20ng/2real/0.1/2real.ini RUN=0 PREF=2real_0.1_
