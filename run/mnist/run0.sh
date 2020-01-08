#! /bin/bash
make run test CONF=ini/mnist/real3/0.1/real3.ini CONF2=ini/mnist/real3/0.1/real3sae.ini RUN=0 PREF=real3_0.1_
make run test CONF=ini/mnist/real3/0.3/real3.ini CONF2=ini/mnist/real3/0.3/real3sae.ini RUN=0 PREF=real3_0.3_
make run test CONF=ini/mnist/real10/skip1/real10.ini CONF2=ini/mnist/real10/skip1/real10sae.ini RUN=0 PREF=real10_skip1_
make run test CONF=ini/mnist/real10/skip2/real10.ini CONF2=ini/mnist/real10/skip2/real10sae.ini RUN=0 PREF=real10_skip2_
./train.py ini/mnist/2real/skip2/2real.ini 0 ini/mnist/2real/skip2/2realsae1.ini ini/mnist/2real/skip2/2realsae2.ini && make test CONF=ini/mnist/2real/skip2/2real.ini RUN=0 PREF=2real_skip2_
