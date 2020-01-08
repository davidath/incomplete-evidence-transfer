#! /bin/bash
make run test CONF=ini/mnist/real4/0.1/real4.ini CONF2=ini/mnist/real4/0.1/real4sae.ini RUN=0 PREF=real4_0.1_
make run test CONF=ini/mnist/real4/0.3/real4.ini CONF2=ini/mnist/real4/0.3/real4sae.ini RUN=0 PREF=real4_0.3_
./train.py ini/mnist/2real/0.3/2real.ini 0 ini/mnist/2real/0.3/2realsae1.ini ini/mnist/2real/0.3/2realsae2.ini && make test CONF=ini/mnist/2real/0.3/2real.ini RUN=0 PREF=2real_0.3_
make run test CONF=ini/mnist/real10/0.3/real10.ini CONF2=ini/mnist/real10/0.3/real10sae.ini RUN=0 PREF=real10_0.3_

