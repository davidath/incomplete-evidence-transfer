#! /bin/bash
make run test CONF=ini/20ng/real6/0.1/real6.ini CONF2=ini/20ng/real6/0.1/real6sae.ini RUN=0 PREF=real6_0.1_
make run test CONF=ini/20ng/real6/0.3/real6.ini CONF2=ini/20ng/real6/0.3/real6sae.ini RUN=0 PREF=real6_0.3_
./train.py ini/20ng/2real/0.3/2real.ini 0 ini/20ng/2real/0.3/2realsae1.ini ini/20ng/2real/0.3/2realsae2.ini && make test CONF=ini/20ng/2real/0.3/2real.ini RUN=0 PREF=2real_0.3_
make run test CONF=ini/20ng/real20/0.3/real20.ini CONF2=ini/20ng/real20/0.3/real20sae.ini RUN=0 PREF=real20_0.3_

