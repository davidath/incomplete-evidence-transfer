#! /bin/bash
make run test CONF=ini/20ng/real5/0.1/real5.ini CONF2=ini/20ng/real5/0.1/real5sae.ini RUN=0 PREF=real5_0.1_
make run test CONF=ini/20ng/real5/0.3/real5.ini CONF2=ini/20ng/real5/0.3/real5sae.ini RUN=0 PREF=real5_0.3_
make run test CONF=ini/20ng/real20/skip1/real20.ini CONF2=ini/20ng/real20/skip1/real20sae.ini RUN=0 PREF=real20_skip1_
make run test CONF=ini/20ng/real20/skip2/real20.ini CONF2=ini/20ng/real20/skip2/real20sae.ini RUN=0 PREF=real20_skip2_
./train.py ini/20ng/2real/skip2/2real.ini 0 ini/20ng/2real/skip2/2realsae1.ini ini/20ng/2real/skip2/2realsae2.ini && make test CONF=ini/20ng/2real/skip2/2real.ini RUN=0 PREF=2real_skip2_
