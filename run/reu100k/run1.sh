#! /bin/bash

make run test CONF=ini/reu100k/real5/0.1/real5.ini CONF2=ini/reu100k/real5/0.1/real5sae.ini RUN=0 PREF=real5_0.1_
make run test CONF=ini/reu100k/real5/0.3/real5.ini CONF2=ini/reu100k/real5/0.3/real5sae.ini RUN=0 PREF=real5_0.3_
make run test CONF=ini/reu100k/real5/skip1/real5.ini CONF2=ini/reu100k/real5/skip1/real5sae.ini RUN=0 PREF=real5_skip1_
make run test CONF=ini/reu100k/real5/skip2/real5.ini CONF2=ini/reu100k/real5/skip2/real5sae.ini RUN=0 PREF=real5_skip2_
