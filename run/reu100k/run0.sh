#! /bin/bash

make run test CONF=ini/reu100k/real4/0.1/real4.ini CONF2=ini/reu100k/real4/0.1/real4sae.ini RUN=0 PREF=real4_0.1_
make run test CONF=ini/reu100k/real4/0.3/real4.ini CONF2=ini/reu100k/real4/0.3/real4sae.ini RUN=0 PREF=real4_0.3_
make run test CONF=ini/reu100k/real4/skip1/real4.ini CONF2=ini/reu100k/real4/skip1/real4sae.ini RUN=0 PREF=real4_skip1_
make run test CONF=ini/reu100k/real4/skip2/real4.ini CONF2=ini/reu100k/real4/skip2/real4sae.ini RUN=0 PREF=real4_skip2_
