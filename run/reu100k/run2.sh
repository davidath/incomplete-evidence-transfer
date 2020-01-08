#! /bin/bash
make run test CONF=ini/reu100k/real10/0.1/real10.ini CONF2=ini/reu100k/real10/0.1/real10sae.ini RUN=0 PREF=real10_0.1_
make run test CONF=ini/reu100k/real10/0.3/real10.ini CONF2=ini/reu100k/real10/0.3/real10sae.ini RUN=0 PREF=real10_0.3_
make run test CONF=ini/reu100k/real10/skip1/real10.ini CONF2=ini/reu100k/real10/skip1/real10sae.ini RUN=0 PREF=real10_skip1_
make run test CONF=ini/reu100k/real10/skip2/real10.ini CONF2=ini/reu100k/real10/skip2/real10sae.ini RUN=0 PREF=real10_skip2_
