#! /bin/bash

make run test CONF=ini/cifar/real3/skip2/real3.ini CONF2=ini/cifar/real3/skip2/real3sae.ini RUN=0 PREF=real3_skip2_
make run test CONF=ini/cifar/real4/skip2/real4.ini CONF2=ini/cifar/real4/skip2/real4sae.ini RUN=0 PREF=real4_skip2_
make run test CONF=ini/cifar/real5/skip2/real5.ini CONF2=ini/cifar/real5/skip2/real5sae.ini RUN=0 PREF=real5_skip2_
make run test CONF=ini/cifar/real10/skip2/real10.ini CONF2=ini/cifar/real10/skip2/real10sae.ini RUN=0 PREF=real10_skip2_
./train.py ini/cifar/2real/skip2/2real.ini 0 ini/cifar/2real/skip2/2realsae1.ini ini/cifar/2real/skip2/2realsae2.ini &&  make test CONF=ini/cifar/2real/skip2/2real.ini RUN=0 PREF=2real_skip2_
./train.py ini/cifar/triple_a/skip2/a.ini 0 ini/cifar/triple_a/skip2/asae1.ini ini/cifar/triple_a/skip2/asae2.ini ini/cifar/triple_a/skip2/asae3.ini && make test CONF=ini/cifar/triple_a/skip2/a.ini RUN=0 PREF=triple_skip2_
