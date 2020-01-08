#! /bin/bash

make run test CONF=ini/cifar/real3/0.3/real3.ini CONF2=ini/cifar/real3/0.3/real3sae.ini RUN=0 PREF=real3_0.3_
make run test CONF=ini/cifar/real4/0.3/real4.ini CONF2=ini/cifar/real4/0.3/real4sae.ini RUN=0 PREF=real4_0.3_
make run test CONF=ini/cifar/real5/0.3/real5.ini CONF2=ini/cifar/real5/0.3/real5sae.ini RUN=0 PREF=real5_0.3_
make run test CONF=ini/cifar/real10/0.3/real10.ini CONF2=ini/cifar/real10/0.3/real10sae.ini RUN=0 PREF=real10_0.3_
./train.py ini/cifar/2real/0.3/2real.ini 0 ini/cifar/2real/0.3/2realsae1.ini ini/cifar/2real/0.3/2realsae2.ini && make test CONF=ini/cifar/2real/0.3/2real.ini RUN=0 PREF=2real_0.3_
./train.py ini/cifar/triple_a/0.3/a.ini 0 ini/cifar/triple_a/0.3/asae1.ini ini/cifar/triple_a/0.3/asae2.ini ini/cifar/triple_a/0.3/asae3.ini && make test CONF=ini/cifar/triple_a/0.3/a.ini RUN=0 PREF=triple_0.3_
