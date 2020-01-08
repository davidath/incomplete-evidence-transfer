#! /bin/bash

./train.py ini/reu100k/2real/0.1/2real.ini 0 ini/reu100k/2real/0.1/2realsae1.ini ini/reu100k/2real/0.1/2realsae2.ini && make test CONF=ini/reu100k/2real/0.1/2real.ini RUN=0 PREF=2real_0.1_
./train.py ini/reu100k/2real/0.3/2real.ini 0 ini/reu100k/2real/0.3/2realsae1.ini ini/reu100k/2real/0.3/2realsae2.ini && make test CONF=ini/reu100k/2real/0.3/2real.ini RUN=0 PREF=2real_0.3_
./train.py ini/reu100k/2real/skip1/2real.ini 0 ini/reu100k/2real/skip1/2realsae1.ini ini/reu100k/2real/skip1/2realsae2.ini && make test CONF=ini/reu100k/2real/skip1/2real.ini RUN=0 PREF=2real_skip1_
./train.py ini/reu100k/2real/skip2/2real.ini 0 ini/reu100k/2real/skip2/2realsae1.ini ini/reu100k/2real/skip2/2realsae2.ini && make test CONF=ini/reu100k/2real/skip2/2real.ini RUN=0 PREF=2real_skip2_

