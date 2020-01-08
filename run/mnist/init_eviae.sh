#! /bin/bash
make run_cmd CONF=ini/mnist/real3/0.1/real3.ini CONF2=ini/mnist/real3/0.1/real3sae.ini RUN=0 PREF=real3_0.1_
make run_cmd CONF=ini/mnist/real3/0.3/real3.ini CONF2=ini/mnist/real3/0.3/real3sae.ini RUN=0 PREF=real3_0.3_
make run_cmd CONF=ini/mnist/real3/skip1/real3.ini CONF2=ini/mnist/real3/skip1/real3sae.ini RUN=0 PREF=real3_skip1_
make run_cmd CONF=ini/mnist/real3/skip2/real3.ini CONF2=ini/mnist/real3/skip2/real3sae.ini RUN=0 PREF=real3_skip2_

make run_cmd CONF=ini/mnist/real4/0.1/real4.ini CONF2=ini/mnist/real4/0.1/real4sae.ini RUN=0 PREF=real4_0.1_
make run_cmd CONF=ini/mnist/real4/0.3/real4.ini CONF2=ini/mnist/real4/0.3/real4sae.ini RUN=0 PREF=real4_0.3_
make run_cmd CONF=ini/mnist/real4/skip1/real4.ini CONF2=ini/mnist/real4/skip1/real4sae.ini RUN=0 PREF=real4_skip1_
make run_cmd CONF=ini/mnist/real4/skip2/real4.ini CONF2=ini/mnist/real4/skip2/real4sae.ini RUN=0 PREF=real4_skip2_

make run_cmd CONF=ini/mnist/real10/0.1/real10.ini CONF2=ini/mnist/real10/0.1/real10sae.ini RUN=0 PREF=real10_0.1_
make run_cmd CONF=ini/mnist/real10/0.3/real10.ini CONF2=ini/mnist/real10/0.3/real10sae.ini RUN=0 PREF=real10_0.3_
make run_cmd CONF=ini/mnist/real10/skip1/real10.ini CONF2=ini/mnist/real10/skip1/real10sae.ini RUN=0 PREF=real10_skip1_
make run_cmd CONF=ini/mnist/real10/skip2/real10.ini CONF2=ini/mnist/real10/skip2/real10sae.ini RUN=0 PREF=real10_skip2_
