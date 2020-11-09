*****************************************************************************************************************
Documentation :
  PyTorch for A64FX (2020/11/09)
*****************************************************************************************************************

0. Target node
  - FX1000

1. Requirements
　・FUJITSU Software Compiler Package is already installed.
  ・The login node has access to the external network.

2. Preparation
  2-1. Checkout from Repository.
          # git clone https://github.com/fujitsu/pytorch.git
          # cd pytorch
          # git checkout -b fujitsu_v1.5.0_for_a64fx origin/fujitsu_v1.5.0_for_a64fx

  2-2. Environment Setting

          # cd scripts/fujitsu

      Modify the following environment variables in "env.src".
     ################################################
     ## Please change the following to suit your environment.
     ## PREFIX    : The directory where this file is located.
     ## TCSDS_PATH: TCS installation path
     ################################################
     export PREFIX=/home/users/ai/ai0003/pytorch/scripts/fujitsu
     export TCSDS_PATH=/opt/FJSVxtclanga/tcsds-1.2.26

      Modify each batch files to suit your environment.
     #!/bin/bash
     #PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
     #PJM -L elapse=04:00:00
     #PJM -L "node=1"
     #PJM -j
     #PJM -S


3. Build PyTorch
  Login Node[Estimated time:4h]

  3-1. Download 
    Run checkout script
        # ./checkout.sh

  3-2. Build binary files for A64FX.
    Run build-script on compute node.
        # pjsub submit_build.sh
          
  3-3. Check the environment
    Run the sample programs
        # pjsub submit_train.sh
        # pjsub submit_train_multi.sh
        # pjsub submit_val.sh
        # pjsub submit_val_multi.sh

        Example of output(submit_train_multi.sh.xxx.out)
        ～
        Model: resnet50
        Batch size: 75
        Number of CPUs: 4
        Running warmup...
        Running benchmark...
        Iter #0: 26.1 img/sec per CPU
        Iter #1: 26.1 img/sec per CPU
        Iter #2: 26.1 img/sec per CPU
        Iter #3: 26.1 img/sec per CPU
        Iter #4: 26.1 img/sec per CPU
        Img/sec per CPU: 26.1 +-0.0
        Total img/sec on 4 CPU(s): 104.5 +-0.1
        ～
4. Acknowledgments
  The torch/csrc/autograd/engine.cpp has been modified by Dr. Kuroda and Mr. Ando of RIKEN to use oneDNN efficiently in aarch64.
