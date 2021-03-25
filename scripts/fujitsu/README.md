## Requirements

* FUJITSU Software Compiler Package is already installed.
* The login node has access to the external network.

### Verified environment

| **Hardware**      | Fujitsu FX1000 / FX700    |
| ------------ | ----------------------- |
| **OS**       | RedHad 8.1 / Centos 8.1 |
| **Compiler** |   FUJITSU Software Compiler Package     |


##  Preparation

1. Checkout from Repository.

```
 $ git clone https://github.com/fujitsu/pytorch.git
 $ cd pytorch
 $ git checkout -b fujitsu_v1.7.0_for_a64fx origin/fujitsu_v1.7.0_for_a64fx
```

2. Environment Setting

```
 $ cd scripts/fujitsu
```

```
     Modify the following environment variables in "env.src".
     ################################################
     ## Please change the following to suit your environment.
     ## PREFIX    : The directory where this file is located.
     ## TCSDS_PATH: TCS installation path
     ################################################
     export PREFIX=/home/users/ai/ai0003/pytorch/scripts/fujitsu
     export TCSDS_PATH=/opt/FJSVxtclanga/tcsds-1.2.26
```


```
      Modify each batch files to suit your environment.
     #!/bin/bash
     #PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
     #PJM -L elapse=04:00:00
     #PJM -L "node=1"
     #PJM -j
     #PJM -S
```

##  Build PyTorch
  Login Node[Estimated time:4h]

### Download
 Run checkout script
```
 $ ./checkout.sh
```

### Build binary files for A64FX.
 Run build-script on compute node.

```
 $ pjsub submit_build.sh
```

### Check the environment
 Run the sample programs

```
 $ pjsub submit_train.sh
 $ pjsub submit_train_multi.sh
 $ pjsub submit_val.sh
 $ pjsub submit_val_multi.sh
```
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

## Build for OpenNMT

```
 $ cd opennmt_build_pack
```

### Environment Setting
 Modify each batch files to suit your environment.

* submit_build.sh
* opennmt-py/submit_prepare.sh
* opennmt-py/submit_opennmt.sh

### Download

```
 $ ./dataset.sh 
 $ ./checkout.sh 
```

### Build [Estimated time:1.5h]

```
 $ pjsub submit_build.sh
```

### Preprocessing for dataset [Estimated time:1h]

```
 $ cd opennmt-py
 $ pjsub submit_prepare.sh
```

### Check the environment

```
 $ pjsub submit_opennmt.sh
```

```
 $ cd ..
```

## Build for BERT

### Requirements
* python installed >= 3.x (login node)

```
 $ cd bert_build_pack/
```

### Environment Setting
 Modify each batch files to suit your environment.

* submit_build.sh
* tranformers/submit_bert_mrpc.sh
* tranformers/submit_bert_lm.sh

### Download

```
 $ ./checkout.sh

 $ cd tranformers
 $ ./prepare.sh
 $ cd ..
```

### Build [Estimated time:1.5h]

```
 $ pjsub submit_build.sh
```

### Check the environment

```
 $ cd transformers
 $ pjsub submit_bert_lm.sh    # BERT Pre-Training
 $ pjsub submit_bert_mrpc.sh  # BERT FIne-Tuning(MRPC)
```

```
 $ cd ..
```

## Build for Mask RCNN

```
 $ cd mask_r_cnn_build_pack/
```

### Environment Setting
 Modify each batch files to suit your environment.

* submit_build.sh
* detectron2/submit_mask_r_cnn.sh

### Download

```
 $ ./checkout.sh
 $ ./dataset.sh
```

### Build [Estimated time:1.5h]

```
 $ pjsub submit_build.sh
```

### Check the environment

```
 $ cd detectron2
 $ pjsub submit_mask_r_cnn.sh
```

```
 $ cd ..
```


## Execute gemm with half-gemm automatically.

### Enable function with api in python script (default:Disable)

```
 torch.set_enabled_auto_half_gemm(True)
```


## Copyright

Copyright RIKEN, Japan 2021
Copyright FUJITSU LIMITED 2021
