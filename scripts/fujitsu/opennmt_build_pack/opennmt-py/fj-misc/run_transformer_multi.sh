#!/bin/bash

suffix=$(date '+%Y%M%d-%s')
pushd "$(cd $(dirname $0); pwd)/.." 1>/dev/null
source fj-misc/$(arch).multi.conf

# INFO
python -c "import torch; print(torch.__version__)"       |  tee    $(arch)/transformer_console_$suffix.log
python -c "import torch; print(torch.__config__.show())" |  tee -a $(arch)/transformer_console_$suffix.log
$MPIRUN -np $NPROC --display-map \
	./fj-misc/run_transformer_multi_impl.sh          |& tee -a $(arch)/transformer_console_$suffix.log

cp $(arch)/transformer_transformer_train_text_1024.log \
   $(arch)/transformer_transformer_train_text_1024_$suffix.log

echo "Log file    : " $(arch)/$(ls -t $(arch) | grep transformer_console | head -1)
echo "API Profile : " $(arch)/transformer_transformer_train_text_1024_$suffix.log
