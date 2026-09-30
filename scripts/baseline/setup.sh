#!/bin/bash
# Clone the baselines into ./baseline at the commits we used and add our L=336 scripts.
# usage (from the repo root): bash scripts/baseline/setup.sh
set -e

REPO=$(pwd)
SRC=$REPO/scripts/baseline
mkdir -p baseline

clone() {  # clone <name> <url> <commit>
  if [ ! -d baseline/$1 ]; then
    git clone -q $2 baseline/$1
  fi
  git -C baseline/$1 checkout -q $3
}

link_data() {  # link_data <dir>: flat csv files plus the ETT-small/weather/electricity/traffic layout
  mkdir -p $1/ETT-small $1/weather $1/electricity $1/traffic
  for f in ETTh1 ETTh2 ETTm1 ETTm2; do
    ln -sfn $REPO/dataset/$f.csv $1/$f.csv
    ln -sfn $REPO/dataset/$f.csv $1/ETT-small/$f.csv
  done
  for f in weather electricity traffic; do
    ln -sfn $REPO/dataset/$f.csv $1/$f.csv
    ln -sfn $REPO/dataset/$f.csv $1/$f/$f.csv
  done
}

clone TimeKAN https://github.com/huangst21/TimeKAN.git 3a7c366
clone Time-TK https://github.com/Cola-Fsm/Time-TK.git ee99266
clone Amplifier https://github.com/aikunyi/Amplifier.git 6cc0893
clone TimeMixer https://github.com/kwuking/TimeMixer.git e246105
clone PatchTST https://github.com/yuqinie98/PatchTST.git 204c21e
clone Time-Series-Library https://github.com/thuml/Time-Series-Library.git 4e938a1

for d in TimeKAN Time-TK Amplifier TimeMixer Time-Series-Library; do
  link_data baseline/$d/dataset
done
link_data baseline/PatchTST/PatchTST_supervised/dataset

# numpy >= 2.0
sed -i 's/np\.Inf/np.inf/' baseline/PatchTST/PatchTST_supervised/utils/tools.py

mkdir -p baseline/TimeKAN/scripts/l_336 baseline/Time-TK/scripts/l336 baseline/Amplifier/scripts/l_336 \
         baseline/TimeMixer/scripts/l_336 baseline/PatchTST/PatchTST_supervised/scripts/l336 \
         baseline/Time-Series-Library/scripts/long_term_forecast/l_336
cp -r $SRC/TimeKAN/* baseline/TimeKAN/scripts/l_336/
cp $SRC/TimeTK/*.sh baseline/Time-TK/scripts/l336/
cp $SRC/Amplifier/*.sh baseline/Amplifier/scripts/l_336/
cp $SRC/TimeMixer/*.sh baseline/TimeMixer/scripts/l_336/
cp $SRC/PatchTST/*.sh baseline/PatchTST/PatchTST_supervised/scripts/l336/
cp $SRC/TimesNet/*.sh baseline/Time-Series-Library/scripts/long_term_forecast/l_336/
