### for samv2
cd AutoMask
cd checkpoints && \
sh ./download_ckpts.sh && \
cd ..

### for mogesam
cd MogeSAM

cd third_party/pointops2
LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
cd ../..

cd third_party/megasam/base
LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
cd ../../..

mkdir checkpoints 
mkdir third_party/megasam/Depth-Anything/checkpoints/
wget https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth ./checkpoints/
curl -L -o third_party/megasam/Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
gdown --folder --fuzzy 'https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT' -O third_party/megasam/cvd_opt/ --remaining-ok --continue 
mv third_party/megasam/cvd_opt/models/raft-things.pth third_party/megasam/cvd_opt
cd ..

### for GVHMR
cd HMR
pip install -e .

