echo "****************** Installing pytorch ******************"
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing lmdb ******************"
conda install -y -c conda-forge python-lmdb

echo ""
echo ""
echo "****************** Installing yaml ******************"
python -m pip install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
python -m pip install easydict

echo ""
echo ""
echo "****************** Installing cython ******************"
python -m pip install cython

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
python -m pip install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
python -m pip install pandas

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
python -m pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
python -m pip install jpeg4py

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
python -m pip install tb-nightly

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
python -m pip install tikzplotlib

echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
python -m pip install thop==0.0.31.post2005241907

echo ""
echo ""
echo "****************** Installing colorama ******************"
python -m pip install colorama

echo ""
echo ""
echo "****************** Installing scipy ******************"
python -m pip install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
python -m pip install visdom

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
python -m pip install tensorboardX

echo ""
echo ""
echo "****************** Downgrade setuptools ******************"
python -m pip install setuptools==59.5.0

echo ""
echo ""
echo "****************** Installing wandb ******************"
python -m pip install wandb

echo ""
echo ""
echo "****************** Installing timm ******************"
python -m pip install timm==0.6.13

echo ""
echo ""
echo "****************** Installation complete! ******************"