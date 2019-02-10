sinr
====

A software library and example applications for generating wireless signal strength maps using GPU acceleration.


Dependencies
------------

Tested on [Ubuntu 18.04.2 64-bit Desktop](https://www.ubuntu.com) using [NVIDIA CUDA Toolkit 10.0](https://developer.nvidia.com/cuda-zone) with a [GeForce GTX 660 Ti](https://www.geforce.com/hardware/desktop-gpus/geforce-gtx-660ti) graphics card.

A rough set of commands follow from [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork):

```bash
sudo apt install freeglut3 freeglut3-dev libxi-dev libxmu-dev
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt update
sudo apt install nvidia-drivers-410
sudo apt install cuda
```

* Set environment variables:

```bash
export PATH=/usr/local/cuda/bin:${PATH}
export PKG_CONFIG_PATH=/usr/local/cuda/pkgconfig:${PKG_CONFIG_PATH}
```

* Install build dependencies:

```bash
sudo apt install build-essential autotools-dev libtool autoconf
```


Build
-----

```bash
git clone git@github.com:jeffreywildman/sinr.git
cd ./sinr
./autogen.sh
make
```


Run
---

```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./src/sinrmap-demo
eog ./sinrmap-demo-maxsinr.bmp
```
