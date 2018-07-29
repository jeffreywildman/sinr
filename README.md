sinr
====

A software library and example applications for generating wireless signal strength maps using GPU acceleration.


Dependencies
------------

Tested on [Ubuntu 18.04.1 64-bit Desktop](https://www.ubuntu.com) using [NVIDIA CUDA Toolkit 9.2](https://developer.nvidia.com/cuda-zone) with a [GeForce GTX 660 Ti](https://www.geforce.com/hardware/desktop-gpus/geforce-gtx-660ti) graphics card.

A rough set of commands follow, but
[this](https://www.pugetsystems.com/labs/hpc/How-to-install-CUDA-9-2-on-Ubuntu-18-04-1184/) was pretty helpful in getting things set up.

* Install 396-series drivers:

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-drivers-396
```

* Install CUDA Toolkit 9.2:

```bash
sudo apt install freeglut3 freeglut3-dev libxi-dev libxmu-dev
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux -O cuda_9.2.148_396.37_linux.run
chmod 755 ./cuda_9.2.148_396.37_linux.run
./cuda_9.2.148_396.37_linux.run
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
