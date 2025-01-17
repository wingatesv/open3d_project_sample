# Point Cloud Preprocessing in C++

## Step 1: Compile and install Open3D

Follow the [Open3D compilation guide](http://www.open3d.org/docs/release/compilation.html),
compile and install Open3D in your preferred location. You can specify the
installation path with `CMAKE_INSTALL_PREFIX` and the number of parallel jobs
to speed up compilation.

On Ubuntu/macOS:

```bash
git clone --recursive https://github.com/intel-isl/Open3D.git
cd Open3D
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${HOME}/open3d_install ..
make install -j 12
cd ../..
```

On Windows:

```batch
git clone --recursive https://github.com/intel-isl/Open3D.git
cd Open3D
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=C:\open3d_install ..
cmake --build . --config Release --parallel 12 --target install
cd ..\..
```

Note: `-DBUILD_SHARED_LIBS=ON` is recommended if `-DBUILD_CUDA_MODULE=ON`.

## Step 2: Use Open3D in this project

On Ubuntu/macOS:

```bash
git clone https://github.com/intel-isl/open3d-cmake-find-package.git
cd open3d-cmake-find-package
mkdir build
cd build
cmake -DOpen3D_ROOT=${HOME}/open3d_install ..
make -j 12
./Process_pc --input-dir [INPUT-DIR] --output-dir [OUTPUT-DIR] --label-dir [LABEL-DIR] --output-label-dir [OUTPUT-LABEL-DIR]
```

On Windows:

```batch
git clone https://github.com/intel-isl/open3d-cmake-find-package.git
cd open3d-cmake-find-package
mkdir build
cmake -DOpen3D_ROOT=C:\open3d_install ..
cmake --build . --config Release --parallel 12
Release\Process_pc --input-dir [INPUT-DIR] --output-dir [OUTPUT-DIR] --label-dir [LABEL-DIR] --output-label-dir [OUTPUT-LABEL-DIR]
```
