cd ..
cp resoures/config.default.py akari.conf
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -r -j `nproc`