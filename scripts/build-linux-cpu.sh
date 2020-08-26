cp ./resources/config.default.py akari.conf
mkdir build-linux-cpu
cd build-linux-cpu
cmake .. -DCMAKE_BUILD_TYPE=Release -DAKR_ENABLE_EMBREE=ON -GNinja 
ninja
./akari-test