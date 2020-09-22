cp ./resources/default.conf akari.conf
mkdir build-linux-cpu
cd build-linux-cpu
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja 
ninja
./akari-test