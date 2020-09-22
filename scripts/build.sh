cp ./resources/default.conf akari.conf
mkdir build-general
cd build-general
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release