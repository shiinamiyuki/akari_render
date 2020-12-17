mkdir build-linux-cpu
cd build-linux-cpu
git clone https://github.com/microsoft/vcpkg --depth 1
cd vcpkg
./bootstrap-vcpkg.sh
cd ..
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja -DDCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
ninja