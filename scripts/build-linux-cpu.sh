mkdir build-linux-cpu
cd build-linux-cpu
git clone https://github.com/microsoft/vcpkg --depth 1
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg update
# prevent travis from terminating our job
./vcpkg install embree3:x64-linux
./vcpkg install openimageio:x64-linux
./vcpkg install assimp:x64-linux
cd ..
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake\
     -DCMAKE_MAKE_PROGRAM=make -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX
ninja