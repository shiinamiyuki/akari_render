cd useless
python3 ./main.py install embree assimp cereal glm cxxopts pybind11 openexr spdlog stb
cd .. 
mkdir build-linux-cpu
cd build-linux-cpu
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
ninja