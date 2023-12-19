#!/ bin / bash

ROOT=/srv01/agrp/nilotpal/projects/tracking/mmCuda #${PWD}
BIN=${ROOT}/bin
#if [[ !-d ${BIN} ]] 
#then 
#    mkdir ${BIN}
#fi

nvcc -std=c++11 -o ${BIN}/memoryscheduler.o -c ${PWD}/src/memoryscheduler.cu -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT
nvcc -std=c++11 -o ${BIN}/datatypes.o -c ${PWD}/src/datatypes.cu -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT
nvcc -std=c++11 -o ${BIN}/triplet_finder.o -c ${PWD}/src/triplet_finder.cu -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT
nvcc -std=c++11 -o ${BIN}/main.o -c ${PWD}/src/main.cu -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT
nvcc -o ${BIN}/main ${BIN}/triplet_finder.o ${BIN}/datatypes.o ${BIN}/memoryscheduler.o ${BIN}/main.o -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT