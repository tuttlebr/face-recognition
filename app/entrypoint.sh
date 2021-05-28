#!/bin/bash
set -e

export PS1="\[\e[31m\]face-recognition\[\e[m\] \[\e[33m\]\w\[\e[m\] > "
export TERM=xterm-256color
alias grep="grep --color=auto"
alias ls="ls --color=auto"

echo -e "\e[1;31m"
cat<<TF
                    ___
       ___....-----'---'-----....___
 =========================================
        ___'---..._______...---'___
       (___)      _|_|_|_      (___)
         \\____.-'_.---._'-.____//
           cccc'.__'---'__.'cccc
                   ccccc
TF
echo -e "\e[0;33m"

echo CPU: $(cat /proc/cpuinfo |grep "model name" | sed 's/^.*: //' | sort -u)

nvidia-smi -L

if [[ "$(find -L /usr -name libcuda.so.1 2>/dev/null | grep -v "compat") " == " " || "$(ls /dev/nvidiactl 2>/dev/null) " == " " ]]; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
  echo "   Use 'nvidia-docker run' to start this container; see"
  echo "   https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker ."
else
  ( /usr/local/bin/checkSMVER.sh )
  DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
  if [[ ! "$DRIVER_VERSION" =~ ^[0-9]*.[0-9]*(.[0-9]*)?$ ]]; then
    echo "Failed to detect NVIDIA driver version."
  elif [[ "${DRIVER_VERSION%%.*}" -lt "${CUDA_DRIVER_VERSION%%.*}" ]]; then
    if [[ "${_CUDA_COMPAT_STATUS}" == "CUDA Driver OK" ]]; then
      echo
      echo "NOTE: Legacy NVIDIA Driver detected.  Compatibility mode ENABLED."
    else
      echo
      echo "ERROR: This container was built for NVIDIA Driver Release ${CUDA_DRIVER_VERSION%.*} or later, but"
      echo "       version ${DRIVER_VERSION} was detected and compatibility mode is UNAVAILABLE."
      echo
      echo "       [[${_CUDA_COMPAT_STATUS}]]"
      sleep 2
    fi
  fi
fi

if ! cat /proc/cpuinfo | grep flags | sort -u | grep avx >& /dev/null; then
  echo
  echo "ERROR: This container was built for CPUs supporting at least the AVX instruction set, but"
  echo "       the CPU detected was $(cat /proc/cpuinfo |grep "model name" | sed 's/^.*: //' | sort -u), which does not report"
  echo "       support for AVX.  An Illegal Instrution exception at runtime is likely to result."
  echo "       See https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX ."
  sleep 2
fi

DETECTED_MOFED=$(cat /sys/module/mlx5_core/version 2>/dev/null || true)
case "${DETECTED_MOFED}" in
  "${MOFED_VERSION}")
    echo
    echo "Detected MOFED ${DETECTED_MOFED}."
    ;;
  "")
    echo
    echo "NOTE: MOFED driver for multi-node communication was not detected."
    echo "      Multi-node communication performance may be reduced."
    ;;
  *)
    if /opt/mellanox/change_mofed_version.sh "${DETECTED_MOFED}" >& /dev/null; then
      echo
      echo "NOTE: Detected MOFED driver ${DETECTED_MOFED}; version automatically updated."
    else
      echo
      echo "ERROR: Detected MOFED driver ${DETECTED_MOFED}, but this container has version ${MOFED_VERSION}."
      echo "       Unable to automatically upgrade this container."
      echo "       Use of RDMA for multi-node communication will be unreliable."
      sleep 2
    fi
    ;;
esac

DETECTED_NVPEERMEM=$(cat /sys/module/nv_peer_mem/version 2>/dev/null || true)
if [[ "${DETECTED_MOFED} " != " " && "${DETECTED_NVPEERMEM} " == " " ]]; then
  echo
  echo "NOTE: MOFED driver was detected, but nv_peer_mem driver was not detected."
  echo "      Multi-node communication performance may be reduced."
fi

if [[ "$(df -k /dev/shm |grep ^shm |awk '{print $2}') " == "65536 " ]]; then
  echo
  echo "NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be"
  echo "   insufficient for TensorFlow.  NVIDIA recommends the use of the following flags:"
  echo "   nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 ..."
fi

echo

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi      
# Turn off colors
echo -e "\e[m"