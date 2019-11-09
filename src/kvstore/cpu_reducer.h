
#ifndef BYTEPS_CPU_REDUCER_H
#define BYTEPS_CPU_REDUCER_H

#if __AVX__ && __F16C__
#include <cpuid.h>
#include <immintrin.h>
#endif

#include <memory>
#include <stdint.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace kvstore {

enum DataType {
  BYTEPS_FLOAT32 = 0,
  BYTEPS_FLOAT64 = 1,
  BYTEPS_FLOAT16 = 2,
  BYTEPS_UINT8 = 3,
  BYTEPS_INT32 = 4,
  BYTEPS_INT8 = 5,
  BYTEPS_INT64 = 6,
  // below are not in mshadow, should avoid using these
  // BYTEPS_UINT16 = 7,
  // BYTEPS_INT16 = 8,
  // BYTEPS_BOOL = 9,
  // BYTEPS_BYTE = 10,
};

class CpuReducer {
 public:
  CpuReducer();
  ~CpuReducer() {
  }

  int sum(void* dst, void* src, size_t len, DataType dtype);

  // Return data of tensor
  void* GetData(const NDArray* tensor) {
    // The following returns an error:
    // return tensor->data().dptr<void>();
    switch (tensor->dtype()) {
      case mshadow::kFloat32:
        return static_cast<void*>(tensor->data().dptr<float>());
      case mshadow::kFloat64:
        return static_cast<void*>(tensor->data().dptr<double>());
      case mshadow::kFloat16:
        return static_cast<void*>(tensor->data().dptr<mshadow::half::half_t>());
      case mshadow::kUint8:
        return static_cast<void*>(tensor->data().dptr<uint8_t>());
      case mshadow::kInt32:
        return static_cast<void*>(tensor->data().dptr<int32_t>());
      case mshadow::kInt8:
        return static_cast<void*>(tensor->data().dptr<int8_t>());
      case mshadow::kInt64:
        return static_cast<void*>(tensor->data().dptr<int64_t>());
      default:
        throw std::logic_error("Type " + std::to_string(tensor->dtype()) +
                              " is not supported in BytePS.");
    }
  }

  DataType GetDType(const NDArray* tensor) {
    switch (tensor->dtype()) {
      case mshadow::kFloat32:
        return DataType::BYTEPS_FLOAT32;
      case mshadow::kFloat64:
        return DataType::BYTEPS_FLOAT64;
      case mshadow::kFloat16:
        return DataType::BYTEPS_FLOAT16;
      case mshadow::kUint8:
        return DataType::BYTEPS_UINT8;
      case mshadow::kInt32:
        return DataType::BYTEPS_INT32;
      case mshadow::kInt8:
        return DataType::BYTEPS_INT8;
      case mshadow::kInt64:
        return DataType::BYTEPS_INT64;
      default:
        throw std::logic_error("GetDType: Type " +
                              std::to_string(tensor->dtype()) +
                              " is not supported.");
    }
  }

  int64_t GetSize(const NDArray* tensor) {
    int64_t element_size = 0;
    switch (tensor->dtype()) {
      case mshadow::kFloat32:
        element_size = kFloat32Size;
        break;
      case mshadow::kFloat64:
        element_size = kFloat64Size;
        break;
      case mshadow::kFloat16:
        element_size = kFloat16Size;
        break;
      case mshadow::kUint8:
        element_size = kUInt8Size;
        break;
      case mshadow::kInt32:
        element_size = kInt32Size;
        break;
      case mshadow::kInt8:
        element_size = kInt8Size;
        break;
      case mshadow::kInt64:
        element_size = kInt64Size;
        break;
      default:
        throw std::logic_error("Type " + std::to_string(tensor->dtype()) +
                              " is not supported in BytePS.");
    }
    return (int64_t)(tensor->shape().Size()) * element_size;
  }

 private:
#if __AVX__ && __F16C__
  // Query CPUID to determine AVX and F16C runtime support.
  bool is_avx_and_f16c() {
    static bool initialized = false;
    static bool result = false;
    if (!initialized) {
      unsigned int eax, ebx, ecx, edx;
      if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        result = (ecx & bit_AVX) && (ecx & bit_F16C);
      }
      initialized = true;
    }
    return result;
  }
#endif

  inline void HalfBits2Float(unsigned short* src, float* res) {
    unsigned h = *src;
    int sign = ((h >> 15) & 1);
    int exp = ((h >> 10) & 0x1f);
    int mantissa = (h & 0x3ff);
    unsigned f = 0;

    if (exp > 0 && exp < 31) {
      // normal
      exp += 112;
      f = (sign << 31) | (exp << 23) | (mantissa << 13);
    } else if (exp == 0) {
      if (mantissa) {
        // subnormal
        exp += 113;
        while ((mantissa & (1 << 10)) == 0) {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= 0x3ff;
        f = (sign << 31) | (exp << 23) | (mantissa << 13);
      } else {
        // sign-preserving zero
        f = (sign << 31);
      }
    } else if (exp == 31) {
      if (mantissa) {
        f = 0x7fffffff;  // not a number
      } else {
        f = (0xff << 23) | (sign << 31);  //  inf
      }
    }

    *res = *reinterpret_cast<float const*>(&f);
  }

  inline void Float2HalfBits(float* src, unsigned short* dest) {
    // software implementation rounds toward nearest even
    unsigned const& s = *reinterpret_cast<unsigned const*>(src);
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
      // sign-preserving zero
      *dest = sign;
      return;
    }

    if (exp > 15) {
      if (exp == 128 && mantissa) {
        // not a number
        u = 0x7fff;
      } else {
        // overflow to infinity
        u = sign | 0x7c00;
      }
      *dest = u;
      return;
    }

    int sticky_bit = 0;

    if (exp >= -14) {
      // normal fp32 to normal fp16
      exp = uint16_t(exp + uint16_t(15));
      u = uint16_t(((exp & 0x1f) << 10));
      u = uint16_t(u | (mantissa >> 13));
    } else {
      // normal single-precision to subnormal half_t-precision representation
      int rshift = (-14 - exp);
      if (rshift < 32) {
        mantissa |= (1 << 23);

        sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

        mantissa = (mantissa >> rshift);
        u = (uint16_t(mantissa >> 13) & 0x3ff);
      } else {
        mantissa = 0;
        u = 0;
      }
    }

    // round to nearest even
    int round_bit = ((mantissa >> 12) & 1);
    sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
      u = uint16_t(u + 1);
    }

    u |= sign;

    *dest = u;
  }

  template <typename T>
  int _sum(T* dst, T* src, size_t len);

  int _sum_float16(void* dst, void* src, size_t len);

  float _convert_half_to_full_precision(uint16_t h);
  uint16_t _convert_full_to_half_precision(float f);

  int _num_threads;

  static const size_t kFloat32Size = 4;
  static const size_t kFloat64Size = 8;
  static const size_t kFloat16Size = 2;
  static const size_t kUInt8Size = 1;
  static const size_t kInt32Size = 4;
  static const size_t kInt8Size = 1;
  static const size_t kInt64Size = 8;
};


CpuReducer::CpuReducer() {
  if (getenv("BYTEPS_SERVER_OMP_THREAD_NUM")) {
    _num_threads = atoi(getenv("BYTEPS_SERVER_OMP_THREAD_NUM"));
  } else {
    _num_threads = 4;
  }
  return;
}

int CpuReducer::sum(void* dst, void* src, size_t len, DataType dtype) {
  switch (dtype) {
    case BYTEPS_FLOAT32:
      return _sum(reinterpret_cast<float*>(dst), reinterpret_cast<float*>(src),
                  len);
    case BYTEPS_FLOAT64:
      return _sum(reinterpret_cast<double*>(dst),
                  reinterpret_cast<double*>(src), len);
    case BYTEPS_FLOAT16:
      return _sum_float16(dst, src, len);
    case BYTEPS_UINT8:
      return _sum(reinterpret_cast<uint8_t*>(dst),
                  reinterpret_cast<uint8_t*>(src), len);
    case BYTEPS_INT32:
      return _sum(reinterpret_cast<int32_t*>(dst),
                  reinterpret_cast<int32_t*>(src), len);
    case BYTEPS_INT8:
      return _sum(reinterpret_cast<int8_t*>(dst),
                  reinterpret_cast<int8_t*>(src), len);
    case BYTEPS_INT64:
      return _sum(reinterpret_cast<int64_t*>(dst),
                  reinterpret_cast<int64_t*>(src), len);
    default:
      return -1;
  }
  return 0;
}

template <typename T>
int CpuReducer::_sum(T* dst, T* src, size_t len) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
    dst[i] = dst[i] + src[i];
  }
  return 0;
}

int CpuReducer::_sum_float16(void* dst, void* src, size_t len) {
  // cast src and dst to your float16 type
  auto in = (unsigned short*)src;
  auto inout = (unsigned short*)dst;
  len = len / (size_t)2;

#if __AVX__ && __F16C__
  if (is_avx_and_f16c()) {
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < (size_t)(len / 8) * 8; i += 8) {
      // convert in & inout to m256
      __m256 in_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in + i)));
      __m256 inout_m256 =
          _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(inout + i)));

      // add them together to new_inout_m256
      __m256 new_inout_m256 = _mm256_add_ps(in_m256, inout_m256);

      // convert back and store in inout
      __m128i new_inout_m128i = _mm256_cvtps_ph(new_inout_m256, 0);
      _mm_storeu_si128((__m128i*)(inout + i), new_inout_m128i);
    }
  }
#endif
  for (size_t i = (len / 8) * 8; i < (size_t)len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float += in_float;
    Float2HalfBits(&inout_float, inout + i);
  }

  return 0;
}

}  // namespace kvstore
}  // namespace mxnet

#endif  // BYTEPS_CPU_REDUCER_H
