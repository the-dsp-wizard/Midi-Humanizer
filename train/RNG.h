#ifndef RNG_H_
#define RNG_H_

#include <stdint.h>

#ifndef finline
#if defined(__clang__)  || defined(__GNUC__) || defined(__GNUG__)
#define finline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define finline __forceinline
#else
#define finline  
#endif 
#endif

typedef struct {
    uint64_t state;
} RNG ;

void rng_seed(RNG *rng, uint32_t seed) {
    rng->state = seed;
}

uint32_t rng_pcg32(RNG *rng) {
	uint64_t x = rng->state;
	uint32_t count = (unsigned)(x >> 59);

	rng->state = x * 6364136223846793005u + 1442695040888963407u;
	x ^= x >> 18;
	return (uint32_t)(x >> 27) >> count | (uint32_t)(x >> 27) << (-count & 31);
}

#endif