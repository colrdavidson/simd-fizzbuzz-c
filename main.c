#include <stdio.h>
#include <stdint.h>
#include <smmintrin.h>
#include <immintrin.h>

// This is basically a translation from a random Java SIMD Fizzbuzz blogpost to C
// Was interesting because Java's IntVector interface doesn't map exactly to intel intrinsics
// Forced me to have to write some helpers and dig into the finer points of the intrinsics guide
// here and there. Could make it fancier, but eh, it's fizzbuzz.

// TODO: Is there an intrinsic / native inst for this?
static inline __m256i broadcast_256(int32_t a) {
	return _mm256_setr_epi32(a, a, a, a, a, a, a, a);
}

// This is goofy nonsense to make blendv_epi8 happy...
// blendv_epi8 wants an 8bit mask spread over 256 bits rather than just taking an imm8.
static inline __m256i spread_256(int8_t a) {
	int v[8];
	for (int i = 0; i < 8; i++) {
		uint8_t bit = !!((1 << i) & a);
		v[i] = bit * -1;
	}
	return _mm256_loadu_si256((__m256i*)v);
}

static void print_256(__m256i v) {
	for (int i = 0; i < 8; i++) {
		printf("%d ", ((int *)&v)[i]);
	}
	printf("\n");
}

int main(void) {

	// This is a *lot* of pre-cursor work which boils down to a fairly straightfowrward process
	// This generates 15 distinct masks (res_masks), and 15 vecs with values-to-swap (res_vecs),
	// res_masks determines if the value needs to be swapped with a const, and res_vecs contains
	// the constant appropriate for that position in the cycle.
	//
	// Ex: For count (1, 2, 3, 4, 5, 6, 7, 8), we apply res_vecs[0] (0, 0, -1, 0, -2, -1, 0, 0)
	// using the -1/0 in the corresponding 32b slot in res_masks[0] (0, 0, -1, 0, -1, -1, 0, 0)
	// to determine which to swap, resulting in (1, 2, -1, 4, -2, -1, 7, 8)
	//
	// The threes and fives masks, read right to left, mark out explicitly,
	// each multiple of 3 or 5 respectively that needs to be handled. This could be generated,
	// but it's arguably clearer written as static data, and won't need to change because FizzBuzz
	//
	// TODO: Is there a better way to do this that'll just swap in the vec val if it's nonzero?

	__m256i threes[3] = {
		spread_256(0b00100100),
		spread_256(0b01001001),
		spread_256(0b10010010)
	};

	__m256i fives[5] = {
		spread_256(0b00010000),
		spread_256(0b01000010),
		spread_256(0b00001000),
		spread_256(0b00100001),
		spread_256(0b10000100)
	};

	__m256i zero = broadcast_256(0);

	__m256i res_masks[15];
	__m256i res_vecs[15];

	__m256i fizz_v = broadcast_256(-1);
	__m256i buzz_v = broadcast_256(-2);
	__m256i fizzbuzz_v = broadcast_256(-3);

	for (int i = 0; i < 15; i++) {
		__m256i three_mask = threes[i % 3];
		__m256i five_mask  = fives[i % 5];
		__m256i three_and_five_mask = _mm256_and_si256(three_mask, five_mask);

		res_masks[i] = _mm256_or_si256(three_mask, five_mask);

		__m256i r1 = _mm256_blendv_epi8(zero, fizz_v, three_mask);
		__m256i r2 = _mm256_blendv_epi8(r1, buzz_v, five_mask);
		res_vecs[i] = _mm256_blendv_epi8(r2, fizzbuzz_v, three_and_five_mask);
	}

	// Building our starting count vec containing ints 1 -> 8
	__m256i chunk_iter = broadcast_256(8);
	int count_arr[8];
	for (int i = 0; i < 8; i++) {
		count_arr[i] = i + 1;
	}
	__m256i count = _mm256_loadu_si256((__m256i*)count_arr);

	// Completely ignoring the case where you might want to fizz a buzz that isn't an evenly-divisble-by-8 count
	// Imagine your own scalar/partial-vec solution to fill in the blanks if you feel so inclined.
	int k = 0;
	int length = 256 / 8;
	for (int i = 0; i < length; i++) {

		// This picks either the count value or the appropriate fizz, buzz, or fizzbuzz constant,
		// using all the pre-work we did before, generating masks, as the determining factor
		__m256i chunk = _mm256_blendv_epi8(count, res_vecs[k], res_masks[k]);

		for (int j = 0; j < 8; j++) {
			int val = ((int *)&chunk)[j];

			// While it might technically be "cleaner" to write this with a string table lookup,
			// the small switch case is faster, and for a SIMD impl speed is king
			switch (val) {
				case -1: {
					printf("fizz\n");
				} break;
				case -2: {
					printf("buzz\n");
				} break;
				case -3: {
					printf("fizzbuzz\n");
				} break;
				default: {
					printf("%d\n", val);
				} break;
			}
		}

		count = _mm256_add_epi32(count, chunk_iter);

		// Restarts the mask/vecs cycle without needing a modulo
		k++;
		if (k == 15) {
			k = 0;
		}
	}

	return 0;
}
