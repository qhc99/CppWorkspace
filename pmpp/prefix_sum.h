#ifndef PPMP_PREFIX_SUM_KERNEL_H
#define PPMP_PREFIX_SUM_KERNEL_H
/**
 * @brief Compute prefix sum of segments
 *
 * @param data
 * @param out
 * @param length
 */
void KoggeStoneSegmentScan(float* data, float* out, unsigned int length);


/**
 * @brief Compute prefix sum of segments
 *
 * @param data
 * @param out
 * @param length
 */
void BrentKungSegmentScan(float* data, float* out, unsigned int length);

#endif