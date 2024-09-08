#ifndef PPMP_HIST_KERNEL_H
#define PPMP_HIST_KERNEL_H

void hist(char* data, unsigned int length, unsigned int* hist_out, unsigned int bin_width, bool use_aggregation = false);

#endif