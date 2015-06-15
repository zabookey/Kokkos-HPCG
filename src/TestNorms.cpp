/*!
 @file TestNorms.cpp

 HPCG routine
 */

#include <cmath>
#include "TestNorms.hpp"

/*!
  Computes the mean and standard deviation of the array of norm results.

  @param[in] testnorms_data data structure with the results of norm test

  @return Returns 0 upon success or non-zero otherwise
*/
int TestNorms(TestNormsData & testnorms_data) {
 double mean_delta = 0.0;
 // Need to mirror data.values.
 host_double_1d_type host_values= Kokkos::create_mirror_view(testnorms_data.values);
 deep_copy(host_values, testnorms_data.values);
 for (int i= 0; i<testnorms_data.samples; ++i) mean_delta += (host_values(i) - host_values(0));
 double mean = host_values(0) + mean_delta/(double)testnorms_data.samples;
 testnorms_data.mean = mean;

 // Compute variance
 double sumdiff = 0.0;
 for (int i= 0; i<testnorms_data.samples; ++i) sumdiff += (host_values(i) - mean) * (host_values(i) - mean);
 testnorms_data.variance = sumdiff/(double)testnorms_data.samples;
 // This may be an unnecessary copy.
 Kokkos::deep_copy(testnorms_data.values, host_values);

 // Determine if variation is sufficiently small to declare success
 testnorms_data.pass = (testnorms_data.variance<1.0e-6);

 return 0;
}
