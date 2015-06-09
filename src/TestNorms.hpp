/*!
 @file TestNorms.hpp

 HPCG data structure
 */

#ifndef TESTNORMS_HPP
#define TESTNORMS_HPP

#include "KokkosSetup.hpp"


struct TestNormsData_STRUCT {
  double_1d_type values; //!< sample values
  double   mean;   //!< mean of all sampes
  double variance; //!< variance of mean
  int    samples;  //!< number of samples
  bool   pass;     //!< pass/fail indicator
};
typedef struct TestNormsData_STRUCT TestNormsData;

extern int TestNorms(TestNormsData & testnorms_data);

#endif  // TESTNORMS_HPP
