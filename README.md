Playing with Alpaka, starting with files in 

https://github.com/cms-sw/cmssw/tree/902185ebb53bc03dd80c6bfd5cb50a5667d572e3/DataFormats/TrackingRecHitSoA/test/alpaka

compiled with `scram b` the executable can be found here:

`CMSSW_14_0_15/tmp/el8_amd64_gcc12/src/DataFormats/TrackingRecHitSoA/test/Hits_testCudaAsync`

The code initialize each thread of each block with a random number for x and y Location, then an added kernel checks for a pattern witnin the x location of the hit
