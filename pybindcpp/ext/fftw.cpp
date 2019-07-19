// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

#include <fftw3.h>
#include <complex>
#include <numpy/arrayobject.h>
#include <pybindcpp/module.h>

using namespace pybindcpp;

void init(module m)
{
  m.add("fft", [](size_t N, std::complex<double> *in, std::complex<double> *out) {
    auto _in = reinterpret_cast<fftw_complex *>(in);
    auto _out = reinterpret_cast<fftw_complex *>(out);
    auto plan = fftw_plan_dft_1d(N, _in, _out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  });


  m.add("fft2", [](size_t N, size_t M, std::complex<double> *_in, std::complex<double> *_out) {
    auto in = reinterpret_cast<fftw_complex *>(_in);
    auto out = reinterpret_cast<fftw_complex *>(_out);
    auto p = fftw_plan_dft_1d(N, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);

    for (size_t i = 0; i < M; i++)
    {
      fftw_execute_dft(p, in, out);
      in += N;
      out += N;
    }
    fftw_destroy_plan(p);
  });

}

PyMODINIT_FUNC PyInit_fftw()
{
  import_array();
  return pybindcpp::init_module("fftw", init);
}
