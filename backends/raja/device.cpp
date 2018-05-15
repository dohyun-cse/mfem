// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
#include "raja.hpp"

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{
   device::device() {}
   device::~device() {}
  // ***************************************************************************
  /*device& device::Get()
  {
    static device device_singleton;
    return device_singleton;
    }*/
   
  // ***************************************************************************
  /*device* device::getDevice()
  {
    static device device_singleton;
    return &device_singleton;
    }*/
    
  // ***************************************************************************
  bool device::hasSeparateMemorySpace(){
    return false;
  }

  // ***************************************************************************
  memory device::malloc(const std::size_t bytes,
                        const void *src){
     assert(src==NULL);
     return memory(bytes,src);
  }

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
