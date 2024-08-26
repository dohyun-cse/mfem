// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "gslib.hpp"
#include "geom.hpp"

#ifdef MFEM_USE_GSLIB

// Ignore warnings from the gslib header (GCC version)
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

// External GSLIB header (the MFEM header is gslib.hpp)
namespace gslib
{
#include "gslib.h"
extern "C" {
   struct hash_data_3
   {
      ulong hash_n;
      struct dbl_range bnd[3];
      double fac[3];
      uint *offset;
   };

   struct hash_data_2
   {
      ulong hash_n;
      struct dbl_range bnd[2];
      double fac[2];
      uint *offset;
   };

   struct findpts_dummy_ms_data
   {
      unsigned int *nsid;
      double       *distfint;
   };

   struct findpts_data_3
   {
      struct crystal cr;
      struct findpts_local_data_3 local;
      struct hash_data_3 hash;
      struct array savpt;
      struct findpts_dummy_ms_data fdms;
      uint   fevsetup;
   };

   struct findpts_data_2
   {
      struct crystal cr;
      struct findpts_local_data_2 local;
      struct hash_data_2 hash;
      struct array savpt;
      struct findpts_dummy_ms_data fdms;
      uint   fevsetup;
   };
} //extern C

} //namespace gslib

#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif

namespace mfem
{

FindPointsGSLIB::FindPointsGSLIB()
   : mesh(NULL),
     fec_map_lin(NULL),
     fdataD(NULL), cr(NULL), gsl_comm(NULL),
     dim(-1), points_cnt(0), setupflag(false), default_interp_value(0),
     avgtype(AvgType::ARITHMETIC), bdr_tol(1e-8)
{
   mesh_split.SetSize(4);
   ir_split.SetSize(4);
   fes_rst_map.SetSize(4);
   gf_rst_map.SetSize(4);
   for (int i = 0; i < mesh_split.Size(); i++)
   {
      mesh_split[i] = NULL;
      ir_split[i] = NULL;
      fes_rst_map[i] = NULL;
      gf_rst_map[i] = NULL;
   }

   gsl_comm = new gslib::comm;
   cr       = new gslib::crystal;
#ifdef MFEM_USE_MPI
   int initialized;
   MPI_Initialized(&initialized);
   if (!initialized) { MPI_Init(NULL, NULL); }
   MPI_Comm comm = MPI_COMM_WORLD;
   comm_init(gsl_comm, comm);
#else
   comm_init(gsl_comm, 0);
#endif
   crystal_init(cr, gsl_comm);
}

FindPointsGSLIB::~FindPointsGSLIB()
{
   crystal_free(cr);
   comm_free(gsl_comm);
   delete gsl_comm;
   delete cr;
   for (int i = 0; i < 4; i++)
   {
      if (mesh_split[i]) { delete mesh_split[i]; mesh_split[i] = NULL; }
      if (ir_split[i]) { delete ir_split[i]; ir_split[i] = NULL; }
      if (fes_rst_map[i]) { delete fes_rst_map[i]; fes_rst_map[i] = NULL; }
      if (gf_rst_map[i]) { delete gf_rst_map[i]; gf_rst_map[i] = NULL; }
   }
   if (fec_map_lin) { delete fec_map_lin; fec_map_lin = NULL; }
}

#ifdef MFEM_USE_MPI
FindPointsGSLIB::FindPointsGSLIB(MPI_Comm comm_)
   : mesh(NULL),
     fec_map_lin(NULL),
     fdataD(NULL), cr(NULL), gsl_comm(NULL),
     dim(-1), points_cnt(0), setupflag(false), default_interp_value(0),
     avgtype(AvgType::ARITHMETIC), bdr_tol(1e-8)
{
   mesh_split.SetSize(4);
   ir_split.SetSize(4);
   fes_rst_map.SetSize(4);
   gf_rst_map.SetSize(4);
   for (int i = 0; i < mesh_split.Size(); i++)
   {
      mesh_split[i] = NULL;
      ir_split[i] = NULL;
      fes_rst_map[i] = NULL;
      gf_rst_map[i] = NULL;
   }

   gsl_comm = new gslib::comm;
   cr      = new gslib::crystal;
   comm_init(gsl_comm, comm_);
   crystal_init(cr, gsl_comm);
}
#endif

void FindPointsGSLIB::Setup(Mesh &m, const double bb_t, const double newt_tol,
                            const int npt_max)
{
   MFEM_VERIFY(m.GetNodes() != NULL, "Mesh nodes are required.");
   const int meshOrder = m.GetNodes()->FESpace()->GetMaxElementOrder();

   // call FreeData if FindPointsGSLIB::Setup has been called already
   if (setupflag) { FreeData(); }

   mesh = &m;
   dim  = mesh->Dimension();
   unsigned dof1D = meshOrder + 1;

   setupSW.Clear();
   setupSW.Start();
   SetupSplitMeshes();
   if (dim == 2)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(3*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);

      if (ir_split[1]) { delete ir_split[1]; ir_split[1] = NULL; }
      ir_split[1] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[1], ir_split[1], meshOrder);
   }
   else if (dim == 3)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);

      if (ir_split[1]) { delete ir_split[1]; ir_split[1] = NULL; }
      ir_split[1] = new IntegrationRule(4*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[1], ir_split[1], meshOrder);

      if (ir_split[2]) { delete ir_split[2]; ir_split[2] = NULL; }
      ir_split[2] = new IntegrationRule(3*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[2], ir_split[2], meshOrder);

      if (ir_split[3]) { delete ir_split[3]; ir_split[3] = NULL; }
      ir_split[3] = new IntegrationRule(8*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[3], ir_split[3], meshOrder);
   }
   setupSW.Stop();
   setup_split_time = setupSW.RealTime();

   setupSW.Clear();
   setupSW.Start();
   GetNodalValues(mesh->GetNodes(), gsl_mesh);
   setupSW.Stop();
   setup_nodalmapping_time = setupSW.RealTime();

   mesh_points_cnt = gsl_mesh.Size()/dim;

   DEV.local_hash_size = mesh_points_cnt;
   DEV.dof1d = (int)dof1D;
   setupSW.Clear();
   setupSW.Start();
   if (dim == 2)
   {
      unsigned nr[2] = { dof1D, dof1D };
      unsigned mr[2] = { 2*dof1D, 2*dof1D };
      double * const elx[2] =
      {
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(0),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(mesh_points_cnt)
      };
      fdataD = findpts_setup_2(gsl_comm, elx, nr, NE_split_total, mr, bb_t,
                               DEV.local_hash_size,
                               mesh_points_cnt, npt_max, newt_tol);
   }
   else
   {
      unsigned nr[3] = { dof1D, dof1D, dof1D };
      unsigned mr[3] = { 2*dof1D, 2*dof1D, 2*dof1D };
      double * const elx[3] =
      {
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(0),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(mesh_points_cnt),
         mesh_points_cnt == 0 ? nullptr : &gsl_mesh(2*mesh_points_cnt)
      };
      fdataD = findpts_setup_3(gsl_comm, elx, nr, NE_split_total, mr, bb_t,
                               DEV.local_hash_size,
                               mesh_points_cnt, npt_max, newt_tol);
   }
   setupSW.Stop();
   setup_findpts_setup_time = setupSW.RealTime();
   setupflag = true;
}

void FindPointsGSLIB::FindPoints(const Vector &point_pos,
                                 int point_pos_ordering)
{
   MFEM_VERIFY(setupflag, "Use FindPointsGSLIB::Setup before finding points.");
   bool dev_mode = (point_pos.UseDevice() && Device::IsEnabled());
   points_cnt = point_pos.Size() / dim;
   gsl_code.SetSize(points_cnt);
   gsl_proc.SetSize(points_cnt);
   gsl_elem.SetSize(points_cnt);
   gsl_ref.SetSize(points_cnt * dim);
   gsl_dist.SetSize(points_cnt);

   setupSW.Clear();
   setupSW.Start();
   if (dev_mode)
   {
      FindPointsOnDevice(point_pos, point_pos_ordering);
      return;
      setupSW.Stop();
      findpts_findpts_time = setupSW.RealTime();
      findpts_mapelemrst_time = 0.0;
   }

   auto xvFill = [&](const double *xv_base[], unsigned xv_stride[])
   {
      for (int d = 0; d < dim; d++)
      {
         if (point_pos_ordering == Ordering::byNODES)
         {
            xv_base[d] = point_pos.GetData() + d*points_cnt;
            xv_stride[d] = sizeof(double);
         }
         else
         {
            xv_base[d] = point_pos.GetData() + d;
            xv_stride[d] = dim*sizeof(double);
         }
      }
   };

   if (dim == 2)
   {
      auto *findptsData = (gslib::findpts_data_2 *)this->fdataD;
      const double *xv_base[2];
      unsigned xv_stride[2];
      xvFill(xv_base, xv_stride);
      findpts_2(gsl_code.GetData(), sizeof(unsigned int),
                gsl_proc.GetData(), sizeof(unsigned int),
                gsl_elem.GetData(), sizeof(unsigned int),
                gsl_ref.GetData(),  sizeof(double) * dim,
                gsl_dist.GetData(), sizeof(double),
                xv_base, xv_stride, points_cnt, findptsData);
   }
   else  // dim == 3
   {
      auto *findptsData = (gslib::findpts_data_3 *)this->fdataD;
      const double *xv_base[3];
      unsigned xv_stride[3];
      xvFill(xv_base, xv_stride);
      findpts_3(gsl_code.GetData(), sizeof(unsigned int),
                gsl_proc.GetData(), sizeof(unsigned int),
                gsl_elem.GetData(), sizeof(unsigned int),
                gsl_ref.GetData(),  sizeof(double) * dim,
                gsl_dist.GetData(), sizeof(double),
                xv_base, xv_stride, points_cnt,
                findptsData);
   }

   setupSW.Stop();
   findpts_findpts_time = setupSW.RealTime();

   setupSW.Clear();
   setupSW.Start();
   // Set the element number and reference position to 0 for points not found
   for (int i = 0; i < points_cnt; i++)
   {
      if (gsl_code[i] == 2 ||
          (gsl_code[i] == 1 && gsl_dist(i) > bdr_tol))
      {
         gsl_elem[i] = 0;
         for (int d = 0; d < dim; d++) { gsl_ref(i*dim + d) = -1.; }
         gsl_code[i] = 2;
      }
   }

   // Map element number for simplices, and ref_pos from [-1,1] to [0,1] for
   // both simplices and quads. Also sets code to 1 for points found on element
   // faces/edges.
   MapRefPosAndElemIndices();

   setupSW.Stop();
   findpts_mapelemrst_time = setupSW.RealTime();
}

static slong lfloor(double x) { return floor(x); }

static ulong hash_index_aux(double low, double fac, ulong n, double x)
{
   const slong i = lfloor((x - low) * fac);
   return i < 0 ? 0 : (n - 1 < (ulong)i ? n - 1 : (ulong)i);
}

static ulong hash_index_3(const gslib::hash_data_3 *p, const double x[3])
{
   const ulong n = p->hash_n;
   return (hash_index_aux(p->bnd[2].min, p->fac[2], n, x[2]) * n +
           hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) *
          n +
          hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

static ulong hash_index_2(const gslib::hash_data_2 *p, const double x[2])
{
   const ulong n = p->hash_n;
   return (hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) * n
          + hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

void FindPointsGSLIB::FindPointsOnDevice(const Vector &point_pos,
                                         int point_pos_ordering)
{
   SW2.Clear();
   SW2.Start();
   if (!DEV.setup_device) { SetupDevice(); }
   SW2.Stop();
   findpts_setup_device_arrays_time = SW2.RealTime();

   gsl_ref.UseDevice(true);
   gsl_dist.UseDevice(true);

   if (dim == 2)
   {
      FindPointsLocal2(point_pos, point_pos_ordering,
                       gsl_code, gsl_elem, gsl_ref, gsl_dist, points_cnt);
   }
   else
   {
      FindPointsLocal3(point_pos, point_pos_ordering,
                       gsl_code, gsl_elem, gsl_ref, gsl_dist, points_cnt);
   }

   // Sync from device to host
   gsl_ref.HostReadWrite();
   gsl_dist.HostReadWrite();
   gsl_code.HostReadWrite();
   gsl_elem.HostReadWrite();
   point_pos.HostRead();
   gsl_proc.HostWrite();

   gsl_mfem_ref.SetSize(points_cnt*dim);
   gsl_mfem_ref = gsl_ref.HostRead();
   gsl_mfem_ref += 1.;  // map  [-1, 1] to [0, 2] to [0, 1]
   gsl_mfem_ref *= 0.5;

   // tolerance for point to be marked as on element edge/face
   double btol = 1e-12; // must match MapRefPosAndElemIndices for consistency

   const int id = gsl_comm->id,
             np = gsl_comm->np;
   for (int i = 0; i < points_cnt; i++)
   {
      gsl_proc[i] = id;
   }

   if (np == 1)
   {
      for (int index = 0; index < points_cnt; index++)
      {
         if (gsl_code[index] == CODE_NOT_FOUND)
         {
            continue;
         }
         IntegrationPoint ip;
         if (dim == 2)
         {
            ip.Set2(gsl_mfem_ref.GetData() + index*dim);
         }
         else if (dim == 3)
         {
            ip.Set3(gsl_mfem_ref.GetData() + index*dim);
         }
         const int elem = gsl_elem[index];
         const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(elem);
         const Geometry::Type gt = fe->GetGeomType();
         if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
         {
            int setcode = Geometry::CheckPoint(gt, ip, -btol) ?
                          CODE_INTERNAL : CODE_BORDER;
            gsl_code[index] = setcode==CODE_BORDER && gsl_dist(index)>bdr_tol ?
                              CODE_NOT_FOUND : setcode;
         }
      }
      return;
   }

   MPI_Barrier(gsl_comm->c);
   /* send unfound and border points to global hash cells */
   struct gslib::array hash_pt, src_pt, out_pt;

   struct srcPt_t
   {
      double x[3];
      unsigned int index, proc;
   };

   struct outPt_t
   {
      double r[3], dist2;
      unsigned int index, code, el, proc;
   };

   {
      int index;
      struct srcPt_t *pt;

      array_init(struct srcPt_t, &hash_pt, points_cnt);
      pt = (struct srcPt_t *)hash_pt.ptr;

      double x[dim];
      for (index = 0; index < points_cnt; ++index)
      {
         if (gsl_code[index] != CODE_INTERNAL)
         {
            for (int d = 0; d < dim; ++d)
            {
               int idx = point_pos_ordering == 0 ?
                         index + d*points_cnt :
                         index*dim + d;
               x[d] = point_pos(idx);
            }
            const auto hi = dim == 2 ? hash_index_2(DEV.hash2, x) :
                            hash_index_3(DEV.hash3, x);
            for (int d = 0; d < dim; ++d)
            {
               pt->x[d] = x[d];
            }
            pt->index = index;
            pt->proc = hi % np;
            ++pt;
         }
      }
      hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
      sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, DEV.cr);
   }
   MPI_Barrier(gsl_comm->c);

   /* look up points in hash cells, route to possible procs */
   {
      const unsigned int *const hash_offset = dim == 2 ? DEV.hash2->offset :
                                              DEV.hash3->offset;
      int count = 0;
      unsigned int *proc, *proc_p;
      const struct srcPt_t *p = (struct srcPt_t *)hash_pt.ptr,
                            *const pe = p + hash_pt.n;
      struct srcPt_t *q;

      for (; p != pe; ++p)
      {
         const int hi = dim == 2 ? hash_index_2(DEV.hash2, p->x)/np :
                        hash_index_3(DEV.hash3, p->x)/np;
         const int i = hash_offset[hi], ie = hash_offset[hi + 1];
         count += ie - i;
      }

      Array<unsigned int> proc_array(count);
      proc = proc_array.GetData();
      proc_p = proc;
      array_init(struct srcPt_t, &src_pt, count);
      q = (struct srcPt_t *)src_pt.ptr;

      p = (struct srcPt_t *)hash_pt.ptr;
      for (; p != pe; ++p)
      {
         const int hi = dim == 2 ? hash_index_2(DEV.hash2, p->x)/np :
                        hash_index_3(DEV.hash3, p->x)/np;
         int i = hash_offset[hi];
         const int ie = hash_offset[hi + 1];
         for (; i != ie; ++i)
         {
            const int pp = hash_offset[i];
            /* don't send back to where it just came from */
            if (pp == p->proc)
            {
               continue;
            }
            *proc_p++ = pp;
            *q++ = *p;
         }
      }

      array_free(&hash_pt);
      src_pt.n = proc_p - proc;

      sarray_transfer_ext(struct srcPt_t, &src_pt, proc, sizeof(uint), DEV.cr);
   }
   MPI_Barrier(gsl_comm->c);

   /* look for other procs' points, send back */
   {
      int n = src_pt.n;
      const struct srcPt_t *spt;
      struct outPt_t *opt;
      array_init(struct outPt_t, &out_pt, n);
      out_pt.n = n;
      spt = (struct srcPt_t *)src_pt.ptr;
      opt = (struct outPt_t *)out_pt.ptr;
      for (; n; --n, ++spt, ++opt)
      {
         opt->index = spt->index;
         opt->proc = spt->proc;
      }
      spt = (struct srcPt_t *)src_pt.ptr;
      opt = (struct outPt_t *)out_pt.ptr;

      n = out_pt.n;
      Vector gsl_ref_l, gsl_dist_l;
      gsl_ref_l.UseDevice(true); gsl_ref_l.SetSize(n*dim);
      gsl_dist_l.UseDevice(true); gsl_dist_l.SetSize(n);

      Vector point_pos_l;
      point_pos_l.UseDevice(true); point_pos_l.SetSize(n*dim);
      auto pointl = point_pos_l.HostWrite();

      Array<unsigned int> gsl_code_l(n), gsl_elem_l(n);

      for (int point = 0; point < n; ++point)
      {
         for (int d = 0; d < dim; d++)
         {
            int idx = point_pos_ordering == 0 ? point + d*n : point*dim + d;
            pointl[idx] = spt[point].x[d];
         }
      }

      if (dim == 2)
      {
         FindPointsLocal2(point_pos_l, point_pos_ordering,
                          gsl_code_l, gsl_elem_l, gsl_ref_l, gsl_dist_l, n);
      }
      else
      {
         FindPointsLocal3(point_pos_l, point_pos_ordering,
                          gsl_code_l, gsl_elem_l, gsl_ref_l, gsl_dist_l, n);
      }

      gsl_ref_l.HostRead();
      gsl_dist_l.HostRead();
      gsl_code_l.HostRead();
      gsl_elem_l.HostRead();

      // unpack arrays into opt
      for (int point = 0; point < n; point++)
      {
         opt[point].code = AsConst(gsl_code_l)[point];
         if (opt[point].code == CODE_NOT_FOUND)
         {
            continue;
         }
         opt[point].el   = AsConst(gsl_elem_l)[point];
         opt[point].dist2 = AsConst(gsl_dist_l)[point];
         for (int d = 0; d < dim; ++d)
         {
            opt[point].r[d] = AsConst(gsl_ref_l)[dim * point + d];
         }
         IntegrationPoint ip;
         ip.Set3(&opt[point].r[0]);
         const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(opt[point].el);
         const Geometry::Type gt = fe->GetGeomType();
         if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
         {
            opt[point].code = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
         }
      }

      array_free(&src_pt);

      /* group by code to eliminate unfound points */
      sarray_sort(struct outPt_t, opt, out_pt.n, code, 0, &DEV.cr->data);

      n = out_pt.n;
      while (n && opt[n - 1].code == CODE_NOT_FOUND)
      {
         --n;
      }
      out_pt.n = n;

      sarray_transfer(struct outPt_t, &out_pt, proc, 1, DEV.cr);
   }
   MPI_Barrier(gsl_comm->c);

   /* merge remote results with user data */
   {
      int n = out_pt.n;
      struct outPt_t *opt = (struct outPt_t *)out_pt.ptr;
      for (; n; --n, ++opt)
      {
         const int index = opt->index;
         if (gsl_code[index] == CODE_INTERNAL)
         {
            continue;
         }
         if (gsl_code[index] == CODE_NOT_FOUND || opt->code == CODE_INTERNAL ||
             opt->dist2 < gsl_dist[index])
         {
            for (int d = 0; d < dim; ++d)
            {
               gsl_ref(dim * index + d) = opt->r[d];
               gsl_mfem_ref(dim*index + d) = 0.5*(opt->r[d] + 1.);
            }
            gsl_dist[index] = opt->dist2;
            gsl_proc[index] = opt->proc;
            gsl_elem[index] = opt->el; // skip setting gsl_mfem_elem
            gsl_code[index] = opt->code;
         }
      }
      array_free(&out_pt);
   }

   // Set code for local points
   for (int index = 0; index < points_cnt; index++)
   {
      if (gsl_code[index] == CODE_NOT_FOUND)
      {
         continue;
      }
      if (gsl_proc[index] != id)
      {
         if (gsl_code[index] == CODE_BORDER && gsl_dist(index) > bdr_tol)
         {
            gsl_code[index] = CODE_NOT_FOUND;
         }
         continue;
      }
      IntegrationPoint ip;
      if (dim == 2)
      {
         ip.Set2(gsl_mfem_ref.GetData() + index*dim);
      }
      else if (dim == 3)
      {
         ip.Set3(gsl_mfem_ref.GetData() + index*dim);
      }
      const int elem = gsl_elem[index];
      const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(elem);
      const Geometry::Type gt = fe->GetGeomType();
      if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
      {
         int setcode = Geometry::CheckPoint(gt, ip, -btol) ?
                       CODE_INTERNAL : CODE_BORDER;
         gsl_code[index] = setcode==CODE_BORDER && gsl_dist(index)>bdr_tol ?
                           CODE_NOT_FOUND : setcode;
      }
   }

   MPI_Barrier(gsl_comm->c);
}

void FindPointsGSLIB::SetupDevice()
{
   auto *findptsData3 = (gslib::findpts_data_3 *)this->fdataD;
   auto *findptsData2 = (gslib::findpts_data_2 *)this->fdataD;

   DEV.tol = dim == 2 ? findptsData2->local.tol : findptsData3->local.tol;
   if (dim == 3)
   {
      DEV.hash3 = &findptsData3->hash;
   }
   else
   {
      DEV.hash2 = &findptsData2->hash;
   }
   DEV.cr = dim == 2 ? &findptsData2->cr : &findptsData3->cr;

   gsl_mesh.UseDevice(true);

   int n_box_ents = 3*dim + dim*dim;
   DEV.bb.UseDevice(true); DEV.bb.SetSize(n_box_ents*NE_split_total);
   auto p_bb = DEV.bb.HostWrite();

   const int dim2 = dim*dim;
   if (dim == 3)
   {
      for (int e = 0; e < NE_split_total; e++)
      {
         auto box = findptsData3->local.obb[e];
         for (int d = 0; d < dim; d++)
         {
            p_bb[n_box_ents*e + d] = box.c0[d];
            p_bb[n_box_ents*e + dim + d] = box.x[d].min;
            p_bb[n_box_ents*e + 2*dim + d] = box.x[d].max;
         }
         for (int d = 0; d < dim2; ++d)
         {
            p_bb[n_box_ents*e + 3*dim + d] = box.A[d];
         }
      }
   }
   else
   {
      for (int e = 0; e < NE_split_total; e++)
      {
         auto box = findptsData2->local.obb[e];
         for (int d = 0; d < dim; d++)
         {
            p_bb[n_box_ents*e + d] = box.c0[d];
            p_bb[n_box_ents*e + dim + d] = box.x[d].min;
            p_bb[n_box_ents*e + 2*dim + d] = box.x[d].max;
         }
         for (int d = 0; d < dim2; ++d)
         {
            p_bb[n_box_ents*e + 3*dim + d] = box.A[d];
         }
      }
   }

   DEV.loc_hash_min.UseDevice(true); DEV.loc_hash_min.SetSize(dim);
   DEV.loc_hash_fac.UseDevice(true); DEV.loc_hash_fac.SetSize(dim);
   if (dim == 2)
   {
      auto hash = findptsData2->local.hd;
      auto p_loc_hash_min = DEV.loc_hash_min.HostWrite();
      auto p_loc_hash_fac = DEV.loc_hash_fac.HostWrite();
      for (int d = 0; d < dim; d++)
      {
         p_loc_hash_min[d] = hash.bnd[d].min;
         p_loc_hash_fac[d] = hash.fac[d];
      }
      DEV.h_nx = hash.hash_n;
   }
   else
   {
      auto hash = findptsData3->local.hd;
      auto p_loc_hash_min = DEV.loc_hash_min.HostWrite();
      auto p_loc_hash_fac = DEV.loc_hash_fac.HostWrite();
      for (int d = 0; d < dim; d++)
      {
         p_loc_hash_min[d] = hash.bnd[d].min;
         p_loc_hash_fac[d] = hash.fac[d];
      }
      DEV.h_nx = hash.hash_n;
   }

   DEV.h_o_size = dim == 2 ?
                  findptsData2->local.hd.offset[(int)std::pow(DEV.h_nx, dim)] :
                  findptsData3->local.hd.offset[(int)std::pow(DEV.h_nx, dim)];

   DEV.loc_hash_offset.SetSize(DEV.h_o_size);
   auto p_ou_offset = DEV.loc_hash_offset.HostWrite();
   for (int i = 0; i < DEV.h_o_size; i++)
   {
      p_ou_offset[i] = dim == 2 ? findptsData2->local.hd.offset[i] :
                       findptsData3->local.hd.offset[i];
   }

   DEV.wtend.UseDevice(true);
   DEV.wtend.SetSize(6*DEV.dof1d);
   DEV.wtend.HostWrite();
   DEV.wtend = dim == 2 ? findptsData2->local.fed.wtend[0] :
               findptsData3->local.fed.wtend[0];

   // Get gll points
   DEV.gll1d.UseDevice(true);
   DEV.gll1d.SetSize(DEV.dof1d);
   DEV.gll1d.HostWrite();
   DEV.gll1d = dim == 2 ? findptsData2->local.fed.z[0] :
               findptsData3->local.fed.z[0];

   DEV.lagcoeff.UseDevice(true);
   DEV.lagcoeff.SetSize(DEV.dof1d);
   DEV.lagcoeff.HostWrite();
   DEV.lagcoeff = dim == 2 ? findptsData2->local.fed.lag_data[0] :
                  findptsData3->local.fed.lag_data[0];

   DEV.setup_device = true;
}

Mesh* FindPointsGSLIB::GetBoundingBoxMesh(int type)
{
   MFEM_VERIFY(setupflag, "Call FindPointsGSLIB::Setup method first");

   auto *findptsData3 = (gslib::findpts_data_3 *)this->fdataD;
   auto *findptsData2 = (gslib::findpts_data_2 *)this->fdataD;

   int save_rank = 0;
   int nve   = dim == 2 ? 4 : 8;
   int nel = 0;
   int hash_l_n = dim == 2 ? findptsData2->local.hd.hash_n :
                  findptsData3->local.hd.hash_n;
   int hash_g_n = dim == 2 ? findptsData2->hash.hash_n :
                  findptsData3->hash.hash_n;

   if (type == 0 || type == 1)
   {
      nel = NE_split_total;
   }
   else if (type == 2)
   {
      nel = std::pow(hash_l_n, dim);
   }
   else if (type == 3)
   {
      nel = std::pow(hash_g_n, dim);
   }

   Vector hashlmin(dim), hashlmax(dim), dxyzl(dim);
   for (int d = 0; d < dim; d++)
   {
      hashlmin[d] = dim == 2 ? findptsData2->local.hd.bnd[d].min :
                    findptsData3->local.hd.bnd[d].min;
      hashlmax[d] =  dim == 2 ? findptsData2->local.hd.bnd[d].max :
                     findptsData3->local.hd.bnd[d].max;
      dxyzl[d] = (hashlmax[d]-hashlmin[d])/hash_l_n;
   }

   Vector hashgmin(dim), hashgmax(dim), dxyzg(dim);
   for (int d = 0; d < dim; d++)
   {
      hashgmin[d] = dim == 2 ? findptsData2->hash.bnd[d].min :
                    findptsData3->hash.bnd[d].min;
      hashgmax[d] =  dim == 2 ? findptsData2->hash.bnd[d].max :
                     findptsData3->hash.bnd[d].max;
      dxyzg[d] = (hashgmax[d]-hashgmin[d])/hash_g_n;
   }

   long long nel_l = nel;
   long long nel_glob_l = nel_l;
   int ne_glob;

   if (type == 0 || type == 1 || type == 2)
   {
#ifdef MFEM_USE_MPI
      MPI_Reduce(&nel_l, &nel_glob_l, 1, MPI_LONG_LONG, MPI_SUM, save_rank,
                 gsl_comm->c);
#endif
      ne_glob = int(nel_glob_l);
   }
   else
   {
      ne_glob = nel;
   }

   int nverts = nve*ne_glob;
   Vector o_xyz(dim*nel*nve);

   Mesh *meshbb = NULL;
   if (gsl_comm->id == save_rank)
   {
      meshbb = new Mesh(dim, nverts, ne_glob, 0, dim);
   }

   Array<int> hash_el_count(gsl_comm->np);
   hash_el_count[0] = nel;

   if (dim == 3)
   {
      for (int e = 0; e < nel; e++)
      {
         auto box = findptsData3->local.obb[e];
         if (type == 0)
         {
            Vector minn(dim), maxx(dim);
            for (int d = 0; d < dim; d++)
            {
               minn[d] = box.x[d].min;
               maxx[d] = box.x[d].max;
            }
            int c = 0;
            o_xyz(e*nve*dim + c++) = minn[0];
            o_xyz(e*nve*dim + c++) = minn[1];
            o_xyz(e*nve*dim + c++) = minn[2];

            o_xyz(e*nve*dim + c++) = maxx[0];
            o_xyz(e*nve*dim + c++) = minn[1];
            o_xyz(e*nve*dim + c++) = minn[2];

            o_xyz(e*nve*dim + c++) = maxx[0];
            o_xyz(e*nve*dim + c++) = maxx[1];
            o_xyz(e*nve*dim + c++) = minn[2];

            o_xyz(e*nve*dim + c++) = minn[0];
            o_xyz(e*nve*dim + c++) = maxx[1];
            o_xyz(e*nve*dim + c++) = minn[2];

            o_xyz(e*nve*dim + c++) = minn[0];
            o_xyz(e*nve*dim + c++) = minn[1];
            o_xyz(e*nve*dim + c++) = maxx[2];

            o_xyz(e*nve*dim + c++) = maxx[0];
            o_xyz(e*nve*dim + c++) = minn[1];
            o_xyz(e*nve*dim + c++) = maxx[2];

            o_xyz(e*nve*dim + c++) = maxx[0];
            o_xyz(e*nve*dim + c++) = maxx[1];
            o_xyz(e*nve*dim + c++) = maxx[2];

            o_xyz(e*nve*dim + c++) = minn[0];
            o_xyz(e*nve*dim + c++) = maxx[1];
            o_xyz(e*nve*dim + c++) = maxx[2];
         }
         else if (type == 1)
         {
            Vector center(dim), A(dim*dim);
            for (int d = 0; d < dim; d++)
            {
               center[d] = box.c0[d];
            }
            for (int d = 0; d < dim*dim; d++)
            {
               A[d] = box.A[d];
            }

            DenseMatrix Amat(A.GetData(), dim, dim);
            Amat.Transpose();
            Amat.Invert();

            Vector v1(dim);
            Vector temp;

            v1(0) = -1.0; v1(1) = -1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 0, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = -1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 3, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = 1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 6, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = 1.0; v1(2) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 9, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = -1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 0, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = -1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 3, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = 1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 6, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = 1.0; v1(2) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 9, dim);
            Amat.Mult(v1, temp);
            temp += center;
         } //type == 1
      } // e < nel
      if (type == 2)
      {
         int ec = 0;
         for (int l = 0; l < hash_l_n; l++)
         {
            for (int k = 0; k < hash_l_n; k++)
            {
               for (int j = 0; j < hash_l_n; j++)
               {
                  double x0 = hashlmin[0] + j*dxyzl[0];
                  double y0 = hashlmin[1] + k*dxyzl[1];
                  double z0 = hashlmin[2] + l*dxyzl[2];
                  o_xyz(ec*nve*dim + 0) = x0;
                  o_xyz(ec*nve*dim + 1) = y0;
                  o_xyz(ec*nve*dim + 2) = z0;

                  o_xyz(ec*nve*dim + 3) = x0+dxyzl[0];
                  o_xyz(ec*nve*dim + 4) = y0;
                  o_xyz(ec*nve*dim + 5) = z0;

                  o_xyz(ec*nve*dim + 6) = x0+dxyzl[0];
                  o_xyz(ec*nve*dim + 7) = y0+dxyzl[1];
                  o_xyz(ec*nve*dim + 8) = z0;

                  o_xyz(ec*nve*dim + 9) = x0;
                  o_xyz(ec*nve*dim + 10) = y0+dxyzl[1];
                  o_xyz(ec*nve*dim + 11) = z0;

                  o_xyz(ec*nve*dim + 12) = x0;
                  o_xyz(ec*nve*dim + 13) = y0;
                  o_xyz(ec*nve*dim + 14) = z0+dxyzl[2];

                  o_xyz(ec*nve*dim + 15) = x0+dxyzl[0];
                  o_xyz(ec*nve*dim + 16) = y0;
                  o_xyz(ec*nve*dim + 17) = z0+dxyzl[2];

                  o_xyz(ec*nve*dim + 18) = x0+dxyzl[0];
                  o_xyz(ec*nve*dim + 19) = y0+dxyzl[1];
                  o_xyz(ec*nve*dim + 20) = z0+dxyzl[2];

                  o_xyz(ec*nve*dim + 21) = x0;
                  o_xyz(ec*nve*dim + 22) = y0+dxyzl[1];
                  o_xyz(ec*nve*dim + 23) = z0+dxyzl[2];

                  ec++;
               }
            }
         }
      } //type == 2
      else if (type == 3)
      {
         int ec = 0;
         for (int l = 0; l < hash_g_n; l++)
         {
            for (int k = 0; k < hash_g_n; k++)
            {
               for (int j = 0; j < hash_g_n; j++)
               {
                  double x0 = hashgmin[0] + j*dxyzg[0];
                  double y0 = hashgmin[1] + k*dxyzg[1];
                  double z0 = hashgmin[2] + l*dxyzg[2];
                  o_xyz(ec*nve*dim + 0) = x0;
                  o_xyz(ec*nve*dim + 1) = y0;
                  o_xyz(ec*nve*dim + 2) = z0;

                  o_xyz(ec*nve*dim + 3) = x0+dxyzg[0];
                  o_xyz(ec*nve*dim + 4) = y0;
                  o_xyz(ec*nve*dim + 5) = z0;

                  o_xyz(ec*nve*dim + 6) = x0+dxyzg[0];
                  o_xyz(ec*nve*dim + 7) = y0+dxyzg[1];
                  o_xyz(ec*nve*dim + 8) = z0;

                  o_xyz(ec*nve*dim + 9) = x0;
                  o_xyz(ec*nve*dim + 10) = y0+dxyzg[1];
                  o_xyz(ec*nve*dim + 11) = z0;

                  o_xyz(ec*nve*dim + 12) = x0;
                  o_xyz(ec*nve*dim + 13) = y0;
                  o_xyz(ec*nve*dim + 14) = z0+dxyzg[2];

                  o_xyz(ec*nve*dim + 15) = x0+dxyzg[0];
                  o_xyz(ec*nve*dim + 16) = y0;
                  o_xyz(ec*nve*dim + 17) = z0+dxyzg[2];

                  o_xyz(ec*nve*dim + 18) = x0+dxyzg[0];
                  o_xyz(ec*nve*dim + 19) = y0+dxyzg[1];
                  o_xyz(ec*nve*dim + 20) = z0+dxyzg[2];

                  o_xyz(ec*nve*dim + 21) = x0;
                  o_xyz(ec*nve*dim + 22) = y0+dxyzg[1];
                  o_xyz(ec*nve*dim + 23) = z0+dxyzg[2];

                  ec++;
               }
            }
         }
      }
   }
   else // dim = 2
   {
      for (int e = 0; e < nel; e++)
      {
         auto box = findptsData2->local.obb[e];
         if (type == 0)
         {
            Vector minn(dim), maxx(dim);
            for (int d = 0; d < dim; d++)
            {
               minn[d] = box.x[d].min;
               maxx[d] = box.x[d].max;
            }
            o_xyz(e*nve*dim + 0) = minn[0];
            o_xyz(e*nve*dim + 1) = minn[1];

            o_xyz(e*nve*dim + 2) = maxx[0];
            o_xyz(e*nve*dim + 3) = minn[1];

            o_xyz(e*nve*dim + 4) = maxx[0];
            o_xyz(e*nve*dim + 5) = maxx[1];

            o_xyz(e*nve*dim + 6) = minn[0];
            o_xyz(e*nve*dim + 7) = maxx[1];
         }
         else if (type == 1)
         {
            Vector center(dim), A(dim*dim);
            for (int d = 0; d < dim; d++)
            {
               center[d] = box.c0[d];
            }
            for (int d = 0; d < dim*dim; d++)
            {
               A[d] = box.A[d];
            }

            DenseMatrix Amat(A.GetData(), dim, dim);
            Amat.Transpose();
            Amat.Invert();

            Vector v1(dim);
            Vector temp;

            v1(0) = -1.0; v1(1) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 0, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = -1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 2, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = 1.0; v1(1) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 4, dim);
            Amat.Mult(v1, temp);
            temp += center;

            v1(0) = -1.0; v1(1) = 1.0;
            temp.SetDataAndSize(o_xyz.GetData() + e*nve*dim + 6, dim);
            Amat.Mult(v1, temp);
            temp += center;
         } //type == 1
      } // e < nel
      if (type == 2)
      {
         int ec = 0;
         for (int k = 0; k < hash_l_n; k++)
         {
            for (int j = 0; j < hash_l_n; j++)
            {
               double x0 = hashlmin[0] + j*dxyzl[0];
               double y0 = hashlmin[1] + k*dxyzl[1];
               o_xyz(ec*nve*dim + 0) = x0;
               o_xyz(ec*nve*dim + 1) = y0;

               o_xyz(ec*nve*dim + 2) = x0+dxyzl[0];
               o_xyz(ec*nve*dim + 3) = y0;

               o_xyz(ec*nve*dim + 4) = x0+dxyzl[0];
               o_xyz(ec*nve*dim + 5) = y0+dxyzl[1];

               o_xyz(ec*nve*dim + 6) = x0;
               o_xyz(ec*nve*dim + 7) = y0+dxyzl[1];

               ec++;
            }
         }
      }
      else if (type == 3)
      {
         int ec = 0;
         for (int k = 0; k < hash_g_n; k++)
         {
            for (int j = 0; j < hash_g_n; j++)
            {
               double x0 = hashgmin[0] + j*dxyzg[0];
               double y0 = hashgmin[1] + k*dxyzg[1];
               o_xyz(ec*nve*dim + 0) = x0;
               o_xyz(ec*nve*dim + 1) = y0;

               o_xyz(ec*nve*dim + 2) = x0+dxyzg[0];
               o_xyz(ec*nve*dim + 3) = y0;

               o_xyz(ec*nve*dim + 4) = x0+dxyzg[0];
               o_xyz(ec*nve*dim + 5) = y0+dxyzg[1];

               o_xyz(ec*nve*dim + 6) = x0;
               o_xyz(ec*nve*dim + 7) = y0+dxyzg[1];

               ec++;
            }
         }
      }
   }

   int nsend = type == 3 ? 0 : nel*nve*dim;
   int nrecv = 0;
   MPI_Status status;
   int vidx = 0;
   int eidx = 0;
   int proc_hash_loc_idx = 0;
   if (gsl_comm->id == save_rank)
   {
      for (int p = 0; p < gsl_comm->np; p++)
      {
         if (p != save_rank)
         {
            MPI_Recv(&nrecv, 1, MPI_INT, p, 444, gsl_comm->c, &status);
            o_xyz.SetSize(nrecv);
            if (nrecv)
            {
               MPI_Recv(o_xyz.GetData(), nrecv, MPI_DOUBLE, p, 445, gsl_comm->c, &status);
            }
         }
         else
         {
            nrecv = type == 3 ? nel*nve*dim : nsend;
         }
         int nel_recv = nrecv/(dim*nve);

         // we keep track of how many hash cells are coming from each rank
         if (p != save_rank)
         {
            hash_el_count[p] = hash_el_count[p-1] + nel_recv;
         }
         for (int e = 0; e < nel_recv; e++)
         {
            for (int j = 0; j < nve; j++)
            {
               Vector ver(o_xyz.GetData() + e*nve*dim + j*dim, dim);
               meshbb->AddVertex(ver);
            }

            if (dim == 2)
            {
               const int inds[4] = {vidx++, vidx++, vidx++, vidx++};
               int attr = eidx+1;
               // for type == 2, we set element attribute based on the
               // proc from which the element must have come.
               if (type == 2)
               {
                  if (eidx >= hash_el_count[proc_hash_loc_idx])
                  {
                     proc_hash_loc_idx++;
                  }
                  attr = proc_hash_loc_idx+1;
               }
               else if (type == 3)
               {
                  attr = eidx % gsl_comm->np;
                  attr += 1;
               }
               meshbb->AddQuad(inds, attr);
               eidx++;
            }
            else
            {
               const int inds[8] = {vidx++, vidx++, vidx++, vidx++,
                                    vidx++, vidx++, vidx++, vidx++
                                   };
               meshbb->AddHex(inds, (eidx++)+1);
            }
         }
      }
      if (dim == 2)
      {
         meshbb->FinalizeQuadMesh(1, 1, true);
      }
      else
      {
         meshbb->FinalizeHexMesh(1, 1, true);
      }
   }
   else
   {
      MPI_Send(&nsend, 1, MPI_INT, save_rank, 444, gsl_comm->c);
      if (nsend)
      {
         MPI_Send(o_xyz.GetData(), nsend, MPI_DOUBLE, save_rank, 445, gsl_comm->c);
      }
   }

   return meshbb;
}

void FindPointsGSLIB::FindPoints(Mesh &m, const Vector &point_pos,
                                 int point_pos_ordering, const double bb_t,
                                 const double newt_tol, const int npt_max)
{
   if (!setupflag || (mesh != &m) )
   {
      Setup(m, bb_t, newt_tol, npt_max);
   }
   FindPoints(point_pos, point_pos_ordering);
}

void FindPointsGSLIB::Interpolate(const Vector &point_pos,
                                  const GridFunction &field_in, Vector &field_out,
                                  int point_pos_ordering)
{
   FindPoints(point_pos, point_pos_ordering);
   Interpolate(field_in, field_out);
}

void FindPointsGSLIB::Interpolate(Mesh &m, const Vector &point_pos,
                                  const GridFunction &field_in, Vector &field_out,
                                  int point_pos_ordering)
{
   FindPoints(m, point_pos, point_pos_ordering);
   Interpolate(field_in, field_out);
}

void FindPointsGSLIB::FreeData()
{
   if (!setupflag) { return; }
   if (dim == 2)
   {
      findpts_free_2((gslib::findpts_data_2 *)this->fdataD);
   }
   else
   {
      findpts_free_3((gslib::findpts_data_3 *)this->fdataD);
   }
   gsl_code.DeleteAll();
   gsl_proc.DeleteAll();
   gsl_elem.DeleteAll();
   gsl_mesh.Destroy();
   gsl_ref.Destroy();
   gsl_dist.Destroy();
   for (int i = 0; i < 4; i++)
   {
      if (mesh_split[i]) { delete mesh_split[i]; mesh_split[i] = NULL; }
      if (ir_split[i]) { delete ir_split[i]; ir_split[i] = NULL; }
      if (fes_rst_map[i]) { delete fes_rst_map[i]; fes_rst_map[i] = NULL; }
      if (gf_rst_map[i]) { delete gf_rst_map[i]; gf_rst_map[i] = NULL; }
   }
   if (fec_map_lin) { delete fec_map_lin; fec_map_lin = NULL; }
   setupflag = false;
   DEV.setup_device = false;
}

void FindPointsGSLIB::SetupSplitMeshes()
{
   fec_map_lin = new H1_FECollection(1, dim);
   if (mesh->Dimension() == 2)
   {
      int Nvert = 7;
      int NEsplit = 3;
      mesh_split[0] = new Mesh(2, Nvert, NEsplit, 0, 2);

      const double quad_v[7][2] =
      {
         {0, 0}, {0.5, 0}, {1, 0}, {0, 0.5},
         {1./3., 1./3.}, {0.5, 0.5}, {0, 1}
      };
      const int quad_e[3][4] =
      {
         {0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 5, 6}
      };

      for (int j = 0; j < Nvert; j++)
      {
         mesh_split[0]->AddVertex(quad_v[j]);
      }
      for (int j = 0; j < NEsplit; j++)
      {
         int attribute = j + 1;
         mesh_split[0]->AddQuad(quad_e[j], attribute);
      }
      mesh_split[0]->FinalizeQuadMesh(1, 1, true);

      fes_rst_map[0] = new FiniteElementSpace(mesh_split[0], fec_map_lin, dim);
      gf_rst_map[0] = new GridFunction(fes_rst_map[0]);
      gf_rst_map[0]->UseDevice(false);
      const int npt = gf_rst_map[0]->Size()/dim;
      for (int k = 0; k < dim; k++)
      {
         for (int j = 0; j < npt; j++)
         {
            (*gf_rst_map[0])(j+k*npt) = quad_v[j][k];
         }
      }

      mesh_split[1] = new Mesh(Mesh::MakeCartesian2D(1, 1,
                                                     Element::QUADRILATERAL));
   }
   else if (mesh->Dimension() == 3)
   {
      mesh_split[0] = new Mesh(Mesh::MakeCartesian3D(1, 1, 1,
                                                     Element::HEXAHEDRON));
      // Tetrahedron
      {
         int Nvert = 15;
         int NEsplit = 4;
         mesh_split[1] = new Mesh(3, Nvert, NEsplit, 0, 3);

         const double hex_v[15][3] =
         {
            {0, 0, 0.}, {1, 0., 0.}, {0., 1., 0.}, {0, 0., 1.},
            {0.5, 0., 0.}, {0.5, 0.5, 0.}, {0., 0.5, 0.},
            {0., 0., 0.5}, {0.5, 0., 0.5}, {0., 0.5, 0.5},
            {1./3., 0., 1./3.}, {1./3., 1./3., 1./3.}, {0, 1./3., 1./3.},
            {1./3., 1./3., 0}, {0.25, 0.25, 0.25}
         };
         const int hex_e[4][8] =
         {
            {7, 10, 4, 0, 12, 14, 13, 6},
            {10, 8, 1, 4, 14, 11, 5, 13},
            {14, 11, 5, 13, 12, 9, 2, 6},
            {7, 3, 8, 10, 12, 9, 11, 14}
         };

         for (int j = 0; j < Nvert; j++)
         {
            mesh_split[1]->AddVertex(hex_v[j]);
         }
         for (int j = 0; j < NEsplit; j++)
         {
            int attribute = j + 1;
            mesh_split[1]->AddHex(hex_e[j], attribute);
         }
         mesh_split[1]->FinalizeHexMesh(1, 1, true);

         fes_rst_map[1] = new FiniteElementSpace(mesh_split[1], fec_map_lin, dim);
         gf_rst_map[1] = new GridFunction(fes_rst_map[1]);
         gf_rst_map[1]->UseDevice(false);
         const int npt = gf_rst_map[1]->Size()/dim;
         for (int k = 0; k < dim; k++)
         {
            for (int j = 0; j < npt; j++)
            {
               (*gf_rst_map[1])(j+k*npt) = hex_v[j][k];
            }
         }
      }
      // Prism
      {
         int Nvert = 14;
         int NEsplit = 3;
         mesh_split[2] = new Mesh(3, Nvert, NEsplit, 0, 3);

         const double hex_v[14][3] =
         {
            {0, 0, 0}, {0.5, 0, 0}, {1, 0, 0}, {0, 0.5, 0},
            {1./3., 1./3., 0}, {0.5, 0.5, 0}, {0, 1, 0},
            {0, 0, 1}, {0.5, 0, 1}, {1, 0, 1}, {0, 0.5, 1},
            {1./3., 1./3., 1}, {0.5, 0.5, 1}, {0, 1, 1}
         };
         const int hex_e[3][8] =
         {
            {0, 1, 4, 3, 7, 8, 11, 10},
            {1, 2, 5, 4, 8, 9, 12, 11},
            {3, 4, 5, 6, 10, 11, 12, 13}
         };

         for (int j = 0; j < Nvert; j++)
         {
            mesh_split[2]->AddVertex(hex_v[j]);
         }
         for (int j = 0; j < NEsplit; j++)
         {
            int attribute = j + 1;
            mesh_split[2]->AddHex(hex_e[j], attribute);
         }
         mesh_split[2]->FinalizeHexMesh(1, 1, true);

         fes_rst_map[2] = new FiniteElementSpace(mesh_split[2], fec_map_lin, dim);
         gf_rst_map[2] = new GridFunction(fes_rst_map[2]);
         gf_rst_map[2]->UseDevice(false);
         const int npt = gf_rst_map[2]->Size()/dim;
         for (int k = 0; k < dim; k++)
         {
            for (int j = 0; j < npt; j++)
            {
               (*gf_rst_map[2])(j+k*npt) = hex_v[j][k];
            }
         }
      }
      // Pyramid
      {
         int Nvert = 23;
         int NEsplit = 8;
         mesh_split[3] = new Mesh(3, Nvert, NEsplit, 0, 3);

         const double hex_v[23][3] =
         {
            {0.0000, 0.0000, 0.0000}, {0.5000, 0.0000, 0.0000},
            {0.0000, 0.0000, 0.5000}, {0.3333, 0.0000, 0.3333},
            {0.0000, 0.5000, 0.0000}, {0.3333, 0.3333, 0.0000},
            {0.0000, 0.3333, 0.3333}, {0.2500, 0.2500, 0.2500},
            {1.0000, 0.0000, 0.0000}, {0.5000, 0.0000, 0.5000},
            {0.5000, 0.5000, 0.0000}, {0.3333, 0.3333, 0.3333},
            {0.0000, 1.0000, 0.0000}, {0.0000, 0.5000, 0.5000},
            {0.0000, 0.0000, 1.0000}, {1.0000, 0.5000, 0.0000},
            {0.6667, 0.3333, 0.3333}, {0.6667, 0.6667, 0.0000},
            {0.5000, 0.5000, 0.2500}, {1.0000, 1.0000, 0.0000},
            {0.5000, 0.5000, 0.5000}, {0.5000, 1.0000, 0.0000},
            {0.3333, 0.6667, 0.3333}
         };
         const int hex_e[8][8] =
         {
            {2, 3, 1, 0, 6, 7, 5, 4}, {3, 9, 8, 1, 7, 11, 10, 5},
            {7, 11, 10, 5, 6, 13, 12, 4}, {2, 14, 9, 3, 6, 13, 11, 7},
            {9, 16, 15, 8, 11, 18, 17, 10}, {16, 20, 19, 15, 18, 22, 21, 17},
            {18, 22, 21, 17, 11, 13, 12, 10}, {9, 14, 20, 16, 11, 13, 22, 18}
         };

         for (int j = 0; j < Nvert; j++)
         {
            mesh_split[3]->AddVertex(hex_v[j]);
         }
         for (int j = 0; j < NEsplit; j++)
         {
            int attribute = j + 1;
            mesh_split[3]->AddHex(hex_e[j], attribute);
         }
         mesh_split[3]->FinalizeHexMesh(1, 1, true);

         fes_rst_map[3] = new FiniteElementSpace(mesh_split[3], fec_map_lin, dim);
         gf_rst_map[3] = new GridFunction(fes_rst_map[3]);
         gf_rst_map[3]->UseDevice(false);
         const int npt = gf_rst_map[3]->Size()/dim;
         for (int k = 0; k < dim; k++)
         {
            for (int j = 0; j < npt; j++)
            {
               (*gf_rst_map[3])(j+k*npt) = hex_v[j][k];
            }
         }
      }
   }

   NE_split_total = 0;
   split_element_map.SetSize(0);
   split_element_index.SetSize(0);
   int NEsplit = 0;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const Geometry::Type gt   = mesh->GetElement(e)->GetGeometryType();
      if (gt == Geometry::TRIANGLE || gt == Geometry::PRISM)
      {
         NEsplit = 3;
      }
      else if (gt == Geometry::TETRAHEDRON)
      {
         NEsplit = 4;
      }
      else if (gt == Geometry::PYRAMID)
      {
         NEsplit = 8;
      }
      else if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
      {
         NEsplit = 1;
      }
      else
      {
         MFEM_ABORT("Unsupported geometry type.");
      }
      NE_split_total += NEsplit;
      for (int i = 0; i < NEsplit; i++)
      {
         split_element_map.Append(e);
         split_element_index.Append(i);
      }
   }
}

void FindPointsGSLIB::SetupIntegrationRuleForSplitMesh(Mesh *meshin,
                                                       IntegrationRule *irule,
                                                       int order)
{
   H1_FECollection fec(order, dim);
   FiniteElementSpace nodal_fes(meshin, &fec, dim);
   meshin->SetNodalFESpace(&nodal_fes);
   const int NEsplit = meshin->GetNE();

   const int dof_cnt = nodal_fes.GetFE(0)->GetDof(),
             pts_cnt = NEsplit * dof_cnt;
   Vector irlist(dim * pts_cnt);

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(nodal_fes.GetFE(0));
   MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
   const Array<int> &dof_map = tbe->GetDofMap();

   DenseMatrix pos(dof_cnt, dim);
   Vector posV(pos.Data(), dof_cnt * dim);
   Array<int> xdofs(dof_cnt * dim);

   // Create an IntegrationRule on the nodes of the reference submesh.
   MFEM_ASSERT(irule->GetNPoints() == pts_cnt, "IntegrationRule does not have"
               "the correct number of points.");
   GridFunction *nodesplit = meshin->GetNodes();
   int pt_id = 0;
   for (int i = 0; i < NEsplit; i++)
   {
      nodal_fes.GetElementVDofs(i, xdofs);
      nodesplit->GetSubVector(xdofs, posV);
      for (int j = 0; j < dof_cnt; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            irlist(pts_cnt * d + pt_id) = pos(dof_map[j], d);
         }
         irule->IntPoint(pt_id).x = irlist(pt_id);
         irule->IntPoint(pt_id).y = irlist(pts_cnt + pt_id);
         if (dim == 3)
         {
            irule->IntPoint(pt_id).z = irlist(2*pts_cnt + pt_id);
         }
         pt_id++;
      }
   }
}

void FindPointsGSLIB::GetNodalValues(const GridFunction *gf_in,
                                     Vector &node_vals)
{
   const GridFunction *nodes     = gf_in;
   const FiniteElementSpace *fes = nodes->FESpace();
   const int NE                  = mesh->GetNE();
   const int vdim                = fes->GetVDim();

   IntegrationRule *ir_split_temp = NULL;

   const int maxOrder = fes->GetMaxElementOrder();
   const int dof_1D =  maxOrder+1;
   const int pts_el = std::pow(dof_1D, dim);
   const int pts_cnt = NE_split_total * pts_el;
   node_vals.SetSize(vdim * pts_cnt);

   if (node_vals.UseDevice())
   {
      node_vals.HostWrite();
   }

   int gsl_mesh_pt_index = 0;

   for (int e = 0; e < NE; e++)
   {
      const FiniteElement *fe   = fes->GetFE(e);
      const Geometry::Type gt   = fe->GetGeomType();
      bool el_to_split = true;
      if (gt == Geometry::TRIANGLE)
      {
         ir_split_temp = ir_split[0];
      }
      else if (gt == Geometry::TETRAHEDRON)
      {
         ir_split_temp = ir_split[1];
      }
      else if (gt == Geometry::PRISM)
      {
         ir_split_temp = ir_split[2];
      }
      else if (gt == Geometry::PYRAMID)
      {
         ir_split_temp = ir_split[3];
      }
      else if (gt == Geometry::SQUARE)
      {
         ir_split_temp = ir_split[1];
         el_to_split = gf_in->FESpace()->IsVariableOrder();
      }
      else if (gt == Geometry::CUBE)
      {
         ir_split_temp = ir_split[0];
         el_to_split = gf_in->FESpace()->IsVariableOrder();
      }
      else
      {
         MFEM_ABORT("Unsupported geometry type.");
      }

      if (el_to_split) // Triangle/Tet/Prism or Quads/Hex but variable order
      {
         // Fill gsl_mesh with location of split points.
         Vector locval(vdim);
         for (int i = 0; i < ir_split_temp->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir_split_temp->IntPoint(i);
            nodes->GetVectorValue(e, ip, locval);
            for (int d = 0; d < vdim; d++)
            {
               node_vals(pts_cnt*d + gsl_mesh_pt_index) = locval(d);
            }
            gsl_mesh_pt_index++;
         }
      }
      else // Quad/Hex and constant polynomial order
      {
         const int dof_cnt_split = fe->GetDof();

         const TensorBasisElement *tbe =
            dynamic_cast<const TensorBasisElement *>(fes->GetFE(e));
         MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
         Array<int> dof_map(dof_cnt_split);
         const Array<int> &dm = tbe->GetDofMap();
         if (dm.Size() > 0) { dof_map = dm; }
         else { for (int i = 0; i < dof_cnt_split; i++) { dof_map[i] = i; } }

         DenseMatrix pos(dof_cnt_split, vdim);
         Vector posV(pos.Data(), dof_cnt_split * vdim);
         Array<int> xdofs(dof_cnt_split * vdim);

         fes->GetElementVDofs(e, xdofs);
         nodes->GetSubVector(xdofs, posV);
         for (int j = 0; j < dof_cnt_split; j++)
         {
            for (int d = 0; d < vdim; d++)
            {
               node_vals(pts_cnt * d + gsl_mesh_pt_index) = pos(dof_map[j], d);
            }
            gsl_mesh_pt_index++;
         }
      }
   }
}


void FindPointsGSLIB::MapRefPosAndElemIndices()
{
   gsl_mfem_ref.SetSize(points_cnt*dim);
   gsl_mfem_elem.SetSize(points_cnt);
   gsl_mfem_ref = gsl_ref.HostRead();
   gsl_mfem_elem = gsl_elem;

   gsl_mfem_ref += 1.;  // map  [-1, 1] to [0, 2] to [0, 1]
   gsl_mfem_ref *= 0.5;

   int nptorig = points_cnt,
       npt = points_cnt;

   // tolerance for point to be marked as on element edge/face
   double btol = 1e-12;

   GridFunction *gf_rst_map_temp = NULL;
   int nptsend = 0;

   for (int index = 0; index < npt; index++)
   {
      if (gsl_code[index] != 2 && gsl_proc[index] != gsl_comm->id)
      {
         nptsend +=1;
      }
   }

   // Pack data to send via crystal router
   struct gslib::array *outpt = new gslib::array;
   struct out_pt { double r[3]; uint index, el, proc, code; };
   struct out_pt *pt;
   array_init(struct out_pt, outpt, nptsend);
   outpt->n=nptsend;
   pt = (struct out_pt *)outpt->ptr;
   for (int index = 0; index < npt; index++)
   {
      if (gsl_code[index] == 2 || gsl_proc[index] == gsl_comm->id)
      {
         continue;
      }
      for (int d = 0; d < dim; ++d)
      {
         pt->r[d]= gsl_mfem_ref(index*dim + d);
      }
      pt->index = index;
      pt->proc  = gsl_proc[index];
      pt->el    = gsl_elem[index];
      pt->code  = gsl_code[index];
      ++pt;
   }

   // Transfer data to target MPI ranks
   sarray_transfer(struct out_pt, outpt, proc, 1, cr);

   // Map received points
   npt = outpt->n;
   pt = (struct out_pt *)outpt->ptr;
   for (int index = 0; index < npt; index++)
   {
      IntegrationPoint ip;
      ip.Set3(&pt->r[0]);
      const int elem = pt->el;
      const int mesh_elem = split_element_map[elem];
      const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(mesh_elem);

      const Geometry::Type gt = fe->GetGeomType();
      pt->el = mesh_elem;

      if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
      {
         // check if it is on element boundary
         pt->code = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
         ++pt;
         continue;
      }
      else if (gt == Geometry::TRIANGLE)
      {
         gf_rst_map_temp = gf_rst_map[0];
      }
      else if (gt == Geometry::TETRAHEDRON)
      {
         gf_rst_map_temp = gf_rst_map[1];
      }
      else if (gt == Geometry::PRISM)
      {
         gf_rst_map_temp = gf_rst_map[2];
      }
      else if (gt == Geometry::PYRAMID)
      {
         gf_rst_map_temp = gf_rst_map[3];
      }

      int local_elem = split_element_index[elem];
      Vector mfem_ref(dim);
      // map to rst of macro element
      gf_rst_map_temp->GetVectorValue(local_elem, ip, mfem_ref);

      for (int d = 0; d < dim; d++)
      {
         pt->r[d] = mfem_ref(d);
      }

      // check if point is on element boundary
      ip.Set3(&pt->r[0]);
      pt->code = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
      ++pt;
   }

   // Transfer data back to source MPI rank
   sarray_transfer(struct out_pt, outpt, proc, 1, cr);
   npt = outpt->n;

   // First copy mapped information for points on other procs
   pt = (struct out_pt *)outpt->ptr;
   for (int index = 0; index < npt; index++)
   {
      gsl_mfem_elem[pt->index] = pt->el;
      for (int d = 0; d < dim; d++)
      {
         gsl_mfem_ref(d + pt->index*dim) = pt->r[d];
      }
      gsl_code[pt->index] = pt->code;
      ++pt;
   }
   array_free(outpt);
   delete outpt;

   // Now map information for points on the same proc
   for (int index = 0; index < nptorig; index++)
   {
      if (gsl_code[index] != 2 && gsl_proc[index] == gsl_comm->id)
      {

         IntegrationPoint ip;
         Vector mfem_ref(gsl_mfem_ref.GetData()+index*dim, dim);
         ip.Set2(mfem_ref.GetData());
         if (dim == 3) { ip.z = mfem_ref(2); }

         const int elem = gsl_elem[index];
         const int mesh_elem = split_element_map[elem];
         const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(mesh_elem);
         const Geometry::Type gt = fe->GetGeomType();
         gsl_mfem_elem[index] = mesh_elem;
         if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
         {
            gsl_code[index] = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
            continue;
         }
         else if (gt == Geometry::TRIANGLE)
         {
            gf_rst_map_temp = gf_rst_map[0];
         }
         else if (gt == Geometry::TETRAHEDRON)
         {
            gf_rst_map_temp = gf_rst_map[1];
         }
         else if (gt == Geometry::PRISM)
         {
            gf_rst_map_temp = gf_rst_map[2];
         }
         else if (gt == Geometry::PYRAMID)
         {
            gf_rst_map_temp = gf_rst_map[3];
         }

         int local_elem = split_element_index[elem];
         gf_rst_map_temp->GetVectorValue(local_elem, ip, mfem_ref);

         // Check if the point is on element boundary
         ip.Set2(mfem_ref.GetData());
         if (dim == 3) { ip.z = mfem_ref(2); }
         gsl_code[index]  = Geometry::CheckPoint(gt, ip, -btol) ? 0 : 1;
      }
   }
}

struct evalSrcPt_t
{
   double r[3];
   unsigned int index, proc, el;
};

struct evalOutPt_t
{
   double out;
   unsigned int index, proc;
};

void FindPointsGSLIB::InterpolateOnDevice(const Vector &field_in_evec,
                                          Vector &field_out,
                                          const int nel,
                                          const int ncomp,
                                          const int dof1Dsol,
                                          const int ordering)
{
   field_out.SetSize(points_cnt*ncomp);
   field_out.UseDevice(true);
   field_out = default_interp_value;

   DEV.dof1d_sol =  dof1Dsol;
   DEV.gll1d_sol.UseDevice(true);  DEV.gll1d_sol.SetSize(dof1Dsol);
   DEV.lagcoeff_sol.UseDevice(true); DEV.lagcoeff_sol.SetSize(dof1Dsol);
   if (DEV.dof1d_sol != DEV.dof1d)
   {
      gslib::lobatto_nodes(DEV.gll1d_sol.HostWrite(), dof1Dsol);
      gslib::gll_lag_setup(DEV.lagcoeff_sol.HostWrite(), dof1Dsol);
   }
   else
   {
      DEV.gll1d_sol = DEV.gll1d.HostRead();
      DEV.lagcoeff_sol = DEV.lagcoeff.HostRead();
   }

   field_out.HostReadWrite(); //Reads in default value from device

   struct gslib::array src, outpt;
   int nlocal = 0;
   /* weed out unfound points, send out */
   Array<int> gsl_elem_temp;
   Vector gsl_ref_temp;
   Array<int> index_temp;
   {
      int index;
      const unsigned int *code = gsl_code.GetData(), *proc = gsl_proc.GetData(),
                          *el   = gsl_elem.GetData();
      const double *r = gsl_ref.GetData();

      int numSend = 0;

      for (index = 0; index < points_cnt; ++index)
      {
         numSend += (gsl_code[index] != CODE_NOT_FOUND &&
                     gsl_proc[index] != gsl_comm->id);
         nlocal += (gsl_code[index] != CODE_NOT_FOUND &&
                    gsl_proc[index] == gsl_comm->id);
      }

      gsl_elem_temp.SetSize(nlocal);
      gsl_elem_temp.HostWrite();

      gsl_ref_temp.SetSize(nlocal*dim);
      gsl_ref_temp.UseDevice(true);
      gsl_ref_temp.HostWrite();

      index_temp.SetSize(nlocal);

      evalSrcPt_t *pt;
      array_init(evalSrcPt_t, &src, numSend);
      pt = (evalSrcPt_t *)src.ptr;

      int ctr = 0;
      for (index = 0; index < points_cnt; ++index)
      {
         if (*code != CODE_NOT_FOUND && *proc != gsl_comm->id)
         {
            for (int d = 0; d < dim; ++d)
            {
               pt->r[d] = r[d];
            }
            pt->index = index;
            pt->proc = *proc;
            pt->el = *el;
            ++pt;
         }
         else if (*code != CODE_NOT_FOUND && *proc == gsl_comm->id)
         {
            gsl_elem_temp[ctr] = *el;
            for (int d = 0; d < dim; ++d)
            {
               gsl_ref_temp(dim*ctr+d) = r[d];
            }
            index_temp[ctr] = index;

            ctr++;
         }
         r += dim;
         code++;
         proc++;
         el++;
      }

      src.n = pt - (evalSrcPt_t *)src.ptr;
      sarray_transfer(evalSrcPt_t, &src, proc, 1, cr);
   }

   //evaluate points that are already local
   {
      Vector interp_vals(nlocal*ncomp);
      interp_vals.UseDevice(true);

      if (dim == 2)
      {
         InterpolateLocal2(field_in_evec,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals,
                           nlocal, ncomp,
                           nel, dof1Dsol);
      }
      else
      {
         InterpolateLocal3(field_in_evec,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals,
                           nlocal, ncomp,
                           nel, dof1Dsol);

      }
      MPI_Barrier(gsl_comm->c);

      interp_vals.HostRead();

      // now put these in correct positions
      int interp_Offset = interp_vals.Size()/ncomp;
      for (int i = 0; i < ncomp; i++)
      {
         for (int j = 0; j < nlocal; j++)
         {
            int pt_index = index_temp[j];
            int idx = ordering == Ordering::byNODES ?
                      pt_index + i*points_cnt :
                      pt_index*ncomp + i;
            field_out(idx) = AsConst(interp_vals)(j + interp_Offset*i);
         }
      }
   }
   MPI_Barrier(gsl_comm->c);

   if (gsl_comm->np == 1)
   {
      array_free(&src);
      return;
   }

   // evaluate points locally
   {
      int n = src.n;
      const evalSrcPt_t *spt;
      evalOutPt_t *opt;
      spt = (evalSrcPt_t *)src.ptr;

      // Copy to host vector
      gsl_elem_temp.SetSize(n);
      gsl_elem_temp.HostWrite();

      gsl_ref_temp.SetSize(n*dim);
      gsl_ref_temp.HostWrite();

      spt = (evalSrcPt_t *)src.ptr;
      //   opt = (evalOutPt_t *)outpt.ptr;
      for (int i = 0; i < n; i++, ++spt)
      {
         gsl_elem_temp[i] = spt->el;
         for (int d = 0; d < dim; d++)
         {
            gsl_ref_temp(i*dim + d) = spt->r[d];
         }
      }

      Vector interp_vals(n*ncomp);
      interp_vals.UseDevice(true);
      if (dim == 2)
      {
         InterpolateLocal2(field_in_evec,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals, n, ncomp,
                           nel, dof1Dsol);
      }
      else
      {
         InterpolateLocal3(field_in_evec,
                           gsl_elem_temp,
                           gsl_ref_temp,
                           interp_vals, n, ncomp,
                           nel, dof1Dsol);
      }
      MPI_Barrier(gsl_comm->c);
      interp_vals.HostRead();

      // Now the interpolated values need to be sent back component wise
      int Offset = interp_vals.Size()/ncomp;
      for (int i = 0; i < ncomp; i++)
      {
         spt = (evalSrcPt_t *)src.ptr;
         array_init(evalOutPt_t, &outpt, n);
         outpt.n = n;
         opt = (evalOutPt_t *)outpt.ptr;

         for (int j = 0; j < n; j++)
         {
            opt->index = spt->index;
            opt->proc = spt->proc;
            opt->out = AsConst(interp_vals)(j + Offset*i);
            spt++;
            opt++;
         }

         sarray_transfer(struct evalOutPt_t, &outpt, proc, 1, cr);

         opt = (evalOutPt_t *)outpt.ptr;
         for (int index = 0; index < outpt.n; index++)
         {
            int idx = ordering == Ordering::byNODES ?
                      opt->index + i*points_cnt :
                      opt->index*ncomp + i;
            field_out(idx) = opt->out;
            ++opt;
         }
         array_free(&outpt);
      }
      array_free(&src);
   }
   //finished evaluating points received from other processors.
}

void FindPointsGSLIB::Interpolate(const GridFunction &field_in,
                                  Vector &field_out)
{
   const int  gf_order   = field_in.FESpace()->GetMaxElementOrder(),
              mesh_order = mesh->GetNodalFESpace()->GetMaxElementOrder();

   const FiniteElementCollection *fec_in =  field_in.FESpace()->FEColl();
   const H1_FECollection *fec_h1 = dynamic_cast<const H1_FECollection *>(fec_in);
   const L2_FECollection *fec_l2 = dynamic_cast<const L2_FECollection *>(fec_in);

   setupSW.Clear();
   setupSW.Start();
   if (Device::IsEnabled() && field_in.UseDevice() &&
       !field_in.FESpace()->IsVariableOrder() &&
       mesh->GetNumGeometries(dim) == 1 && mesh->GetNE() > 0 &&
       (mesh->GetElementType(0) == Element::QUADRILATERAL ||
        mesh->GetElementType(0) == Element::HEXAHEDRON))
   {
      MFEM_VERIFY(fec_h1,"Only h1 functions supported on device right now.");
      MFEM_VERIFY(fec_h1->GetBasisType() == BasisType::GaussLobatto,
                  "basis not supported");
      Vector node_vals;
      node_vals.UseDevice(true);
      const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *R = field_in.FESpace()->GetElementRestriction(ordering);
      node_vals.SetSize(R->Height(), Device::GetMemoryType());
      node_vals.UseDevice(true);
      R->Mult(field_in, node_vals);

      const int ncomp  = field_in.FESpace()->GetVDim();
      const int maxOrder = field_in.FESpace()->GetMaxElementOrder();

      // At this point make sure FindPoints was called with device mode?
      // Otherwise copy necessary data?
      InterpolateOnDevice(node_vals, field_out, NE_split_total, ncomp,
                          maxOrder+1, field_in.FESpace()->GetOrdering());

      setupSW.Stop();
      interpolate_h1_time = setupSW.RealTime();
      return;
   }


   if (fec_h1 && gf_order == mesh_order &&
       fec_h1->GetBasisType() == BasisType::GaussLobatto &&
       field_in.FESpace()->IsVariableOrder() ==
       mesh->GetNodalFESpace()->IsVariableOrder())
   {
      InterpolateH1(field_in, field_out);
      setupSW.Stop();
      interpolate_h1_time = setupSW.RealTime();
      return;
   }
   else
   {
      InterpolateGeneral(field_in, field_out);
      setupSW.Stop();
      interpolate_general_time = setupSW.RealTime();
      if (!fec_l2 || avgtype == AvgType::NONE) { return; }
   }

   // For points on element borders, project the L2 GridFunction to H1 and
   // re-interpolate.
   setupSW.Clear();
   setupSW.Start();
   if (fec_l2)
   {
      Array<int> indl2;
      for (int i = 0; i < points_cnt; i++)
      {
         if (gsl_code[i] == 1) { indl2.Append(i); }
      }
      int borderPts = indl2.Size();
#ifdef MFEM_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &borderPts, 1, MPI_INT, MPI_SUM, gsl_comm->c);
#endif
      if (borderPts == 0) { return; } // no points on element borders

      Vector field_out_l2(field_out.Size());
      VectorGridFunctionCoefficient field_in_dg(&field_in);
      int gf_order_h1 = std::max(gf_order, 1); // H1 should be at least order 1
      H1_FECollection fec(gf_order_h1, dim);
      const int ncomp = field_in.FESpace()->GetVDim();
      FiniteElementSpace fes(mesh, &fec, ncomp);
      GridFunction field_in_h1(&fes);

      if (avgtype == AvgType::ARITHMETIC)
      {
         field_in_h1.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
      }
      else if (avgtype == AvgType::HARMONIC)
      {
         field_in_h1.ProjectDiscCoefficient(field_in_dg, GridFunction::HARMONIC);
      }
      else
      {
         MFEM_ABORT("Invalid averaging type.");
      }

      if (gf_order_h1 == mesh_order) // basis is GaussLobatto by default
      {
         InterpolateH1(field_in_h1, field_out_l2);
      }
      else
      {
         InterpolateGeneral(field_in_h1, field_out_l2);
      }

      // Copy interpolated values for the points on element border
      for (int j = 0; j < ncomp; j++)
      {
         for (int i = 0; i < indl2.Size(); i++)
         {
            int idx = field_in.FESpace()->GetOrdering() == Ordering::byNODES ?
                      indl2[i] + j*points_cnt:
                      indl2[i]*ncomp + j;
            field_out(idx) = field_out_l2(idx);
         }
      }
   }
   setupSW.Stop();
   interpolate_l2_pass2_time = setupSW.RealTime();
}

void FindPointsGSLIB::InterpolateH1(const GridFunction &field_in,
                                    Vector &field_out)
{
   FiniteElementSpace ind_fes(mesh, field_in.FESpace()->FEColl());
   if (field_in.FESpace()->IsVariableOrder())
   {
      for (int e = 0; e < ind_fes.GetMesh()->GetNE(); e++)
      {
         ind_fes.SetElementOrder(e, field_in.FESpace()->GetElementOrder(e));
      }
      ind_fes.Update(false);
   }
   GridFunction field_in_scalar(&ind_fes);
   Vector node_vals;

   const int ncomp      = field_in.FESpace()->GetVDim(),
             points_fld = field_in.Size() / ncomp;
   MFEM_VERIFY(points_cnt == gsl_code.Size(),
               "FindPointsGSLIB::InterpolateH1: Inconsistent size of gsl_code");

   field_out.SetSize(points_cnt*ncomp);
   field_out = default_interp_value;

   for (int i = 0; i < ncomp; i++)
   {
      const int dataptrin  = i*points_fld,
                dataptrout = i*points_cnt;
      if (field_in.FESpace()->GetOrdering() == Ordering::byNODES)
      {
         field_in_scalar.NewDataAndSize(field_in.GetData()+dataptrin, points_fld);
      }
      else
      {
         for (int j = 0; j < points_fld; j++)
         {
            field_in_scalar(j) = field_in(i + j*ncomp);
         }
      }
      GetNodalValues(&field_in_scalar, node_vals);

      if (dim==2)
      {
         findpts_eval_2(field_out.GetData()+dataptrout, sizeof(double),
                        gsl_code.GetData(),       sizeof(unsigned int),
                        gsl_proc.GetData(),    sizeof(unsigned int),
                        gsl_elem.GetData(),    sizeof(unsigned int),
                        gsl_ref.GetData(),     sizeof(double) * dim,
                        points_cnt, node_vals.GetData(),
                        (gslib::findpts_data_2 *)this->fdataD);
      }
      else
      {
         findpts_eval_3(field_out.GetData()+dataptrout, sizeof(double),
                        gsl_code.GetData(),       sizeof(unsigned int),
                        gsl_proc.GetData(),    sizeof(unsigned int),
                        gsl_elem.GetData(),    sizeof(unsigned int),
                        gsl_ref.GetData(),     sizeof(double) * dim,
                        points_cnt, node_vals.GetData(),
                        (gslib::findpts_data_3 *)this->fdataD);
      }
   }
   if (field_in.FESpace()->GetOrdering() == Ordering::byVDIM)
   {
      Vector field_out_temp = field_out;
      for (int i = 0; i < ncomp; i++)
      {
         for (int j = 0; j < points_cnt; j++)
         {
            field_out(i + j*ncomp) = field_out_temp(j + i*points_cnt);
         }
      }
   }
}

void FindPointsGSLIB::InterpolateGeneral(const GridFunction &field_in,
                                         Vector &field_out)
{
   int ncomp   = field_in.VectorDim(),
       nptorig = points_cnt,
       npt     = points_cnt;

   field_out.SetSize(points_cnt*ncomp);
   field_out = default_interp_value;

   if (gsl_comm->np == 1) // serial
   {
      for (int index = 0; index < npt; index++)
      {
         if (gsl_code[index] == 2) { continue; }
         IntegrationPoint ip;
         ip.Set2(gsl_mfem_ref.GetData()+index*dim);
         if (dim == 3) { ip.z = gsl_mfem_ref(index*dim + 2); }
         Vector localval(ncomp);
         field_in.GetVectorValue(gsl_mfem_elem[index], ip, localval);
         if (field_in.FESpace()->GetOrdering() == Ordering::byNODES)
         {
            for (int i = 0; i < ncomp; i++)
            {
               field_out(index + i*npt) = localval(i);
            }
         }
         else //byVDIM
         {
            for (int i = 0; i < ncomp; i++)
            {
               field_out(index*ncomp + i) = localval(i);
            }
         }
      }
   }
   else // parallel
   {
      // Determine number of points to be sent
      int nptsend = 0;
      for (int index = 0; index < npt; index++)
      {
         if (gsl_code[index] != 2) { nptsend +=1; }
      }

      // Pack data to send via crystal router
      struct gslib::array *outpt = new gslib::array;
      struct out_pt { double r[3], ival; uint index, el, proc; };
      struct out_pt *pt;
      array_init(struct out_pt, outpt, nptsend);
      outpt->n=nptsend;
      pt = (struct out_pt *)outpt->ptr;
      for (int index = 0; index < npt; index++)
      {
         if (gsl_code[index] == 2) { continue; }
         for (int d = 0; d < dim; ++d) { pt->r[d]= gsl_mfem_ref(index*dim + d); }
         pt->index = index;
         pt->proc  = gsl_proc[index];
         pt->el    = gsl_mfem_elem[index];
         ++pt;
      }

      // Transfer data to target MPI ranks
      sarray_transfer(struct out_pt, outpt, proc, 1, cr);

      if (ncomp == 1)
      {
         // Interpolate the grid function
         npt = outpt->n;
         pt = (struct out_pt *)outpt->ptr;
         for (int index = 0; index < npt; index++)
         {
            IntegrationPoint ip;
            ip.Set3(&pt->r[0]);
            pt->ival = field_in.GetValue(pt->el, ip, 1);
            ++pt;
         }

         // Transfer data back to source MPI rank
         sarray_transfer(struct out_pt, outpt, proc, 1, cr);
         npt = outpt->n;
         pt = (struct out_pt *)outpt->ptr;
         for (int index = 0; index < npt; index++)
         {
            field_out(pt->index) = pt->ival;
            ++pt;
         }
         array_free(outpt);
         delete outpt;
      }
      else // ncomp > 1
      {
         // Interpolate data and store in a Vector
         npt = outpt->n;
         pt = (struct out_pt *)outpt->ptr;
         Vector vec_int_vals(npt*ncomp);
         for (int index = 0; index < npt; index++)
         {
            IntegrationPoint ip;
            ip.Set3(&pt->r[0]);
            Vector localval(vec_int_vals.GetData()+index*ncomp, ncomp);
            field_in.GetVectorValue(pt->el, ip, localval);
            ++pt;
         }

         // Save index and proc data in a struct
         struct gslib::array *savpt = new gslib::array;
         struct sav_pt { uint index, proc; };
         struct sav_pt *spt;
         array_init(struct sav_pt, savpt, npt);
         savpt->n=npt;
         spt = (struct sav_pt *)savpt->ptr;
         pt  = (struct out_pt *)outpt->ptr;
         for (int index = 0; index < npt; index++)
         {
            spt->index = pt->index;
            spt->proc  = pt->proc;
            ++pt; ++spt;
         }

         array_free(outpt);
         delete outpt;

         // Copy data from save struct to send struct and send component wise
         struct gslib::array *sendpt = new gslib::array;
         struct send_pt { double ival; uint index, proc; };
         struct send_pt *sdpt;
         for (int j = 0; j < ncomp; j++)
         {
            array_init(struct send_pt, sendpt, npt);
            sendpt->n=npt;
            spt  = (struct sav_pt *)savpt->ptr;
            sdpt = (struct send_pt *)sendpt->ptr;
            for (int index = 0; index < npt; index++)
            {
               sdpt->index = spt->index;
               sdpt->proc  = spt->proc;
               sdpt->ival  = vec_int_vals(j + index*ncomp);
               ++sdpt; ++spt;
            }

            sarray_transfer(struct send_pt, sendpt, proc, 1, cr);
            sdpt = (struct send_pt *)sendpt->ptr;
            for (int index = 0; index < static_cast<int>(sendpt->n); index++)
            {
               int idx = field_in.FESpace()->GetOrdering() == Ordering::byNODES ?
                         sdpt->index + j*nptorig :
                         sdpt->index*ncomp + j;
               field_out(idx) = sdpt->ival;
               ++sdpt;
            }
            array_free(sendpt);
         }
         array_free(savpt);
         delete sendpt;
         delete savpt;
      } // ncomp > 1
   } // parallel
}

void OversetFindPointsGSLIB::Setup(Mesh &m, const int meshid,
                                   GridFunction *gfmax,
                                   const double bb_t, const double newt_tol,
                                   const int npt_max)
{
   MFEM_VERIFY(m.GetNodes() != NULL, "Mesh nodes are required.");
   const int meshOrder = m.GetNodes()->FESpace()->GetMaxElementOrder();

   // FreeData if OversetFindPointsGSLIB::Setup has been called already
   if (setupflag) { FreeData(); }

   mesh = &m;
   dim  = mesh->Dimension();
   const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(0);
   unsigned dof1D = fe->GetOrder() + 1;

   SetupSplitMeshes();
   if (dim == 2)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(3*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);

      if (ir_split[1]) { delete ir_split[1]; ir_split[1] = NULL; }
      ir_split[1] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[1], ir_split[1], meshOrder);
   }
   else if (dim == 3)
   {
      if (ir_split[0]) { delete ir_split[0]; ir_split[0] = NULL; }
      ir_split[0] = new IntegrationRule(pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[0], ir_split[0], meshOrder);

      if (ir_split[1]) { delete ir_split[1]; ir_split[1] = NULL; }
      ir_split[1] = new IntegrationRule(4*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[1], ir_split[1], meshOrder);

      if (ir_split[2]) { delete ir_split[2]; ir_split[2] = NULL; }
      ir_split[2] = new IntegrationRule(3*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[2], ir_split[2], meshOrder);

      if (ir_split[3]) { delete ir_split[3]; ir_split[3] = NULL; }
      ir_split[3] = new IntegrationRule(8*pow(dof1D, dim));
      SetupIntegrationRuleForSplitMesh(mesh_split[3], ir_split[3], meshOrder);
   }

   GetNodalValues(mesh->GetNodes(), gsl_mesh);

   MFEM_ASSERT(meshid>=0, " The ID should be greater than or equal to 0.");

   const int pts_cnt = gsl_mesh.Size()/dim,
             NEtot = pts_cnt/(int)pow(dof1D, dim);

   distfint.SetSize(pts_cnt);
   if (!gfmax)
   {
      distfint = 0.0;
   }
   else
   {
      GetNodalValues(gfmax, distfint);
   }
   u_meshid = (unsigned int)meshid;

   if (dim == 2)
   {
      unsigned nr[2] = { dof1D, dof1D };
      unsigned mr[2] = { 2*dof1D, 2*dof1D };
      double * const elx[2] =
      {
         pts_cnt == 0 ? nullptr : &gsl_mesh(0),
         pts_cnt == 0 ? nullptr : &gsl_mesh(pts_cnt)
      };
      fdataD = findptsms_setup_2(gsl_comm, elx, nr, NEtot, mr, bb_t,
                                 pts_cnt, pts_cnt, npt_max, newt_tol,
                                 &u_meshid, &distfint(0));
   }
   else
   {
      unsigned nr[3] = { dof1D, dof1D, dof1D };
      unsigned mr[3] = { 2*dof1D, 2*dof1D, 2*dof1D };
      double * const elx[3] =
      {
         pts_cnt == 0 ? nullptr : &gsl_mesh(0),
         pts_cnt == 0 ? nullptr : &gsl_mesh(pts_cnt),
         pts_cnt == 0 ? nullptr : &gsl_mesh(2*pts_cnt)
      };
      fdataD = findptsms_setup_3(gsl_comm, elx, nr, NEtot, mr, bb_t,
                                 pts_cnt, pts_cnt, npt_max, newt_tol,
                                 &u_meshid, &distfint(0));
   }
   setupflag = true;
   overset   = true;
}

void OversetFindPointsGSLIB::FindPoints(const Vector &point_pos,
                                        Array<unsigned int> &point_id,
                                        int point_pos_ordering)
{
   MFEM_VERIFY(setupflag, "Use OversetFindPointsGSLIB::Setup before "
               "finding points.");
   MFEM_VERIFY(overset, " Please setup FindPoints for overlapping grids.");
   points_cnt = point_pos.Size() / dim;
   unsigned int match = 0; // Don't find points in the mesh if point_id = mesh_id

   gsl_code.SetSize(points_cnt);
   gsl_proc.SetSize(points_cnt);
   gsl_elem.SetSize(points_cnt);
   gsl_ref.SetSize(points_cnt * dim);
   gsl_dist.SetSize(points_cnt);

   auto xvFill = [&](const double *xv_base[], unsigned xv_stride[])
   {
      for (int d = 0; d < dim; d++)
      {
         if (point_pos_ordering == Ordering::byNODES)
         {
            xv_base[d] = point_pos.GetData() + d*points_cnt;
            xv_stride[d] = sizeof(double);
         }
         else
         {
            xv_base[d] = point_pos.GetData() + d;
            xv_stride[d] = dim*sizeof(double);
         }
      }
   };
   if (dim == 2)
   {
      auto *findptsData = (gslib::findpts_data_2 *)this->fdataD;
      const double *xv_base[2];
      unsigned xv_stride[2];
      xvFill(xv_base, xv_stride);
      findptsms_2(gsl_code.GetData(), sizeof(unsigned int),
                  gsl_proc.GetData(), sizeof(unsigned int),
                  gsl_elem.GetData(), sizeof(unsigned int),
                  gsl_ref.GetData(),  sizeof(double) * dim,
                  gsl_dist.GetData(), sizeof(double),
                  xv_base,            xv_stride,
                  point_id.GetData(), sizeof(unsigned int), &match,
                  points_cnt, findptsData);
   }
   else  // dim == 3
   {
      auto *findptsData = (gslib::findpts_data_3 *)this->fdataD;
      const double *xv_base[3];
      unsigned xv_stride[3];
      xvFill(xv_base, xv_stride);
      findptsms_3(gsl_code.GetData(), sizeof(unsigned int),
                  gsl_proc.GetData(), sizeof(unsigned int),
                  gsl_elem.GetData(), sizeof(unsigned int),
                  gsl_ref.GetData(),  sizeof(double) * dim,
                  gsl_dist.GetData(), sizeof(double),
                  xv_base,            xv_stride,
                  point_id.GetData(), sizeof(unsigned int), &match,
                  points_cnt, findptsData);
   }

   // Set the element number and reference position to 0 for points not found
   for (int i = 0; i < points_cnt; i++)
   {
      if (gsl_code[i] == 2 ||
          (gsl_code[i] == 1 && gsl_dist(i) > bdr_tol))
      {
         gsl_elem[i] = 0;
         for (int d = 0; d < dim; d++) { gsl_ref(i*dim + d) = -1.; }
         gsl_code[i] = 2;
      }
   }

   // Map element number for simplices, and ref_pos from [-1,1] to [0,1] for both
   // simplices and quads.
   MapRefPosAndElemIndices();
}

void OversetFindPointsGSLIB::Interpolate(const Vector &point_pos,
                                         Array<unsigned int> &point_id,
                                         const GridFunction &field_in,
                                         Vector &field_out,
                                         int point_pos_ordering)
{
   FindPoints(point_pos, point_id, point_pos_ordering);
   Interpolate(field_in, field_out);
}


} // namespace mfem
#undef CODE_INTERNAL
#undef CODE_BORDER
#undef CODE_NOT_FOUND

#endif // MFEM_USE_GSLIB
