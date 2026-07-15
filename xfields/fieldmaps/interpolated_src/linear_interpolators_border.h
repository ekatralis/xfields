// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_LINEAR_INTERPOLATORS_BORDER_H
#define XFIELDS_LINEAR_INTERPOLATORS_BORDER_H

#include "xobjects/headers/common.h"
#include "linear_interpolators.h"


GPUFUN
double TriLinearInterpolatedFieldMap_interpolate_3d_map_scalar_border_xy(
    GPUGLMEM const double* map,
    const IndicesAndWeights iw,
    GPUGLMEM const int8_t* inside_xy // size nx*ny, 0/1
){
    if (iw.ix < 0) return 0.;

    const int64_t ix = iw.ix;
    const int64_t iy = iw.iy;
    const int64_t iz = iw.iz;

    // mask for the 4 (x,y) corners (applies to both z-planes)
    const double m00 = (inside_xy[(ix  ) + (iy  )*iw.nx] != 0) ? 1.0 : 0.0;
    const double m10 = (inside_xy[(ix+1) + (iy  )*iw.nx] != 0) ? 1.0 : 0.0;
    const double m01 = (inside_xy[(ix  ) + (iy+1)*iw.nx] != 0) ? 1.0 : 0.0;
    const double m11 = (inside_xy[(ix+1) + (iy+1)*iw.nx] != 0) ? 1.0 : 0.0;

    // apply mask to the 8 weights
    const double w000 = iw.w000 * m00;
    const double w100 = iw.w100 * m10;
    const double w010 = iw.w010 * m01;
    const double w110 = iw.w110 * m11;
    const double w001 = iw.w001 * m00;
    const double w101 = iw.w101 * m10;
    const double w011 = iw.w011 * m01;
    const double w111 = iw.w111 * m11;

    const double sumw = w000+w100+w010+w110+w001+w101+w011+w111;
    if (sumw <= 0.) return 0.;

    double val =
          w000 * map[ix   + (iy  )*iw.nx + (iz  )*iw.nx*iw.ny]
        + w100 * map[ix+1 + (iy  )*iw.nx + (iz  )*iw.nx*iw.ny]
        + w010 * map[ix   + (iy+1)*iw.nx + (iz  )*iw.nx*iw.ny]
        + w110 * map[ix+1 + (iy+1)*iw.nx + (iz  )*iw.nx*iw.ny]
        + w001 * map[ix   + (iy  )*iw.nx + (iz+1)*iw.nx*iw.ny]
        + w101 * map[ix+1 + (iy  )*iw.nx + (iz+1)*iw.nx*iw.ny]
        + w011 * map[ix   + (iy+1)*iw.nx + (iz+1)*iw.nx*iw.ny]
        + w111 * map[ix+1 + (iy+1)*iw.nx + (iz+1)*iw.nx*iw.ny];

    // renormalize (only matters when some corners got masked)
    return val / sumw;
}

GPUKERN
void TriLinearInterpolatedFieldMap_interpolate_3d_map_vector_border_xy(
    TriLinearInterpolatedFieldMapData  fmap,
    const int64_t n_points,
    GPUGLMEM const double* x,
    GPUGLMEM const double* y,
    GPUGLMEM const double* z,
    const int64_t n_quantities,
    GPUGLMEM const int8_t* buffer_mesh_quantities,
    GPUGLMEM const int64_t* offsets_mesh_quantities,
    GPUGLMEM const int8_t* inside_xy, 
    GPUGLMEM double* particles_quantities
){
    VECTORIZE_OVER(pidx, n_points);
        const IndicesAndWeights iw =
            TriLinearInterpolatedFieldMap_compute_indeces_and_weights(
                fmap, x[pidx], y[pidx], z[pidx]);

        for (int iq = 0; iq < n_quantities; iq++) {
            particles_quantities[iq*n_points + pidx] =
                TriLinearInterpolatedFieldMap_interpolate_3d_map_scalar_border_xy(
                    (GPUGLMEM const double*)(buffer_mesh_quantities + offsets_mesh_quantities[iq]),
                    iw, inside_xy);
        }
    END_VECTORIZE;
}
#endif