from __future__ import print_function
import numpy as np

"""
This file contains utility functions to fit the free-water corrected tensor
field from DWIs.
"""

water_diffusivity = 3.0e-3

# Convention for the tensor: we always represent the tensor elements
# in this order: Dxx, Dxy, Dyy, Dxz, Dyx, Dzz
def tensor_from_iwasawa(w, out=None):
    """Returns the tensor coefficients from its Iwasawa coordinates.
    """
    if out == None:
        out = np.zeros(w.shape, dtype=np.float32)
    out[..., 0] = w[..., 0]
    out[..., 1] = w[..., 0]*w[..., 3] 
    out[..., 2] = w[..., 1] + w[..., 0]*w[..., 3]*w[..., 3]
    out[..., 3] = w[..., 0]*w[..., 4]
    out[..., 4] = w[..., 0]*w[..., 3]*w[..., 4] + w[..., 1]*w[..., 5]
    out[..., 5] = w[..., 2] + w[..., 0]*w[..., 4]*w[..., 4] \
                + w[..., 1]*w[..., 5]*w[..., 5]
    return out


def iwasawa_from_tensor(d, out=None):
    """
    Computes the iwasawa coefficients from the upper triangular tensor
    coefficients.
    """
    if out == None:
        out = np.zeros(d.shape, dtype=np.float32)
    out[..., 0] = d[..., 0]  # w_1 = D_xx
    out[..., 1] = np.where(d[..., 0] == 0, d[..., 2], 
                           d[..., 2] - d[..., 1] * d[..., 1] / d[..., 0])
    out[..., 3] = np.where(d[..., 0] == 0, 0, d[..., 1] / d[..., 0])
    out[..., 4] = np.where(d[..., 0] == 0, 0, d[..., 3] / d[..., 0])

    denominator = d[..., 0]*d[..., 2] - d[..., 1]*d[..., 1]
    mask1 = np.logical_and(d[..., 0] != 0, denominator != 0)
    mask2 = np.logical_and(d[..., 0] == 0, d[..., 2] != 0)
    out[mask1, 5] = \
        (d[mask1, 0]*d[mask1, 4] - d[mask1, 1]*d[mask1, 3]) / denominator[mask1]
    out[mask2, 5] = d[mask2, 4] / d[mask2, 2]

    out[..., 2] = d[..., 5] - out[..., 0]*out[..., 4]*out[..., 4] \
                - out[..., 1]*out[..., 5]*out[..., 5]
    return out


def clip_tensor_evals(evals, evecs, l_min, l_max, out=None):
    """
    Clip tensor eigenvalues. Spectral elements of the tensor are passed 
    as input.
    """
    np.clip(evals, l_min, l_max, evals)
    tensor_shape = list(evals.shape[:-1])
    tensor_shape.append(6)
    if out==None:
        out = np.zeros(tensor_shape, dtype=np.float32)
    out[..., 0] = (evecs[..., 0, :] * evals[..., :] * evecs[..., 0, :]).sum(-1)
    out[..., 1] = (evecs[..., 1, :] * evals[..., :] * evecs[..., 0, :]).sum(-1)
    out[..., 2] = (evecs[..., 1, :] * evals[..., :] * evecs[..., 1, :]).sum(-1)
    out[..., 3] = (evecs[..., 2, :] * evals[..., :] * evecs[..., 0, :]).sum(-1)
    out[..., 4] = (evecs[..., 2, :] * evals[..., :] * evecs[..., 1, :]).sum(-1)
    out[..., 5] = (evecs[..., 2, :] * evals[..., :] * evecs[..., 2, :]).sum(-1)
    return out


def grad_tensor_iwasawa(w, out=None):
    """
    Returns the jacobian of the tensor (Dxx, ...) with respect to Iwasawa 
    coefficients.
    """
    shape_img = list(w.shape[:-1])
    shape_jacobian = shape_img
    shape_jacobian.extend([6, 6])
    if out == None:
        out = np.zeros(shape_jacobian)
    out[..., 0, 0] = 1
    out[..., 0, 1] = w[..., 3]
    out[..., 0, 2] = w[..., 3]*w[..., 3]
    out[..., 0, 3] = w[..., 4]
    out[..., 0, 4] = w[..., 3]*w[..., 4]
    out[..., 0, 5] = w[..., 4]*w[..., 4]
    out[..., 1, 2] = 1
    out[..., 1, 4] = w[..., 5]
    out[..., 1, 5] = w[..., 5]*w[..., 5]
    out[..., 2, 5] = 1
    out[..., 3, 1] = w[..., 0]
    out[..., 3, 2] = 2*w[..., 0]*w[..., 3]
    out[..., 3, 4] = w[..., 0]*w[..., 4]
    out[..., 4, 3] = w[..., 0]
    out[..., 4, 4] = w[..., 0]*w[..., 3]
    out[..., 4, 5] = 2*w[..., 0]*w[..., 4]
    out[..., 5, 4] = w[..., 1]
    out[..., 5, 5] = 2*w[..., 1]*w[..., 5]
    return out


def grad_data_fit_tensor(tensor, signal, design_matrix, bvals, 
                         volume_fraction):
    """
    Returns the gradient of the data fit term with respect to native tensor 
    coordinates.
    """
    predicted_signal_t = np.exp(np.dot(tensor, design_matrix.T)) 
    predicted_signal_w = np.exp(-bvals * water_diffusivity)[np.newaxis, :]
    predicted_signal = volume_fraction[..., np.newaxis] * predicted_signal_t \
        + (1-volume_fraction[..., np.newaxis]) * predicted_signal_w

    a_k = (predicted_signal - signal) * predicted_signal_t
    a_k = a_k[..., np.newaxis, :]
    a_k = a_k * design_matrix.T

    return a_k.sum(-1), predicted_signal_t, predicted_signal_w


def grad_data_fit(tensor_iwasawa, signal, design_matrix, bvals, 
                  volume_fraction):
    """
    Returns the gradient of the data fit term with respect to Iwasawa 
    coordinates.
    """
    tensor = tensor_from_iwasawa(tensor_iwasawa)

    grad_data_tensor, predicted_signal_t, predicted_signal_w = \
        grad_data_fit_tensor(tensor, signal, design_matrix, bvals, 
                             volume_fraction)

    grad_tensor = grad_tensor_iwasawa(tensor_iwasawa)
    
    return ((grad_data_tensor[..., np.newaxis] * grad_tensor).sum(-2),
            predicted_signal_t, 
            predicted_signal_w)


def spatial_gradient(vector_field, out=None):
    """
    Given a vector field, computes the spatial gradient.
    """
    dim_x, dim_y, dim_z, dim_vector = vector_field.shape
    if out == None:
        out = np.zeros((dim_x, dim_y, dim_z, dim_vector, 3))
    out[1:-1, :, :, :, 0] = vector_field[2:] - vector_field[:-2]
    out[:, 1:-1, :, :, 1] = vector_field[:, 2:] - vector_field[:, :-2]
    out[:, :, 1:-1, :, 2] = vector_field[:, :, 2:] - vector_field[:, :, :-2]
    return out


def _intermediate_step(w, spatial_grad_tensor, inv_gamma, det_gamma):
    """
    Computes an intermediate step necessary for the computation of
    grad_reg_constraint.
    """
    res = np.zeros((3, 6))
    for mu in range(3):
        for nu in range(3):
            res[mu] += inv_gamma[mu, nu] * spatial_grad_tensor[:, mu]
    return np.sqrt(det_gamma) * res


def induced_metric(h, spatial_grad_tensor, x, y, z):
    """
    Computes the induced metric.
    """
    gamma = np.eye(3)
    for mu in range(3):
        for nu in range(3):
            gamma[mu, nu] += np.dot(np.dot(h, spatial_grad_tensor[x, y, z, :, mu]), 
                  spatial_grad_tensor[x, y, z, :, nu])
    return gamma


def _second_term(w, spatial_grad_tensor, inv_gamma):
    """
    Returns the spatial regularization part of the gradient descent scheme.  NB:
    the tensor_iwasawa should be given in (dim_x, dim_y, dim_z, 6) shape.
    """
    grad = np.zeros(6)
    Gamma = christoffel_symbols(w)
    for i, j, k in zip(_indices_christoffel_i, _indices_christoffel_j, _indices_christoffel_k):
        grad[i] += Gamma[i, j, k] * \
                   np.dot(np.dot(inv_gamma, spatial_grad_tensor[j]), 
                          spatial_grad_tensor[k])
    return grad


def grad_spatial_reg(tensor_iwasawa, mask=None):
    """
    """
    dim_x, dim_y, dim_z, _ = tensor_iwasawa.shape
    if mask == None:
        mask = np.ones((dim_x, dim_y, dim_z), dtype=np.bool)
    grad = np.zeros((dim_x, dim_y, dim_z, 6))
    first_term = np.zeros((dim_x, dim_y, dim_z, 3, 6))
    spatial_grad_tensor = spatial_gradient(tensor_iwasawa)
    print("Max gradient intensity: {}".format(np.max(np.sqrt((spatial_grad_tensor ** 2).sum(-1)))))
    det_gammas = np.zeros((dim_x, dim_y, dim_z))
    for x in range(dim_x):
        print("\tx={0:d}...".format(x))
        for y in range(dim_y):
            for z in range(dim_z):
                if mask[x, y, z]:
                    w = tensor_iwasawa[x, y, z]
                    h = feature_metric(w)
                    gamma = induced_metric(h, spatial_grad_tensor, x, y, z)
                    det_gamma = np.linalg.det(gamma)
                    if det_gamma == 0 or np.isnan(det_gamma):
                        det_gamma = 0
                        continue
                    inv_gamma = np.linalg.inv(gamma)
                    det_gammas[x, y, z] = det_gamma
                    first_term[x, y, z] = _intermediate_step(w, 
                                            spatial_grad_tensor[x, y, z],
                                            inv_gamma, det_gamma)
                    grad[x, y, z] = _second_term(w, spatial_grad_tensor[x, y, z], inv_gamma)
    grad[1:-1] += (first_term[2:, :, :, 0] - first_term[:-2, :, :, 0]) / \
                  np.sqrt(det_gammas[1:-1,:,:,None])
    grad[:, 1:-1] += (first_term[:, 2:, :, 1] - first_term[:, :-2, :, 1]) / \
                     np.sqrt(det_gammas[:,1:-1,:,None])
    grad[:, :, 1:-1] += (first_term[:, :, 2:, 2] - first_term[:, :, :-2, 2]) / \
                        np.sqrt(det_gammas[:,:,1:-1,None])
    return grad, det_gammas


_indices_christoffel_i = [0,0,0,0,0,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5]
_indices_christoffel_j = [0,3,3,4,4,1,3,5,2,3,3,4,4,5,0,1,3,3,3,4,5,5,0,1,2,2,3,3,3,4,4,4,5,5,1,2,3,3,4,5,5]
_indices_christoffel_k = [0,3,4,3,4,1,3,5,2,3,4,3,4,5,3,3,0,1,5,5,3,4,4,3,3,4,1,2,5,0,2,5,3,4,5,5,3,4,3,1,2]
def christoffel_symbols(w):
    """
    Christoffel symbols are computed voxel-wise.
    """
    h = feature_metric(w)
    inv_h = np.linalg.inv(h)
    dh = partial_feature_metric(w)
    Gamma = np.zeros((6, 6, 6))
    for i, j, k in zip(_indices_christoffel_i, _indices_christoffel_j, _indices_christoffel_k):
        for l in range(6):
            Gamma[i, j, k] += 0.5 * inv_h[i, l] * \
                              (dh[j, l, k] + dh[k, j, l] - dh[l, j, k])
    return Gamma


def feature_metric(w):
    """
    Returns the feature metric, computed voxel-wise.
    """
    h = np.zeros((6, 6))
    h[0, 0] = 1 / (w[0]*w[0])
    h[1, 1] = 1 / (w[1]*w[1])
    h[2, 2] = 1 / (w[2]*w[2])
    h[3, 3] = 2*w[0]*(w[2] + w[1]*w[5]*w[5]) / (w[1]*w[2])
    h[4, 4] = 2*w[0] / w[2]
    h[5, 5] = 2*w[1] / w[2]
    h[4, 3] = h[3, 4] = -2*w[0]*w[5] / w[2]
    return h


def partial_feature_metric(w):
    """
    Returns the partial derivatives of the feature metric, computed voxel-wise.
    partial_h[i,j,k] contains partial_i h[j,k]
    """
    partial_h = np.zeros((6, 6, 6))
    partial_h[0, 0, 0] = -2 / (w[0]*w[0]*w[0])
    partial_h[0, 3, 3] = 2*(w[2] + w[1]*w[5]*w[5]) / (w[1]*w[2])
    partial_h[0, 3, 4] = partial_h[0, 4, 3] = -2*w[5]/w[2]
    partial_h[0, 4, 4] = 2/w[2]
    partial_h[1, 1, 1] = -2 / (w[1]*w[1]*w[1])
    partial_h[1, 3, 3] = -2*w[0] / (w[1]*w[1])
    partial_h[1, 5, 5] = 2/w[2]
    partial_h[2, 2, 2] = -2 / (w[2]*w[2]*w[2])
    partial_h[2, 3, 3] = -2*w[0]*w[5]*w[5] / (w[2]*w[2])
    partial_h[2, 3, 4] = partial_h[2, 4, 3] = 2*w[0]*w[5] / (w[2]*w[2])
    partial_h[2, 4, 4] = -2*w[0] / (w[2]*w[2])
    partial_h[2, 5, 5] = -2*w[1] / (w[2]*w[2])
    partial_h[5, 3, 3] = 4*w[0]*w[5] / w[2]
    partial_h[5, 3, 4] = partial_h[5, 4, 3] = -2*w[0] / w[2]
    return partial_h
