import math

import torch


def solve_matrix_ode_rk4(superop_static: torch.Tensor, superop_dynamic: torch.Tensor, n_steps: int):
    """
    Solve RK4 equation at the interval 2pi.

    :param superop_static: Time independent part of Liouvillian superoperator
    :param superop_dynamic: Time dependent part of Liouvillian superoperator
    :param n_steps: Total integration steps
    :return: Tuple containing:
        - Propagator U(2pi)
        - Integral of U(phi) * sin(phi) dphi from 0 to 2pi
    """
    n = superop_static.shape[-1]
    batch_shape = superop_static.shape[:-2]
    device = superop_static.device
    dtype = superop_static.dtype

    U = torch.eye(n, device=device, dtype=dtype).expand(*batch_shape, n, n).contiguous()
    k1 = torch.empty_like(U)
    k2 = torch.empty_like(U)
    k3 = torch.empty_like(U)
    k4 = torch.empty_like(U)
    temp_U = torch.empty_like(U)
    L_t = torch.empty_like(U)

    integral = torch.zeros_like(superop_static)

    a21 = a32 = 0.5
    a43 = 1.0
    b1 = b4 = 1.0 / 6.0
    b2 = b3 = 1.0 / 3.0

    phi = 0.0
    d_phi = 2 * math.pi / n_steps

    steps = torch.arange(n_steps, device=device, dtype=torch.long)
    phi_1 = steps * d_phi
    phi_mid = phi_1 + 0.5 * d_phi
    phi_4 = (steps + 1) * d_phi

    cos1_vals = torch.cos(phi_1)
    cos_mid_vals = torch.cos(phi_mid)
    cos4_vals = torch.cos(phi_4)
    sin_mid_vals = torch.sin(phi_mid)

    for step in range(n_steps):
        cos1 = cos1_vals[step]
        cos_mid = cos_mid_vals[step]
        cos4 = cos4_vals[step]
        sin_mid = sin_mid_vals[step]
        torch.add(superop_static, superop_dynamic, alpha=cos1, out=L_t)
        torch.matmul(L_t, U, out=k1)

        torch.add(superop_static, superop_dynamic, alpha=cos_mid, out=L_t)
        torch.add(U, k1, alpha=a21 * d_phi, out=temp_U)
        torch.matmul(L_t, temp_U, out=k2)

        torch.add(superop_static, superop_dynamic, alpha=cos_mid, out=L_t)
        torch.add(U, k2, alpha=a32 * d_phi, out=temp_U)
        torch.matmul(L_t, temp_U, out=k3)

        torch.add(superop_static, superop_dynamic, alpha=cos4, out=L_t)
        torch.add(U, k3, alpha=a43 * d_phi, out=temp_U)
        torch.matmul(L_t, temp_U, out=k4)

        k1.mul_(b1)
        k2.mul_(b2)
        k3.mul_(b3)
        k4.mul_(b4)
        k1.add_(k2).add_(k3).add_(k4)
        U.add_(k1, alpha=d_phi)

        integral.add_(U, alpha=sin_mid)
        phi = phi + d_phi
    return U, integral