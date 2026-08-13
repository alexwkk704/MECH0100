"""
tensor_ops.py — rotate a stiffness tensor and recompute its direction-dependent
labels. Shared by the PointNet rotation-augmentation (Farooq's full-tensor method).

Why this exists
---------------
Random 3D rotation of a point cloud with FROZEN labels is wrong: C11/C12/C44 and
TAI are direction-dependent (fixed frame), so rotating the shape changes them.
Farooq's method: rotate the cloud AND apply the same rotation to the 4th-order
stiffness tensor, then recompute the direction-dependent labels from the rotated
tensor. The rotation-INVARIANT labels (VRH E, G, nu) are unchanged automatically.

Voigt convention (STIFFNESS): the 6x6 maps directly to the 3x3x3x3 tensor with NO
factors of 2 (those factors are for compliance/strain). Verified: identity rotation
is a no-op, and the VRH shear modulus is exactly invariant under rotation.
"""

import numpy as np

# Voigt index -> (i,j) pair.  1..6  ->  11,22,33,23,13,12
_VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def voigt_to_tensor(C6):
    """6x6 Voigt stiffness -> 3x3x3x3 4th-order tensor (minor+major symmetric)."""
    C6 = np.asarray(C6, float)
    T = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VOIGT):
        for J, (k, l) in enumerate(_VOIGT):
            v = C6[I, J]
            for a, b in {(i, j), (j, i)}:
                for c, d in {(k, l), (l, k)}:
                    T[a, b, c, d] = v
    return T


def tensor_to_voigt(T):
    """3x3x3x3 -> 6x6 Voigt (inverse of voigt_to_tensor)."""
    C6 = np.zeros((6, 6))
    for I, (i, j) in enumerate(_VOIGT):
        for J, (k, l) in enumerate(_VOIGT):
            C6[I, J] = T[i, j, k, l]
    return C6


def rotate_C(C6, R):
    """Rotate a 6x6 Voigt stiffness by 3x3 rotation R (C'_ijkl = R.R.R.R : C)."""
    T = voigt_to_tensor(C6)
    T2 = np.einsum("ai,bj,ck,dl,ijkl->abcd", R, R, R, R, T)
    return tensor_to_voigt(T2)


def random_rotation(rng):
    """Uniform-ish random proper rotation (QR of a Gaussian matrix, det=+1)."""
    A = rng.normal(size=(3, 3))
    Q, Rm = np.linalg.qr(A)
    Q *= np.sign(np.diag(Rm))          # fix QR sign ambiguity
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1                  # ensure a proper rotation (no reflection)
    return Q


def voigt_reuss_hill(C6):
    """Return (K_H, G_H): rotation-INVARIANT bulk and shear moduli (Hill average)."""
    C = np.asarray(C6, float)
    S = np.linalg.inv(C)
    Kv = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    Gv = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2])
          + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
    Kr = 1.0 / ((S[0, 0] + S[1, 1] + S[2, 2]) + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    Gr = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
                 + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    return (Kv + Kr) / 2, (Gv + Gr) / 2


def tai(C6):
    """Total Anisotropy Index: Frobenius distance to the nearest isotropic tensor
    (VRH projection), normalized. Dimensionless -> scale-invariant. Matches the
    definition in ntop_batch.tensorial_anisotropy_index."""
    C = np.asarray(C6, float)
    K, G = voigt_reuss_hill(C)
    lam, mu = K - 2 * G / 3.0, G
    C_iso = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            C_iso[i, j] = lam + (2 * mu if i == j else 0.0)
    for i in range(3, 6):
        C_iso[i, i] = mu
    return float(np.linalg.norm(C - C_iso, "fro") / np.linalg.norm(C, "fro"))


# Map a target column name -> how to recompute it from a (raw-GPa) rotated tensor.
# es_gpa normalizes the stiffness entries; TAI is a ratio (no normalization).
# Anything not listed here is rotation-INVARIANT or not tensor-derivable, so the
# caller keeps its original value.
def frame_dependent_value(target, C6_gpa, es_gpa):
    t = target.strip()
    if t == "C11_n":
        return C6_gpa[0, 0] / es_gpa
    if t == "C12_n":
        return C6_gpa[0, 1] / es_gpa
    if t == "C44_n":
        return C6_gpa[3, 3] / es_gpa
    if t == "TAI":
        return tai(C6_gpa)
    return None    # not frame-dependent (or not derivable) -> caller keeps original


if __name__ == "__main__":
    # self-test: identity is a no-op; VRH-G invariant; frame targets move
    C = np.array([[41.607, 13.445, 13.665, 0.585, -0.178, 0.613],
                  [13.445, 42.094, 13.861, 0.453, -0.816, 0.203],
                  [13.665, 13.861, 44.005, 0.212, -0.630, 0.370],
                  [0.585, 0.453, 0.212, 12.879, 0.184, -0.133],
                  [-0.178, -0.816, -0.630, 0.184, 13.474, 0.203],
                  [0.613, 0.203, 0.370, -0.133, 0.203, 13.115]])
    assert np.allclose(rotate_C(C, np.eye(3)), C, atol=1e-9)
    rng = np.random.RandomState(0)
    g0 = voigt_reuss_hill(C)[1]
    for _ in range(50):
        Cr = rotate_C(C, random_rotation(rng))
        assert abs(voigt_reuss_hill(Cr)[1] - g0) < 1e-6   # G invariant
    print("tensor_ops self-test PASS")
