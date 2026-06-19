import math


def validate_pose_matrix(matrix):
    if len(matrix) != 4:
        raise ValueError("pose matrix must have 4 rows")
    rows = []
    for row in matrix:
        if len(row) != 4:
            raise ValueError("pose matrix must have 4 columns")
        rows.append([float(value) for value in row])

    bottom = rows[3]
    if any(abs(bottom[i] - [0.0, 0.0, 0.0, 1.0][i]) > 1e-6 for i in range(4)):
        raise ValueError("pose matrix must be homogeneous with bottom row [0, 0, 0, 1]")
    return rows


def normalize_quaternion(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("quaternion norm must be non-zero")
    return x / norm, y / norm, z / norm, w / norm


def rotation_matrix_to_quaternion(matrix):
    m = validate_pose_matrix(matrix)
    r00, r01, r02 = m[0][0], m[0][1], m[0][2]
    r10, r11, r12 = m[1][0], m[1][1], m[1][2]
    r20, r21, r22 = m[2][0], m[2][1], m[2][2]

    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (r21 - r12) / scale
        y = (r02 - r20) / scale
        z = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / scale
        x = 0.25 * scale
        y = (r01 + r10) / scale
        z = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / scale
        x = (r01 + r10) / scale
        y = 0.25 * scale
        z = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / scale
        x = (r02 + r20) / scale
        y = (r12 + r21) / scale
        z = 0.25 * scale

    return normalize_quaternion(x, y, z, w)


def matrix_to_pose_components(matrix):
    m = validate_pose_matrix(matrix)
    qx, qy, qz, qw = rotation_matrix_to_quaternion(m)
    return {
        "position": {
            "x": m[0][3],
            "y": m[1][3],
            "z": m[2][3],
        },
        "orientation": {
            "x": qx,
            "y": qy,
            "z": qz,
            "w": qw,
        },
    }
