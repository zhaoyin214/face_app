# 关键点凸包
def draw_convex_hull(img, points, color):
    """mask"""
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color=color)

# 蒙板
def get_face_mask(img, landmarks):
    """face mask"""

    imgMask = np.zeros(img.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(imgMask,
                              landmarks[group],
                              color=1)

    imgMask = np.array([imgMask, imgMask, imgMask]).transpose((1, 2, 0))

    imgMask = (cv2.GaussianBlur(imgMask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    imgMask = cv2.GaussianBlur(imgMask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return imgMask

# 普氏分析
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    # 平移
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    # 归一化
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    # 旋转
    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

# 仿射变换 - 平移、缩放、旋转
def warp_im(img, M, dshape):
    output_im = np.zeros(dshape, dtype=img.dtype)
    cv2.warpAffine(img,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
